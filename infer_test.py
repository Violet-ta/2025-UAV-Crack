import os
import zipfile
import torch
import cv2
import numpy as np
from PIL import Image
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules

register_all_modules()


def load_models(config_path, checkpoint_paths, device):
    models = []
    for ckpt in checkpoint_paths:
        model = init_model(config_path, ckpt, device=device)
        model.to(device)
        models.append(model)
    return models


def ensemble_predict(models, img_resized, device):
    """优化：概率锐化+通道加权"""
    all_preds = []
    with torch.no_grad():
        for model in models:
            result = inference_model(model, img_resized)
            pred = result.pred_sem_seg.data.squeeze().cpu().numpy()
            # 1. 概率锐化（强化高/低概率区分度）
            pred = np.clip(pred, 0.05, 0.95)  # 缩小模糊区间
            pred = (pred - 0.5) * 1.5 + 0.5  # 拉伸概率分布
            pred = np.clip(pred, 0, 1)
            all_preds.append(pred)
    # 2. 更极端的加权（best模型权重占60%）
    weights = [0.6, 0.15, 0.15, 0.1]
    avg_pred = np.average(all_preds, axis=0, weights=weights)
    return avg_pred


def dynamic_postprocess(mask, img_shape):
    h, w = img_shape
    # 优化：分场景调整形态学核
    if h < 400 or w < 700:
        min_area = 12
        denoise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))  # 十字核更适合密集裂缝
    else:
        min_area = 8
        denoise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    # 1. 开运算去噪
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, denoise_kernel, iterations=1)
    # 2. 闭运算连接+边缘增强
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, connect_kernel, iterations=2)
    # 3. 边缘膨胀（填补细裂缝）
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.dilate(mask, kernel_edge, iterations=1)
    # 4. 面积过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
    return mask


def main():
    config_path = 'outputs/uav_crack_unet_optimized/uav_crack_fcn_min.py'
    checkpoint_paths = [
        'outputs/uav_crack_unet_optimized/best_mIoU_iter_2250.pth',
        'outputs/uav_crack_unet_optimized/iter_2175.pth',
        'outputs/uav_crack_unet_optimized/iter_2200.pth',
        'outputs/uav_crack_unet_optimized/iter_2300.pth'
    ]
    img_dir = r'/tools/data\uav_crack\img_dir\val'
    output_dir = 'result'
    target_h, target_w = 378, 672

    # 优化：动态阈值+裂缝区域优先
    def get_dynamic_thres(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        # 新增：检测图像中是否有疑似裂缝（高对比度区域）
        edge = cv2.Canny(gray, 50, 150)
        has_crack = np.sum(edge) > 1000
        if has_crack:
            # 有裂缝时降低阈值，避免漏检
            return 0.28 if brightness < 80 else 0.33
        else:
            # 无裂缝时提高阈值，减少假阳性
            return 0.35 if brightness < 80 else 0.40

    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = load_models(config_path, checkpoint_paths, device)
    print(f"🚀 加载 {len(models)} 个模型完成！使用设备：{device}")

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    print(f"📁 开始推理（共{len(img_files)}张图像）...")

    for idx, img_name in enumerate(sorted(img_files)):
        img_path = os.path.join(img_dir, img_name)
        image_id = os.path.splitext(img_name)[0]
        output_mask_path = os.path.join(output_dir, f"{image_id}.png")

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (672, 384), interpolation=cv2.INTER_LINEAR)

        avg_pred = ensemble_predict(models, img_resized, device)

        # 动态阈值（新增裂缝检测逻辑）
        dynamic_thres = get_dynamic_thres(img_rgb)
        combined_mask = (avg_pred > dynamic_thres).astype(np.uint8)

        # 动态后处理（新增边缘膨胀）
        if combined_mask.max() > 0:
            combined_mask = dynamic_postprocess(combined_mask, img_rgb.shape[:2])

        pred_final = cv2.resize(combined_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_img = Image.fromarray(pred_final, mode='L')
        mask_img.save(output_mask_path, 'PNG', compress_level=0)

        if idx % 50 == 0:
            print(f"[{idx + 1:3d}/{len(img_files)}] 生成掩码：{image_id}.png（阈值：{dynamic_thres:.2f}）")

    # 生成提交包
    zip_output_path = 'result.zip'
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.basename(file))
    print(f"✅ 最终优化提交包生成完成：{os.path.abspath(zip_output_path)}")


if __name__ == '__main__':
    main()

