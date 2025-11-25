import argparse
from mmseg.apis import init_model
from mmseg.models import build_segmentor
import torch
from mmengine.config import Config


def main():
    parser = argparse.ArgumentParser(description='Convert MMSeg model to ONNX')
    # 你的配置文件路径（按实际路径修改）
    parser.add_argument('--config', default='configs/fcn/uav_crack_fcn_min.py', type=str)
    # 训练好的.pth权重路径
    parser.add_argument('--checkpoint',
                        default='C:/Users/86198/PycharmProjects/mmsegmentation/outputs/uav_crack_unet_optimized'
                                '/best_mIoU_iter_2250.pth',
                        type=str)    # 输出ONNX模型路径
    parser.add_argument('--output', default='uav_crack_model.onnx', type=str)
    # 验证转换是否正确（可选，建议开启）
    parser.add_argument('--verify', action='store_true', default=True)
    args = parser.parse_args()

    # 加载配置和模型
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device='cpu')  # CPU环境即可转换
    model.eval()  # 切换到评估模式

    # 构造模拟输入（匹配你的模型输入尺寸672×384，3通道）
    dummy_input = torch.randn(1, 3, 384, 672)  # 格式：(batch, channel, height, width)

    # 转换ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,  # 固定尺寸，避免推理兼容问题
        opset_version=11  # 兼容主流推理框架
    )
    print(f"ONNX模型已保存到：{args.output}")

    # 验证转换正确性（对比PyTorch和ONNX输出）
    if args.verify:
        import onnx
        import onnxruntime as ort
        # 加载ONNX并检查有效性
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        # 构造ONNX运行时
        ort_session = ort.InferenceSession(args.output)
        # 运行PyTorch推理
        with torch.no_grad():
            pytorch_output = model(dummy_input)[0]  # 模型输出格式适配
        # 运行ONNX推理
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        # 对比输出误差（允许微小差异）
        diff = torch.abs(torch.tensor(onnx_output) - pytorch_output).max()
        print(f"PyTorch与ONNX输出最大误差：{diff.item()}")
        if diff.item() < 1e-3:
            print("模型转换验证通过！")
        else:
            print("模型转换存在误差，请检查配置！")


if __name__ == '__main__':
    main()
