import os
import os.path as osp
import numpy as np
import torch
import random
import cv2
from mmcv import BaseTransform

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS


# ==============================================================================
# === 1. 随机Gamma校正 ===
# ==============================================================================
@TRANSFORMS.register_module()
class RandomGamma(BaseTransform):
    """随机Gamma校正"""

    def __init__(self, prob=0.5, gamma_range=(0.8, 1.2)):
        self.prob = prob
        self.gamma_range = gamma_range
        if not isinstance(gamma_range, (list, tuple)) or len(gamma_range) != 2:
            raise ValueError(f"'gamma_range' 必须是一个包含两个元素的列表或元组，当前为 {gamma_range}")
        self.min_gamma, self.max_gamma = gamma_range

    def transform(self, results):
        img = results['img']
        if random.random() < self.prob:
            img = img.astype(np.float32) / 255.0
            img = np.power(img, random.uniform(self.min_gamma, self.max_gamma))
            img = (img * 255.0).astype(np.uint8)
        results['img'] = img
        return results


# ==============================================================================
# === 2. 随机弹性形变 ===
# ==============================================================================
@TRANSFORMS.register_module()
class RandomElasticDeformation(BaseTransform):
    """随机弹性形变"""

    def __init__(self, prob=0.5, alpha=100, sigma=10, alpha_affine=50):
        self.prob = prob
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine

    def transform(self, results):
        img = results['img']
        seg_map = results.get('gt_seg_map', None)

        if random.random() < self.prob:
            height, width = img.shape[:2]
            img = img.astype(np.float32)
            seg_map = seg_map.astype(np.float32) if seg_map is not None else None

            # 随机仿射变换
            center = (width / 2, height / 2)
            angle = random.uniform(-self.alpha_affine, self.alpha_affine)
            scale = random.uniform(1 - self.alpha_affine / 100, 1 + self.alpha_affine / 100)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REFLECT)
            if seg_map is not None:
                seg_map = cv2.warpAffine(seg_map, M, (width, height), borderMode=cv2.BORDER_REFLECT)

            # 随机弹性形变场
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            dx = cv2.GaussianBlur((np.random.rand(height, width) * 2 - 1), (self.sigma * 2 + 1, self.sigma * 2 + 1),
                                  self.sigma) * self.alpha
            dy = cv2.GaussianBlur((np.random.rand(height, width) * 2 - 1), (self.sigma * 2 + 1, self.sigma * 2 + 1),
                                  self.sigma) * self.alpha

            # 应用形变
            map_x, map_y = (grid_x + dx).astype(np.float32), (grid_y + dy).astype(np.float32)
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if seg_map is not None:
                seg_map = cv2.remap(seg_map, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

            # 转换回 uint8
            img = img.astype(np.uint8)
            seg_map = seg_map.astype(np.uint8) if seg_map is not None else None

        results['img'] = img
        if seg_map is not None:
            results['gt_seg_map'] = seg_map
        return results


# ==============================================================================
# === 3. 新增：统一的标签转换变换 ===
# ==============================================================================
@TRANSFORMS.register_module()
class ConvertLabels(BaseTransform):
    """将标签图中所有大于0的值转换为1，其余为0。"""

    def __init__(self):
        pass

    def transform(self, results):
        if 'gt_seg_map' in results:
            seg_map = results['gt_seg_map']
            # 将所有大于0的像素值设为1，其余设为0
            seg_map = np.where(seg_map > 0, 1, 0).astype(seg_map.dtype)
            results['gt_seg_map'] = seg_map
        return results


# ==============================================================================
# === 4. UAV裂纹数据集定义 ===
# ==============================================================================
@DATASETS.register_module()
class UAVCrackDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'Crack'),
        palette=[[0, 0, 0], [255, 0, 0]]
    )

    def __init__(self,
                 data_root,
                 split=None,
                 data_prefix=None,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 convert_labels=True,  # 此参数现在仅作为兼容保留
                 **kwargs):

        # 处理数据路径
        if split is not None:
            img_path = osp.join('img_dir', split)
            seg_map_path = osp.join('ann_dir', split)
            data_prefix = dict(img_path=img_path, seg_map_path=seg_map_path)
        elif data_prefix is None:
            data_prefix = dict(img_path='', seg_map_path='')
        elif isinstance(data_prefix, str):
            data_prefix = dict(img_path=data_prefix, seg_map_path=data_prefix)

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )

        self.convert_labels = convert_labels  # 不再使用，但保留以避免配置错误
        print(f"✅ UAVCrackDataset 初始化成功！")
        print(f"  - 数据根目录: {data_root}")
        print(f"  - Split: {split}")

    def __getitem__(self, idx):
        try:
            # 调用父类方法获取原始数据，不做额外处理
            results = super().__getitem__(idx)
            return results
        except Exception as e:
            print(f"❌ 获取数据项 {idx} 时出错: {e}")
            raise
