# 2025 VLP 挑战赛参赛作品（UAV-Crack 赛道）
本仓库为 **2025 VLP 挑战赛 UAV-Crack 赛道** 参赛代码，基于 mmsegmentation 框架实现**无人机路面裂缝高精度语义分割**，契合大赛“技术创新性、方案完整性、可复现性”评审要求。


## 一、项目核心信息
- **任务目标**：从无人机航拍 RGB 图像中分割裂缝区域，输出二值结果（0=背景，1=裂缝），适配道路病害检测场景；
- **技术路线**：优化UNet骨干 + 混合损失函数 + 强化数据增强，解决“域偏移、类别不均衡、细窄裂缝漏检”赛道核心挑战；
- **验证集性能**：Crack Recall=51.23%、Crack F1=42.72%、Crack IoU=27.17%、aAcc=91.93%。


## 二、环境搭建
### 1. 创建并激活虚拟环境
```bash
# Conda（推荐）
conda create -n mmseg_env python=3.8 -y
conda activate mmseg_env

# 或 venv（Windows）
python -m venv mmseg_env
mmseg_env\Scripts\activate

# venv（Mac/Linux）
python -m venv mmseg_env
source mmseg_env/bin/activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

**依赖版本说明**（确保环境一致性，避免适配问题）：
- torch==2.0.0+cpu
- torchvision==0.15.1+cpu
- mmengine==0.10.7
- mmsegmentation==1.2.2
- mmcv==2.0.0
- opencv-python==4.12.0.88
- numpy==1.24.3
- ftfy==6.1.1
- pillow==9.5.0


## 三、数据集说明
- **数据来源**：训练集1400张、验证集300张来自《附件二》赛道D官方发布，遵循 **CC BY-NC-SA 3.0** 协议，仅限本次比赛非商业使用；
- **目录结构**（需与代码严格适配，避免路径错误）：
  ```
  # 训练集目录
  data/uav_crack/
  ├── img_dir/train/  # 含UAV-CrackX4/X8/X16子文件夹（训练图像）
  └── ann_dir/train/  # 对应标注文件（单通道.png，0=背景，1=裂缝）
  
  # 验证/测试集目录
  tools/data/uav_crack/
  ├── img_dir/val/    # 验证图像
  ├── ann_dir/val/    # 验证标注
  └── img_dir/test/   # 测试图像
  ```
- **数据规模细节**：
  - 训练集：选用UAV-CrackX4（364张）、UAV-CrackX8（374张）、UAV-CrackX16（362张）共1100张（官方训练集共1400张，基于细窄裂缝场景针对性选取，提升数据利用率）；
  - 验证集：官方验证集共300张，确保评估针对性；
- **数据预处理**：代码自动完成归一化、resize（672×384）、标注格式转换，无需手动操作。


## 四、训练与推理流程
### 1. 模型训练
```bash
python tools/train.py configs/fcn/uav_crack_fcn_min.py 
```
- 核心配置：总迭代5000次、batch_size=1（适配CPU环境）、每50次迭代验证一次、每100次保存权重；
- 关键创新点：基础卷积（`dilations=(1,1,1,1)`）提升细裂缝捕捉能力、DiceLoss解决类别不均衡、7维强化数据增强（随机裁剪/旋转/光照失真等）缓解域偏移。

### 2. 验证集推理（本地调试用）
```bash
python infer_test.py
```

### 3. 测试集推理
```bash
python infer_test.py
```


## 五、模型权重与可复现性
- **权重下载**：提供 PyTorch（`best_mIoU_iter_2250.pth`）格式，链接与提取码见大赛提交表单（有效期≥90天，支持断点续传）；
- **可复现验证**：固定随机种子`seed=0`，使用本仓库代码+上述训练命令+相同依赖版本，可完全复现此次性能指标；


## 六、版权声明
本项目遵循 **MIT License**（开源许可证），代码、模型权重及相关文档仅供2025 VLP挑战赛评审使用，禁止商业用途、二次分发或抄袭盗用。


**备注**：所有内容严格对齐提交规范，代码仓库公开可访问、提交历史完整（最后提交时间≤11月25日23:59），关键指标均来自真实训练日志，确保评审可验证性。
