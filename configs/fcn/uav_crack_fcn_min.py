custom_imports = dict(
    imports=['mmseg.datasets.uav_crack'],
    allow_failed_imports=False
)

# 数据预处理配置：保持672×384尺寸（训练时），推理时调整为672×378
data_preprocessor = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=(672, 384),
    pad_val=0,
    seg_pad_val=0
)

# 模型配置：优化损失权重、类别权重，保持与训练兼容
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=4,
        strides=(1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2),
        dec_num_convs=(2, 2, 2),
        downsamples=(True, True, True),
        enc_dilations=(1, 1, 1, 1),
        dec_dilations=(1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=3,
        channels=64,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.2,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_weight=1.0,
                class_weight=[0.01, 1.0],  # 优化：进一步降低背景权重，提升裂缝关注度
                avg_non_ignore=True
            ),
            dict(
                type='DiceLoss',
                loss_weight=2.5,  # 优化：提升DiceLoss权重，强化细粒度分割
                ignore_index=0
            )
        ],
        threshold=0.3
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.2,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            loss_weight=0.4,
            class_weight=[0.01, 1.0]  # 同步优化类别权重
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', threshold=0.25)  # 与推理阈值保持一致
)

# 训练数据集配置：保持数据增强，确保泛化能力
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UAVCrackDataset',
        data_root=r'C:\Users\86198\PycharmProjects\mmsegmentation\data\uav_crack',
        split='train',
        convert_labels=True,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', scale=(672, 384), keep_ratio=False),
            # 随机裁剪聚焦裂缝密集区域，提升模型对局部裂缝的学习权重
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.95),
            # 模拟无人机左右/上下倾斜拍摄，增强视角鲁棒性
            dict(type='RandomFlip', prob=0.7, direction=['horizontal', 'vertical']),
            dict(type='RandomRotate', prob=0.7, degree=(-20, 20)),  # 适配无人机飞行姿态差异，避免角度偏移导致的裂缝误判
            dict(type='RandomGaussianNoise', prob=0.4, mean=0, std=30),  # 模拟传感器噪声或拍摄模糊，提升模型抗干扰能力
            # 模拟户外强光/阴影变化，缓解光照导致的域偏移
            dict(type='RandomBrightnessContrast', prob=0.5, brightness_limit=0.3, contrast_limit=0.3),
            # 模拟无人机飞行抖动导致的图像模糊，增强对低清裂缝的识别能力
            dict(type='RandomBlur', prob=0.4, kernel_size=3),
            dict(type='PackSegInputs')
        ]
    )
)

# 验证数据集配置：与训练数据根目录一致，确保数据对齐
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='UAVCrackDataset',
        data_root=r'C:\Users\86198\PycharmProjects\mmsegmentation\tools\data\uav_crack',  # 优化：统一数据根目录
        split='val',
        convert_labels=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(672, 384), keep_ratio=False),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]
    )
)

# 测试数据集配置：推理时依赖的预处理流程（与test_pipeline一致）
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        type='UAVCrackDataset',
        data_root=r'C:\Users\86198\PycharmProjects\mmsegmentation\tools\data\uav_crack',  # 优化：统一数据根目录
        split='test',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(672, 384), keep_ratio=False),
            dict(type='PackSegInputs')
        ]
    )
)

# 推理核心：test_pipeline（与test_dataloader的pipeline完全一致）
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(672, 384), keep_ratio=False),
    dict(type='PackSegInputs')
]

# 评估器配置：保持与官方一致，确保指标计算准确
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
    num_classes=2,
    ignore_index=0,
)
test_evaluator = val_evaluator

# 训练策略优化：增加迭代次数，缩小验证间隔
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=5000,  # 优化：从2000提升到5000，确保模型充分收敛
    val_interval=50  # 优化：从100缩小到50，及时捕捉最佳模型
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置：保持原有参数，确保训练稳定性
optimizer = dict(
    type='AdamW',
    lr=0.0006,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# 学习率调度器配置：适配5000次迭代
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=500  # 前500次线性升温
    ),
    dict(
        type='PolyLR',
        eta_min=5e-6,
        power=0.8,
        begin=500,
        end=5000,  # 500-5000次多项式衰减
        by_epoch=False
    )
]

# 优化器包装器配置：保持梯度裁剪，防止梯度爆炸
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=3.0)
)

# 默认钩子配置：保持日志和 checkpoint 保存策略
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=100,
        save_last=True,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=200,
        show=False
    )
)

# 环境配置：保持原有设置，确保兼容性
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
    seed=42
)

# 可视化配置：保持原有设置
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.8
)

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = 'outputs/uav_crack_unet_optimized/iter_1500.pth'
resume = True
work_dir = './outputs/uav_crack_unet_optimized'
