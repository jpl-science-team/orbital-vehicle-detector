angle_cfg = dict(start_angle=0, width_longer=True)
angle_factor = 3.141592653589793
auto_scale_lr = dict(base_batch_size=8, enable=False)
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=0.1)
backend_args = None
custom_hooks = [
    dict(
        ema_type='mmdet.models.layers.ema.ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        type='mmengine.hooks.ema_hook.EMAHook',
        update_buffers=True),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmrotate.visualization.local_visualizer',
    ])
custom_keys = dict({
    'backbone':
    dict(lr_mult=0.1),
    'backbone.layer1.0.bn':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer1.0.downsample.1':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer1.1.bn':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer1.1.downsample.1':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer2.0.bn':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer2.0.downsample.2':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer2.1.bn':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer2.1.downsample.2':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer3.0.bn':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer3.0.downsample.2':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer3.1.bn':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer3.1.downsample.2':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer4.0.bn':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer4.0.downsample.2':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer4.1.bn':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.layer4.1.downsample.2':
    dict(decay_mult=0.0, lr_mult=0.1)
})
data_root = 'data/cowc_512_detr/'
dataset_type = 'DOTADataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=99999, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'ai4rs'
downsample_norm_idx_list = (
    2,
    3,
    3,
    3,
)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 72
model = dict(
    bbox_head=dict(num_classes=1, type='RotatedRTDETRHead'),
    type='RotatedRTDETR')
num_blocks_list = (
    2,
    2,
    2,
    2,
)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(
        lr=0.0001, type='torch.optim.adamw.AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys=dict({
            'backbone':
            dict(lr_mult=0.1),
            'backbone.layer1.0.bn':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer1.0.downsample.1':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer1.1.bn':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer1.1.downsample.1':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer2.0.bn':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer2.0.downsample.2':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer2.1.bn':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer2.1.downsample.2':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer3.0.bn':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer3.0.downsample.2':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer3.1.bn':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer3.1.downsample.2':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer4.0.bn':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer4.0.downsample.2':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer4.1.bn':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.layer4.1.downsample.2':
            dict(decay_mult=0.0, lr_mult=0.1)
        }),
        norm_decay_mult=0),
    type='mmengine.optim.optimizer.OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=2000,
        start_factor=0.001,
        type='mmengine.optim.scheduler.lr_scheduler.LinearLR'),
]
pretrained = 'https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rtdetr/resnet18vd_pretrained_55f5a0d6.pth'
resume = False
study_resolution = (
    512,
    512,
)
test_cfg = dict(type='mmengine.runner.loops.TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='val/annfiles/',
        data_prefix=dict(img='val/images/'),
        data_root='data/cowc_512_detr/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='mmdet.Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'),
    num_workers=4)
test_evaluator = dict(
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/Task1',
    type='DOTAMetric')
test_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='mmdet.Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_cfg = dict(max_epochs=150, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='train/annfiles/',
        data_prefix=dict(img='train/images/'),
        data_root='data/cowc_512_detr/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='mmdet.Resize'),
            dict(direction='horizontal', prob=0.5, type='mmdet.RandomFlip'),
            dict(direction='vertical', prob=0.5, type='mmdet.RandomFlip'),
            dict(angle=(
                -180,
                180,
            ), prob=1.0, type='mmdet.RandomRotate'),
            dict(
                brightness_delta=32,
                contrast_range=(
                    0.5,
                    1.5,
                ),
                hue_delta=18,
                saturation_range=(
                    0.5,
                    1.5,
                ),
                type='mmdet.PhotoMetricDistortion'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'),
    num_workers=4)
train_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='mmdet.Resize'),
    dict(direction='horizontal', prob=0.5, type='mmdet.RandomFlip'),
    dict(direction='vertical', prob=0.5, type='mmdet.RandomFlip'),
    dict(angle=(
        -180,
        180,
    ), prob=1.0, type='mmdet.RandomRotate'),
    dict(
        brightness_delta=32,
        contrast_range=(
            0.5,
            1.5,
        ),
        hue_delta=18,
        saturation_range=(
            0.5,
            1.5,
        ),
        type='mmdet.PhotoMetricDistortion'),
    dict(type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='mmengine.runner.loops.ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='val/annfiles/',
        data_prefix=dict(img='val/images/'),
        data_root='data/cowc_512_detr/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='mmdet.Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'),
    num_workers=4)
val_evaluator = dict(metric='mAP', type='DOTAMetric')
val_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            1024,
            1024,
        ),
        type='mmdet.Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmrotate.RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/cowc_rtdetr_512'
