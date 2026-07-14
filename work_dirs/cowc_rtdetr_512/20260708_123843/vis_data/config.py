angle_cfg = dict(start_angle=0, width_longer=True)
angle_factor = 3.141592653589793
auto_scale_lr = dict(base_batch_size=8, enable=False)
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
        'projects.rotated_rtdetr.rotated_rtdetr.rotated_rtdetr',
        'projects.rotated_rtdetr.rotated_rtdetr.rotated_rtdetr_head',
    ])
data_root = 'data/split_ss_dota/'
dataset_type = 'DOTADataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=99999, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'ai4rs'
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
    as_two_stage=True,
    backbone=dict(
        depth=50,
        frozen_stages=0,
        init_cfg=dict(
            checkpoint=
            'https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rtdetr/resnet50vd_ssld_v2_pretrained_d037e232.pth',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            1,
            2,
            3,
        ),
        style='pytorch',
        type='projects.rotated_rtdetr.rotated_rtdetr.ResNetV1dPaddle'),
    bbox_head=dict(
        angle_cfg=dict(start_angle=0, width_longer=True),
        angle_factor=3.141592653589793,
        loss_bbox=dict(loss_weight=5.0, type='mmdet.models.losses.L1Loss'),
        loss_cls=dict(
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0,
            type='projects.rotated_rtdetr.rotated_rtdetr.RTDETRVarifocalLoss',
            use_sigmoid=True,
            varifocal_loss_iou_type='hbox_iou'),
        loss_iou=dict(
            fun='log1p',
            loss_type='kld',
            loss_weight=2.0,
            sqrt=False,
            tau=1,
            type='ai4rs.models.losses.GDLoss'),
        num_classes=1,
        sync_cls_avg_factor=True,
        type='projects.rotated_rtdetr.rotated_rtdetr.RotatedRTDETRHead'),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        boxtype2tensor=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='mmdet.models.data_preprocessors.DetDataPreprocessor'),
    decoder=dict(
        angle_factor=3.141592653589793,
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=3),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        angle_cfg=dict(start_angle=0, width_longer=True),
        angle_factor=3.141592653589793,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5,
        noise_mode='only_xyxy'),
    encoder=dict(
        fpn_cfg=dict(
            expansion=1.0,
            in_channels=[
                256,
                256,
                256,
            ],
            norm_cfg=dict(requires_grad=True, type='BN'),
            out_channels=256,
            type='projects.rotated_rtdetr.rotated_rtdetr.RTDETRFPN'),
        in_channels=[
            256,
            256,
            256,
        ],
        layer_cfg=dict(
            ffn_cfg=dict(
                act_cfg=dict(type='GELU'),
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_encoder_layers=1,
        use_encoder_idx=[
            -1,
        ]),
    neck=dict(
        act_cfg=None,
        in_channels=[
            512,
            1024,
            2048,
        ],
        init_cfg=dict(
            a=2.23606797749979,
            distribution='uniform',
            layer='Conv2d',
            mode='fan_in',
            nonlinearity='leaky_relu',
            type='Kaiming'),
        kernel_size=1,
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_outs=3,
        out_channels=256,
        type='mmdet.models.necks.ChannelMapper'),
    num_queries=300,
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(
                    type='mmdet.models.task_modules.FocalLossCost',
                    weight=2.0),
                dict(
                    box_format='xywha',
                    type=
                    'projects.rotated_dino.rotated_dino.match_cost.ChamferCost',
                    weight=5.0),
                dict(
                    fun='log1p',
                    loss_type='kld',
                    sqrt=False,
                    tau=1,
                    type='projects.rotated_dino.rotated_dino.match_cost.GDCost',
                    weight=2.0),
            ],
            type='mmdet.models.task_modules.HungarianAssigner')),
    type='projects.rotated_rtdetr.rotated_rtdetr.RotatedRTDETR',
    with_box_refine=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(
        lr=0.0001, type='torch.optim.adamw.AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys=dict(backbone=dict(lr_mult=0.1)),
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
pretrained = 'https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rtdetr/resnet50vd_ssld_v2_pretrained_d037e232.pth'
resume = False
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
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
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
        ],
        test_mode=True,
        type='DOTADataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/Task1',
    type='DOTAMetric')
test_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
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
train_cfg = dict(
    max_epochs=150,
    type='mmengine.runner.loops.EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=16,
    dataset=dict(
        ann_file='train/annfiles/',
        data_prefix=dict(img='train/images/'),
        data_root='data/cowc_512_detr/',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(
                angle_range=180,
                prob=0.5,
                rect_obj_labels=[
                    9,
                    11,
                ],
                type='RandomRotate'),
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
            dict(type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(
        angle_range=180,
        prob=0.5,
        rect_obj_labels=[
            9,
            11,
        ],
        type='RandomRotate'),
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
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
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
        ],
        test_mode=True,
        type='DOTADataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
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
