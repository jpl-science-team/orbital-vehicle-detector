# Inherit everything from your updated 512 base script
_base_ = ['./cowc_rtdetr_512.py']

# Override ONLY the resolution for this study leg
study_resolution = (128, 128) 

# --- MATCHED AUGMENTATIONS WITH 128 DOWN-SAMPLING ---
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    
    # Forces aggressive dynamic downsampling in RAM to 128x128
    dict(type='mmdet.Resize', scale=study_resolution, keep_ratio=True),
    
    # Mirrored YOLO Augmentations
    dict(type='mmdet.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmdet.RandomFlip', prob=0.5, direction='vertical'),
    dict(type='mmdet.RandomRotate', prob=1.0, angle=(-180, 180)), # Fixed typo here
    dict(
        type='mmdet.PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
        
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.Resize', scale=study_resolution, keep_ratio=True),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Re-apply the updated pipelines to the dataloaders
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))