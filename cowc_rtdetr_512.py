# Inherit the base ResNet-18 architecture
_base_ = ['O2-RT-DETR/configs/o2_rtdetr/o2_rtdetr_r18_8xb2-72e_dota.py']

dataset_type = 'DOTADataset'
data_root = 'data/cowc_512_dota/' 

study_resolution = (512, 512) 

# --- SAFE AERIAL AUGMENTATIONS (PERFECTLY MATCHED TO YOLO) ---
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    
    # 1. Image Sizing
    dict(type='mmdet.Resize', scale=study_resolution, keep_ratio=True),
    
    # 2. Flips (Horizontal & Vertical both set to 50% probability)
    dict(type='mmdet.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmdet.RandomFlip', prob=0.5, direction='vertical'),
    
    # 3. Full 180-Degree Rotations for OBB Orientation Stability
    dict(type='mmdet.RandomRotate', prob=1.0, angle=(-180, 180)),
    
    # 4. Color & Lighting Variation (Maps to YOLO's HSV adjustments)
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

# --- DATALOADER ASSIGNMENT ---
train_dataloader = dict(
    batch_size=16, # Matched to your YOLO script
    num_workers=4,  # Matched to your YOLO script
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annfiles/',
        data_prefix=dict(img='train/images/'),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',
        data_prefix=dict(img='val/images/'),
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader

# --- OPTIMIZATION & TRAINING SCHEDULE ---
# Force training to stop at exactly 150 epochs to match your YOLO script
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=1)

# Set the number of classes (COWC: 1 class)
model = dict(bbox_head=dict(num_classes=1))