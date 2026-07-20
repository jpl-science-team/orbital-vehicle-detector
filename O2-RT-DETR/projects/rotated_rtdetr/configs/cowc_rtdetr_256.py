import sys
import os
sys.path.append(os.path.abspath('./O2-RT-DETR'))
sys.path.append(os.path.abspath('./O2-RT-DETR/projects/rotated_rtdetr'))

from mmengine.config import read_base
with read_base():
    from .o2_rtdetr_r50vd_2xb4_72e_dota import *

custom_imports = dict(
    imports=[
        'mmrotate.visualization.local_visualizer',
        'projects.rotated_rtdetr.rotated_rtdetr.rotated_rtdetr',
        'projects.rotated_rtdetr.rotated_rtdetr.rotated_rtdetr_head'
    ],
    allow_failed_imports=False
)

# OVERRIDE MODEL CLASSES
model['bbox_head']['num_classes'] = 1

# OVERRIDE DATASET METAINFO
metainfo = dict(classes=('vehicle',))  

# CONFIGURE DATALOADERS
train_dataloader['batch_size'] = 2
train_dataloader['num_workers'] = 2
train_dataloader['dataset']['data_root'] = 'data/cowc_512_degraded_dota/'
train_dataloader['dataset']['ann_file'] = 'train/annfiles/'
train_dataloader['dataset']['data_prefix'] = dict(img_path='train/images/')
train_dataloader['dataset']['metainfo'] = metainfo  

val_dataloader['batch_size'] = 2
val_dataloader['num_workers'] = 2
val_dataloader['dataset']['data_root'] = 'data/cowc_512_degraded_dota/'
val_dataloader['dataset']['ann_file'] = 'val/annfiles/'
val_dataloader['dataset']['data_prefix'] = dict(img_path='val/images/')
val_dataloader['dataset']['metainfo'] = metainfo  

# =====================================================================
# SAFELY OVERRIDE RESOLUTION TO 256x256 IN THE EXISTING PIPELINES
# =====================================================================
study_resolution = (256, 256)

# Find the Resize step in the inherited train pipeline and change only the scale
for transform in train_dataloader['dataset']['pipeline']:
    if 'Resize' in transform['type']:
        transform['scale'] = study_resolution

# Find the Resize step in the inherited val pipeline and change only the scale
for transform in val_dataloader['dataset']['pipeline']:
    if 'Resize' in transform['type']:
        transform['scale'] = study_resolution
# =====================================================================

# Copies val_dataloader configuration, including the resized pipeline
test_dataloader = val_dataloader.copy()

# OVERRIDE TRAINING SCHEDULE
train_cfg['max_epochs'] = 150
train_cfg['val_interval'] = 1

# ULTRALYTICS-STYLE CHECKPOINTING (Save only Best and Last)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,      # Deletes all older epochs, keeping only the most recent "last"
        save_best='dota/mAP',  # Independently saves and updates the "best" model
        rule='greater',
        save_last=True         # Generates a 'last_checkpoint' pointer
    )
)

visualizer['type'] = 'mmrotate.RotLocalVisualizer'
work_dir = './work_dirs/20260715_rtdetr_obb_256'

# CLEANUP (Bypasses MMEngine pickling errors)
del sys
del os