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

# OVERRIDE DATASET
train_dataloader['batch_size'] = 16
train_dataloader['num_workers'] = 4
train_dataloader['dataset']['data_root'] = 'data/cowc_512_detr/'
train_dataloader['dataset']['ann_file'] = 'train/annfiles/'
train_dataloader['dataset']['data_prefix'] = dict(img='train/images/')

val_dataloader['batch_size'] = 16
val_dataloader['num_workers'] = 4
val_dataloader['dataset']['data_root'] = 'data/cowc_512_detr/'
val_dataloader['dataset']['ann_file'] = 'val/annfiles/'
val_dataloader['dataset']['data_prefix'] = dict(img='val/images/')

test_dataloader = val_dataloader.copy()

# OVERRIDE TRAINING SCHEDULE
train_cfg['max_epochs'] = 150
train_cfg['val_interval'] = 1

visualizer['type'] = 'mmrotate.RotLocalVisualizer'
work_dir = './work_dirs/cowc_rtdetr_512'

# CLEANUP (Bypasses MMEngine pickling errors)
del sys
del os
