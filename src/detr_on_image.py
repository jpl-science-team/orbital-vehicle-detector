import os
import sys
import re
import inspect
from functools import partial

# ---------------------------------------------------------
# AUTO-PATCHER (Fixes Python 3.9 typing crash in external file)
# ---------------------------------------------------------
def patch_prob_iou():
    target_file = os.path.abspath('./O2-RT-DETR/projects/rotated_rtdetr/rotated_rtdetr/prob_iou.py')
    if os.path.exists(target_file):
        with open(target_file, 'r') as f:
            content = f.read()
        if '| np.ndarray' in content:
            fixed_content = re.sub(
                r'obb1:\s*torch\.Tensor\s*\|\s*np\.ndarray,\s*obb2:\s*torch\.Tensor\s*\|\s*np\.ndarray',
                'obb1, obb2', 
                content
            )
            with open(target_file, 'w') as f:
                f.write(fixed_content)
            print("Successfully patched 'prob_iou.py' for Python 3.9 compatibility!")

patch_prob_iou()
# ---------------------------------------------------------

# Path injections placed at the top so MMEngine can resolve submodules
sys.path.append(os.path.abspath('./O2-RT-DETR'))
sys.path.append(os.path.abspath('./O2-RT-DETR/projects/rotated_rtdetr'))

import torch
import mmcv

# ---------------------------------------------------------
# GLOBAL PATCHES (MUST happen before MMDetection APIs load)
# ---------------------------------------------------------

import mmdet.structures.bbox.box_type as box_type_module

# 1. Preserving Box Registry Bypass
# Force-registers boxes while preserving existing class mappings in _box_type_to_name
_orig_register_box = box_type_module._register_box
def _force_register_box(name, box_type, *args, **kwargs):
    old_type_mappings = dict(box_type_module._box_type_to_name)
    kwargs['force'] = True
    res = _orig_register_box(name, box_type, *args, **kwargs)
    # Restore any previously registered box classes so get_box_type() never fails
    for b_type, b_info in old_type_mappings.items():
        if b_type not in box_type_module._box_type_to_name:
            box_type_module._box_type_to_name[b_type] = b_info
    return res

box_type_module._register_box = _force_register_box

# 2. The Ultimate Box CONVERTER Registry Bypass
_orig_register_converter = box_type_module._register_box_converter
def _force_register_converter(*args, **kwargs):
    kwargs['force'] = True
    return _orig_register_converter(*args, **kwargs)
box_type_module._register_box_converter = _force_register_converter

# 3. Dynamic Transform Registry Auto-Resolver
from mmengine.registry import TRANSFORMS
_orig_transforms_build = TRANSFORMS.build

def _auto_resolve_transforms_build(cfg, *args, **kwargs):
    try:
        return _orig_transforms_build(cfg, *args, **kwargs)
    except KeyError as e:
        t_name = None
        if isinstance(cfg, dict):
            t_name = cfg.get('type')
        elif hasattr(cfg, 'type'):
            t_name = getattr(cfg, 'type')
            
        if isinstance(t_name, str):
            clean_name = t_name.split('::')[-1]
            mod = None
            
            for module_path in [
                'mmdet.datasets.transforms',
                'mmrotate.datasets.transforms',
                'mmengine.dataset.transforms'
            ]:
                try:
                    imported_mod = __import__(module_path, fromlist=[clean_name])
                    if hasattr(imported_mod, clean_name):
                        mod = getattr(imported_mod, clean_name)
                        break
                except ImportError:
                    continue
            
            if mod is not None:
                TRANSFORMS.register_module(name=clean_name, module=mod, force=True)
                TRANSFORMS.register_module(name=f'ai4rs::{clean_name}', module=mod, force=True)
                TRANSFORMS.register_module(name=t_name, module=mod, force=True)
                return _orig_transforms_build(cfg, *args, **kwargs)
        raise e

TRANSFORMS.build = _auto_resolve_transforms_build

# 4. Universal DETR / DINO / RT-DETR Signature and Argument Auto-Bridge
from mmdet.models.detectors.dino import DINO
_orig_dino_forward_transformer = DINO.forward_transformer

def _patched_dino_forward_transformer(self, *args, **kwargs):
    if hasattr(self, 'decoder') and hasattr(self, 'bbox_head'):
        cls_b = getattr(self.bbox_head, 'cls_branches', None)
        reg_b = getattr(self.bbox_head, 'reg_branches', None)
        setattr(self.decoder, '_patch_cls_branches', cls_b)
        setattr(self.decoder, '_patch_reg_branches', reg_b)

        if not hasattr(self.decoder, '_forward_patched'):
            orig_decoder_forward = self.decoder.forward
            dec_sig = inspect.signature(orig_decoder_forward)
            dec_has_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in dec_sig.parameters.values())

            def _robust_decoder_forward(*d_args, **d_kwargs):
                if 'cls_branches' not in d_kwargs and 'cls_branches' in dec_sig.parameters:
                    d_kwargs['cls_branches'] = getattr(self.decoder, '_patch_cls_branches', None)
                if 'reg_branches' not in d_kwargs and 'reg_branches' in dec_sig.parameters:
                    d_kwargs['reg_branches'] = getattr(self.decoder, '_patch_reg_branches', None)
                
                if not dec_has_var:
                    valid_dec_params = set(dec_sig.parameters.keys())
                    d_kwargs = {k: v for k, v in d_kwargs.items() if k in valid_dec_params}
                return orig_decoder_forward(*d_args, **d_kwargs)

            self.decoder.forward = _robust_decoder_forward
            self.decoder._forward_patched = True

    orig_fd = self.forward_decoder
    fd_sig = inspect.signature(orig_fd)
    fd_has_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in fd_sig.parameters.values())

    if not fd_has_var:
        valid_fd_params = set(fd_sig.parameters.keys())
        def _robust_fd(*f_args, **f_kwargs):
            if 'cls_branches' in f_kwargs:
                setattr(self.decoder, '_patch_cls_branches', f_kwargs['cls_branches'])
            if 'reg_branches' in f_kwargs:
                setattr(self.decoder, '_patch_reg_branches', f_kwargs['reg_branches'])

            filtered_fd_kwargs = {k: v for k, v in f_kwargs.items() if k in valid_fd_params}
            return orig_fd(*f_args, **filtered_fd_kwargs)
        
        self.__dict__['forward_decoder'] = _robust_fd

    try:
        return _orig_dino_forward_transformer(self, *args, **kwargs)
    finally:
        self.__dict__.pop('forward_decoder', None)

DINO.forward_transformer = _patched_dino_forward_transformer

# 5. Tell PyTorch 2.6+ to trust our MMEngine checkpoint and load custom objects
torch.load = partial(torch.load, weights_only=False)

# NOW we can safely import the APIs
from mmdet.apis import init_detector, inference_detector
from mmengine.registry import VISUALIZERS
# ---------------------------------------------------------


def generate_prediction_image(config_path, checkpoint_path, image_path, output_name="annotated_output.jpg", device='cpu'):
    """
    Loads an MMRotate-based custom RT-DETR model, runs inference, 
    and saves the image with oriented bounding boxes drawn.
    """
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return
    if not os.path.exists(config_path):
        print(f"Error: Could not find config at {config_path}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find model checkpoint at {checkpoint_path}")
        return
    
    # 1. Intercept and modify the config in memory
    from mmengine.config import Config
    print(f"Loading config '{config_path}'...")
    cfg = Config.fromfile(config_path)
    
    # Safely override the dataset type and inject custom class metainfo
    if 'test_dataloader' in cfg and 'dataset' in cfg.test_dataloader:
        if cfg.test_dataloader.dataset.get('type') == 'DOTADataset':
            cfg.test_dataloader.dataset.type = 'mmrotate.DOTADataset'
        cfg.test_dataloader.dataset.metainfo = dict(classes=('vehicle',))
        
    if 'val_dataloader' in cfg and 'dataset' in cfg.val_dataloader:
        if cfg.val_dataloader.dataset.get('type') == 'DOTADataset':
            cfg.val_dataloader.dataset.type = 'mmrotate.DOTADataset'
        cfg.val_dataloader.dataset.metainfo = dict(classes=('vehicle',))

    # 2. Initialize the detector with the patched config
    print(f"Initializing model with checkpoint '{checkpoint_path}'...")
    model = init_detector(cfg, checkpoint_path, device=device)
    
    # 3. Run inference
    print("Running inference...")
    result = inference_detector(model, image_path)
    
    # 4. Initialize visualizer from the config
    print("Drawing and saving results...")
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    
    # 5. Load image for drawing using mmcv
    img = mmcv.imread(image_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    
    # 6. Draw and save the oriented bounding boxes
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        show=False,
        out_file=output_name,
        pred_score_thr=0.1  # Adjust this threshold to filter out low-confidence boxes
    )
    
    print(f"Success! Annotated image saved to: {output_name}")


if __name__ == "__main__":
    # Using the config stored in the runs folder to perfectly match the training state
    MY_CONFIG = "runs/20260715_rtdetr_obb_512/cowc_rtdetr_512.py"  
    
    # The path to your PyTorch model weights
    MY_MODEL = "runs/20260715_rtdetr_obb_512/best_dota_mAP_epoch_27.pth"  
    
    # The path to the image you want to test
    MY_IMAGE = "data/Angel/images/Isub_u181_v6141_512x512.tif"  
    
    # What you want the final saved image to be called
    MY_OUTPUT_IMAGE = "predictions_drawn.jpg" 
    
    # Run the function
    generate_prediction_image(MY_CONFIG, MY_MODEL, MY_IMAGE, MY_OUTPUT_IMAGE, device='cpu')