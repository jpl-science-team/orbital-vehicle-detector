import os
import sys
import torch
from functools import partial

import mmcv
import mmrotate
from mmdet.apis import init_detector, inference_detector
from mmengine.registry import VISUALIZERS, DATASETS
from mmrotate.datasets import DOTADataset

# ---------------------------------------------------------
# GLOBAL PATCHES
# ---------------------------------------------------------
# 1. Tell PyTorch 2.6+ to trust our MMEngine checkpoint and load custom objects
torch.load = partial(torch.load, weights_only=False)

# 2. Forcefully inject the missing dataset into the active registry
DATASETS.register_module(name='DOTADataset', module=DOTADataset, force=True)


def generate_prediction_image(config_path, checkpoint_path, image_path, output_name="annotated_output.jpg", device='mps'):
    """
    Loads an MMRotate-based custom RT-DETR model, runs inference, 
    and saves the image with oriented bounding boxes drawn.
    """
    # Quick check to ensure your files exist
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return
    if not os.path.exists(config_path):
        print(f"Error: Could not find config at {config_path}")
        return
        
    # Ensure custom imports from your repo can be found
    sys.path.append(os.path.abspath('./O2-RT-DETR'))
    sys.path.append(os.path.abspath('./O2-RT-DETR/projects/rotated_rtdetr'))
    
    # 1. Initialize the detector
    print(f"Loading config '{config_path}' and model '{checkpoint_path}'...")
    model = init_detector(config_path, checkpoint_path, device=device)
    
    # 2. Run inference
    print("Running inference...")
    result = inference_detector(model, image_path)
    
    # 3. Initialize visualizer from the config
    print("Drawing and saving results...")
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    
    # 4. Load image for drawing using mmcv
    img = mmcv.imread(image_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    
    # 5. Draw and save the oriented bounding boxes
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        show=False,
        out_file=output_name,
        pred_score_thr=0.3  # Adjust this threshold to filter out low-confidence boxes
    )
    
    print(f"Success! Annotated image saved to: {output_name}")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # Target File Paths
    # ---------------------------------------------------------
    
    # Your custom config file
    MY_CONFIG = "O2-RT-DETR/projects/rotated_rtdetr/configs/cowc_rtdetr_512.py"  
    
    # The path to your PyTorch model weights
    MY_MODEL = "runs/20260715_rtdetr_obb_512/best_dota_mAP_epoch_27.pth"  
    
    # The path to the image you want to test
    MY_IMAGE = "data/Angel/images/Isub_u181_v6141_512x512.tif"  
    
    # What you want the final saved image to be called
    MY_OUTPUT_IMAGE = "predictions_drawn.jpg" 
    
    # Run the function (configured for Apple Silicon)
    generate_prediction_image(MY_CONFIG, MY_MODEL, MY_IMAGE, MY_OUTPUT_IMAGE, device='mps')