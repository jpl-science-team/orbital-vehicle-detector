import os
import shutil
from pathlib import Path
from PIL import Image

def create_resolution_dataset(src_img_dir, src_lbl_dir, output_base_dir, resolutions):
    """
    Resizes test images to specific resolutions and copies corresponding labels.
    """
    src_img_dir = Path(src_img_dir)
    src_lbl_dir = Path(src_lbl_dir)
    output_base_dir = Path(output_base_dir)
    
    # Supported image extensions
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # Find all images in source directory
    img_files = [f for f in src_img_dir.iterdir() if f.suffix.lower() in img_extensions]
    print(f"Found {len(img_files)} images in {src_img_dir}")

    for res in resolutions:
        print(f"\nProcessing resolution: {res}x{res}...")
        
        # Setup new directory paths
        res_dir = output_base_dir / f"test_study_{res}"
        new_img_dir = res_dir / "images"
        new_lbl_dir = res_dir / "labels"
        
        new_img_dir.mkdir(parents=True, exist_ok=True)
        new_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in img_files:
            # 1. Resize and save image
            try:
                with Image.open(img_path) as img:
                    # LANCZOS is excellent for downsampling quality
                    img_resized = img.resize((res, res), Image.Resampling.LANCZOS)
                    img_resized.save(new_img_dir / img_path.name)
            except Exception as e:
                print(f"Failed to process image {img_path.name}: {e}")
                continue
                
            # 2. Find and copy corresponding label file
            lbl_name = img_path.stem + ".txt"
            src_lbl_path = src_lbl_dir / lbl_name
            
            if src_lbl_path.exists():
                shutil.copy(src_lbl_path, new_lbl_dir / lbl_name)
            else:
                # Create an empty label file if it's a background image
                open(new_lbl_dir / lbl_name, 'a').close()

        print(f"Saved {res}x{res} dataset to: {res_dir}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Point these to your original test dataset folders
    SOURCE_IMAGES = "data/cowc_512/images/test"
    SOURCE_LABELS = "data/cowc_512/labels/test"
    
    # Where you want the new test sets to be generated
    OUTPUT_DIRECTORY = "data/resolution_study"
    
    # Target resolutions
    TARGET_RESOLUTIONS = [512, 256, 128]
    # ---------------------
    
    create_resolution_dataset(SOURCE_IMAGES, SOURCE_LABELS, OUTPUT_DIRECTORY, TARGET_RESOLUTIONS)