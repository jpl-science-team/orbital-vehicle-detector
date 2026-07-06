import os
import shutil
import math
from PIL import Image

def split_and_resize_test_dataset(base_dataset_dir, output_dir):
    """
    Targets the 'test' folder of a YOLO dataset and splits it equally into:
    - test_512 (original)
    - test_256 (bilinear downsampled)
    - test_128 (bilinear downsampled)
    """
    # Point directly to your test subsets based on your directory tree
    img_test_dir = os.path.join(base_dataset_dir, 'images', 'test')
    lbl_test_dir = os.path.join(base_dataset_dir, 'labels', 'test')
    
    if not os.path.exists(img_test_dir) or not os.path.exists(lbl_test_dir):
        raise FileNotFoundError("Could not find 'images/test' or 'labels/test' in the provided directory.")

    # Gather and sort test images
    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    all_test_images = sorted([f for f in os.listdir(img_test_dir) if f.lower().endswith(supported_exts)])
    total_imgs = len(all_test_images)
    
    print(f"Found {total_imgs} images in the test set.")
    
    # Calculate equal chunks
    chunk_size = math.ceil(total_imgs / 3)
    splits = {
        '512': (all_test_images[0:chunk_size], None),
        '256': (all_test_images[chunk_size:chunk_size*2], (256, 256)),
        '128': (all_test_images[chunk_size*2:], (128, 128))
    }
    
    # Process and build the new structure
    for category, (img_list, target_size) in splits.items():
        # Recreate the standard YOLO structure for each split variation
        cat_img_out = os.path.join(output_dir, f'test_{category}', 'images', 'test')
        cat_lbl_out = os.path.join(output_dir, f'test_{category}', 'labels', 'test')
        os.makedirs(cat_img_out, exist_ok=True)
        os.makedirs(cat_lbl_out, exist_ok=True)
        
        print(f"Generating test_{category} split with {len(img_list)} files...")
        
        for img_name in img_list:
            src_img_path = os.path.join(img_test_dir, img_name)
            dst_img_path = os.path.join(cat_img_out, img_name)
            
            # 1. Process Image
            if target_size is None:
                shutil.copy2(src_img_path, dst_img_path)
            else:
                with Image.open(src_img_path) as img:
                    resized_img = img.resize(target_size, resample=Image.BILINEAR)
                    resized_img.save(dst_img_path)
            
            # 2. Process Label (YOLO labels match perfectly across resolutions)
            base_name = os.path.splitext(img_name)[0]
            lbl_name = f"{base_name}.txt"
            src_lbl_path = os.path.join(lbl_test_dir, lbl_name)
            dst_lbl_path = os.path.join(cat_lbl_out, lbl_name)
            
            if os.path.exists(src_lbl_path):
                shutil.copy2(src_lbl_path, dst_lbl_path)
                
    print("\nProcessing complete! Your test splits are ready.")

# --- CONFIGURATION ---
# Change these paths to point to your directories
DATASET_PATH = "data/cowc_512" 
OUTPUT_PATH = "data/cowc_split_test"

if __name__ == "__main__":
    split_and_resize_test_dataset(DATASET_PATH, OUTPUT_PATH)