import os
from pathlib import Path
from PIL import Image

def scale_up_in_place(src_dir, target_gsds=[30.0, 15.0], current_gsd=50.0):
    """
    Upscales 50 GSD images to target GSDs and saves them directly
    inside the original Angel/images and Angel/labels directories.
    """
    base_path = Path(src_dir)
    img_dir = base_path / "images"
    lbl_dir = base_path / "labels"
    
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Ensure both 'images' and 'labels' folders exist inside: {base_path}")
        
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
    
    # Grab ONLY the original images (ignoring any already upscaled ones)
    images = [
        f for f in img_dir.iterdir() 
        if f.suffix.lower() in valid_extensions 
        and not any(f"_{int(g)}gsd" in f.name for g in target_gsds)
    ]
    
    print(f"Loaded {len(images)} original {current_gsd} GSD source images.")
    
    for target_gsd in target_gsds:
        upscale_factor = current_gsd / target_gsd
        print(f"Generating scale tier -> {target_gsd} GSD (Factor: {upscale_factor:.3f}x)")
        
        for img_path in images:
            try:
                # 1. Load and upscale image
                with Image.open(img_path) as img:
                    w, h = img.size
                    new_w = int(w * upscale_factor)
                    new_h = int(h * upscale_factor)
                    
                    # Using NEAREST to preserve physical pixel edges
                    img_scaled = img.resize((new_w, new_h), Image.Resampling.NEAREST)
                    
                    # Save back to the SAME images folder with a suffix
                    output_name = img_path.stem + f"_{int(target_gsd)}gsd.png"
                    img_scaled.save(img_dir / output_name, "PNG")
                    
                # 2. Copy and rename corresponding label file
                src_lbl = lbl_dir / (img_path.stem + ".txt")
                dest_lbl = lbl_dir / (img_path.stem + f"_{int(target_gsd)}gsd.txt")
                
                if src_lbl.exists():
                    with open(src_lbl, 'r') as f_in, open(dest_lbl, 'w') as f_out:
                        f_out.write(f_in.read())
                else:
                    # Keep empty label file for background images to prevent val errors
                    open(dest_lbl, 'a').close()
                    
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                
    print("\n✔ All scaled tiers generated in-place successfully!")

if __name__ == "__main__":
    # Point this to your 'Angel' folder
    DATASET_ROOT = "data/Angel" 
    
    scale_up_in_place(DATASET_ROOT, target_gsds=[30.0, 15.0], current_gsd=50.0)