import os
from pathlib import Path
from PIL import Image

def generate_cowc_centerpoint_study(src_img_dir, src_obb_lbl_dir, output_dir, native_gsd=15.0):
    """
    Processes the raw 15 GSD COW-C test images into 15, 30, and 60 GSD tiers,
    while converting 8-coordinate OBB labels into normalized center points (class_id x_c y_c)
    directly inside a single clean directory structure.
    """
    img_dir = Path(src_img_dir)
    lbl_dir = Path(src_obb_lbl_dir)
    out_base = Path(output_dir)
    
    out_img_dir = out_base / "images"
    out_lbl_dir = out_base / "labels"
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in valid_extensions]
    
    print(f"Loaded {len(images)} raw COW-C images from {img_dir}")
    print("Generating GSD tiers and converting labels to center points...")

    for img_path in images:
        try:
            stem = img_path.stem
            
            # --- 1. Process & Convert Labels First ---
            # Extract center points from 8-coordinate OBB labels
            src_lbl_path = lbl_dir / f"{stem}.txt"
            center_lines = []
            
            if src_lbl_path.exists():
                with open(src_lbl_path, 'r') as f_in:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) < 9: # Skip invalid/corrupt lines
                            continue
                        
                        class_id = parts[0]
                        coords = [float(val) for val in parts[1:]]
                        
                        # Take every second value to separate X and Y coordinates
                        xs = coords[0::2]
                        ys = coords[1::2]
                        
                        # Calculate center mass
                        x_center = sum(xs) / len(xs)
                        y_center = sum(ys) / len(ys)
                        
                        # Clamp safely to image boundaries
                        x_center = min(max(x_center, 0.0), 1.0)
                        y_center = min(max(y_center, 0.0), 1.0)
                        
                        center_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f}\n")
            
            # --- Helper function to write labels cleanly ---
            def write_label(dest_stem):
                dest_lbl_path = out_lbl_dir / f"{dest_stem}.txt"
                with open(dest_lbl_path, 'w') as f_out:
                    f_out.writelines(center_lines)

            # --- 2. Process and Save Images across Tiers ---
            with Image.open(img_path) as img:
                orig_w, orig_h = img.size
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # --- Tier A: 15 GSD Baseline (Pristine) ---
                img.save(out_img_dir / f"{stem}_15gsd.png", "PNG")
                write_label(f"{stem}_15gsd")
                
                # --- Tier B: 30 GSD Tiers ---
                scale_30 = native_gsd / 30.0
                w_30, h_30 = int(orig_w * scale_30), int(orig_h * scale_30)
                
                # Naive 30 GSD (Physically downsampled to 256x256 px)
                img_30_naive = img.resize((w_30, h_30), Image.Resampling.BILINEAR)
                img_30_naive.save(out_img_dir / f"{stem}_30gsd_naive.png", "PNG")
                write_label(f"{stem}_30gsd_naive")
                
                # Normalized 30 GSD (Upscaled back to original 512x512 px)
                img_30_norm = img_30_naive.resize((orig_w, orig_h), Image.Resampling.NEAREST)
                img_30_norm.save(out_img_dir / f"{stem}_30gsd_normalized.png", "PNG")
                write_label(f"{stem}_30gsd_normalized")
                
                # --- Tier C: 60 GSD Tiers ---
                scale_60 = native_gsd / 60.0
                w_60, h_60 = int(orig_w * scale_60), int(orig_h * scale_60)
                
                # Naive 60 GSD (Physically downsampled to 128x128 px)
                img_60_naive = img.resize((w_60, h_60), Image.Resampling.BILINEAR)
                img_60_naive.save(out_img_dir / f"{stem}_60gsd_naive.png", "PNG")
                write_label(f"{stem}_60gsd_naive")
                
                # Normalized 60 GSD (Upscaled back to original 512x512 px)
                img_60_norm = img_60_naive.resize((orig_w, orig_h), Image.Resampling.NEAREST)
                img_60_norm.save(out_img_dir / f"{stem}_60gsd_normalized.png", "PNG")
                write_label(f"{stem}_60gsd_normalized")

        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")
            continue

    print(f"\n✔ Center-Point GSD Study Dataset successfully saved to: {out_base.resolve()}")

if __name__ == "__main__":
    # Point these to your raw test source paths
    RAW_IMAGES = "data/cowc_test/images"
    RAW_OBB_LABELS = "data/cowc_test/labels"
    
    # Destination folder where your clean experiment directory will be created
    EXPERIMENT_OUTPUT = "data/cowc_test_study"
    
    generate_cowc_centerpoint_study(RAW_IMAGES, RAW_OBB_LABELS, EXPERIMENT_OUTPUT)