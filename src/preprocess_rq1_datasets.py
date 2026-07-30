import os
import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------
RAW_DATA_DIR = Path("data")       # STRICTLY READ-ONLY
SANDBOX_DIR = Path("test_data")   # ALL EXPERIMENTAL OUTPUTS SAVED HERE

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def get_image_files(directory_path):
    """Recursively scans for valid images, ignoring OS hidden files."""
    if not directory_path.exists():
        return []
    
    image_paths = []
    for p in directory_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS and not p.name.startswith("."):
            image_paths.append(str(p))
            
    return sorted(image_paths)


def find_label_file(dataset_dir, stem, anno_stem=None):
    """Locates label text files across common directory naming conventions."""
    stems_to_check = [stem]
    if anno_stem and anno_stem != stem:
        stems_to_check.append(anno_stem)

    possible_dirs = ["labels", "annfiles", "Annotations512", "Annotations1024", "annotations", ""]

    for d in possible_dirs:
        for s in stems_to_check:
            lbl_p = dataset_dir / d / f"{s}.txt"
            if lbl_p.exists():
                return lbl_p

    for s in stems_to_check:
        for lbl_p in dataset_dir.rglob(f"{s}.txt"):
            return lbl_p

    return None


# ---------------------------------------------------------
# LABEL PARSERS (PRESERVES CLASSES & METADATA)
# ---------------------------------------------------------
def convert_angel_label(line, img_w, img_h):
    """Converts Angel 8-point polygon while preserving class ID if present."""
    parts = line.strip().split()
    if len(parts) < 8:
        return None
    
    if len(parts) >= 9:
        class_id = int(parts[0]) if parts[0].isdigit() else 0
        raw_coords = [float(x) for x in parts[1:9]]
    else:
        class_id = 0
        raw_coords = [float(x) for x in parts[:8]]
    
    is_normalized = max(raw_coords) <= 1.0
    
    pixel_coords = []
    for i in range(0, 8, 2):
        x = raw_coords[i] * img_w if is_normalized else raw_coords[i]
        y = raw_coords[i+1] * img_h if is_normalized else raw_coords[i+1]
        pixel_coords.extend([x, y])
        
    return {'type': 'angel', 'coords': pixel_coords, 'class_id': class_id}


def convert_cowc_label(line):
    """Parses raw COWC centerpoint coordinates."""
    parts = line.strip().split()
    if len(parts) == 2:
        return {'type': 'cowc', 'x': float(parts[0]), 'y': float(parts[1])}
    elif len(parts) >= 3:
        return {'type': 'cowc', 'x': float(parts[1]), 'y': float(parts[2])}
    return None


def convert_vedai_label(line):
    """Parses VEDAI format perfectly to preserve class IDs and orientation."""
    parts = line.strip().split()
    if not parts:
        return None
    
    # Standard VEDAI raw format: 14/15 fields [cx, cy, angle, class, cont, occl, x1..x4, y1..y4]
    if len(parts) in (14, 15):
        offset = 1 if len(parts) == 15 else 0
        try:
            return {
                'type': 'vedai',
                'cx': float(parts[0 + offset]),
                'cy': float(parts[1 + offset]),
                'angle': float(parts[2 + offset]),
                'class_id': int(parts[3 + offset]),
                'cont': int(parts[4 + offset]),
                'occl': int(parts[5 + offset]),
                'x_coords': [float(x) for x in parts[6 + offset : 10 + offset]],
                'y_coords': [float(y) for y in parts[10 + offset : 14 + offset]]
            }
        except ValueError:
            pass

    # Fallbacks for manually annotated files
    if len(parts) == 8:
        return {'type': 'angel', 'coords': [float(x) for x in parts[:8]], 'class_id': 0}
    if len(parts) == 9:
        class_id = int(parts[0]) if parts[0].isdigit() else 0
        return {'type': 'angel', 'coords': [float(x) for x in parts[-8:]], 'class_id': class_id}

    return None


# ---------------------------------------------------------
# VARIANT BUILDERS & SAVER
# ---------------------------------------------------------
def scale_target(t, scale):
    """Mathematically scales coordinates based on label type."""
    if t['type'] == 'angel':
        return {**t, 'coords': [c * scale for c in t['coords']]}
    elif t['type'] == 'cowc':
        return {**t, 'x': t['x'] * scale, 'y': t['y'] * scale}
    elif t['type'] == 'vedai':
        return {
            **t,
            'cx': t['cx'] * scale,
            'cy': t['cy'] * scale,
            'x_coords': [x * scale for x in t['x_coords']],
            'y_coords': [y * scale for y in t['y_coords']]
        }
    return t


def save_variants(output_ds_dir, file_stem, variants):
    """Saves images and properly formatted metadata back to text files."""
    for var_name, (v_img, v_targets) in variants.items():
        img_out = output_ds_dir / var_name / "images"
        lbl_out = output_ds_dir / var_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(img_out / f"{file_stem}.png"), v_img)

        with open(lbl_out / f"{file_stem}.txt", "w") as f:
            for t in v_targets:
                if t['type'] == 'cowc':
                    f.write(f"0 {t['x']:.2f} {t['y']:.2f}\n")
                elif t['type'] == 'angel':
                    str_coords = " ".join([f"{c:.2f}" for c in t['coords']])
                    f.write(f"{str_coords} vehicle {t['class_id']}\n")
                elif t['type'] == 'vedai':
                    # Reconstruct exact VEDAI format: cx cy angle class cont occl x1..x4 y1..y4
                    xs = " ".join([f"{x:.2f}" for x in t['x_coords']])
                    ys = " ".join([f"{y:.2f}" for y in t['y_coords']])
                    f.write(f"{t['cx']:.2f} {t['cy']:.2f} {t['angle']:.6f} {t['class_id']} {t['cont']} {t['occl']} {xs} {ys}\n")


def build_angel_variants(img, targets):
    """Angel (50cm native) -> Upscaled to 30cm (1.667x) and 15cm (3.333x)."""
    h_orig, w_orig = img.shape[:2]
    s_30 = 50.0 / 30.0
    s_15 = 50.0 / 15.0

    # 50cm Base
    img_50 = img
    targets_50 = targets

    # 30cm Upscale
    w_30, h_30 = max(1, int(round(w_orig * s_30))), max(1, int(round(h_orig * s_30)))
    img_30 = cv2.resize(img, (w_30, h_30), interpolation=cv2.INTER_CUBIC)
    targets_30 = [scale_target(t, s_30) for t in targets]

    # 15cm Upscale
    w_15, h_15 = max(1, int(round(w_orig * s_15))), max(1, int(round(h_orig * s_15)))
    img_15 = cv2.resize(img, (w_15, h_15), interpolation=cv2.INTER_CUBIC)
    targets_15 = [scale_target(t, s_15) for t in targets]

    return {
        '50cm_base': (img_50, targets_50),
        '30cm_upscale': (img_30, targets_30),
        '15cm_upscale': (img_15, targets_15)
    }


def build_degradation_variants(img, targets, native_gsd):
    """
    COWC (15cm) & VEDAI (12.5cm) -> Downscaled to 15cm base, 30cm, and 60cm variants.
    """
    h_orig, w_orig = img.shape[:2]

    s_15 = native_gsd / 15.0
    s_30 = native_gsd / 30.0
    s_60 = native_gsd / 60.0

    # 15cm Base Space
    w_15, h_15 = max(1, int(round(w_orig * s_15))), max(1, int(round(h_orig * s_15)))
    img_15 = cv2.resize(img, (w_15, h_15), interpolation=cv2.INTER_AREA) if s_15 != 1.0 else img
    targets_15 = [scale_target(t, s_15) for t in targets]

    # 30cm Naive & Scalenorm
    w_30, h_30 = max(1, int(round(w_orig * s_30))), max(1, int(round(h_orig * s_30)))
    img_30_naive = cv2.resize(img, (w_30, h_30), interpolation=cv2.INTER_AREA)
    targets_30 = [scale_target(t, s_30) for t in targets]
    img_30_scalenorm = cv2.resize(img_30_naive, (w_15, h_15), interpolation=cv2.INTER_CUBIC)

    # 60cm Naive & Scalenorm
    w_60, h_60 = max(1, int(round(w_orig * s_60))), max(1, int(round(h_orig * s_60)))
    img_60_naive = cv2.resize(img, (w_60, h_60), interpolation=cv2.INTER_AREA)
    targets_60 = [scale_target(t, s_60) for t in targets]
    img_60_scalenorm = cv2.resize(img_60_naive, (w_15, h_15), interpolation=cv2.INTER_CUBIC)

    return {
        '15cm_base': (img_15, targets_15),
        '30cm_naive': (img_30_naive, targets_30),
        '30cm_scalenorm': (img_30_scalenorm, targets_15),
        '60cm_naive': (img_60_naive, targets_60),
        '60cm_scalenorm': (img_60_scalenorm, targets_15)
    }


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def run_preprocessing():
    print("==================================================")
    print(" UNIFIED TEST DATA PREPROCESSING PIPELINE")
    print("==================================================")
    print(f"Reading raw data from: {RAW_DATA_DIR.resolve()} (READ-ONLY)")
    print(f"Writing sandbox to:   {SANDBOX_DIR.resolve()}\n")

    # 1. ANGEL (Native 50cm GSD -> Upscaling to 30cm & 15cm)
    angel_base = RAW_DATA_DIR / "Angel"
    angel_imgs = get_image_files(angel_base / "images") if (angel_base / "images").exists() else get_image_files(angel_base)
    print(f"Processing Angel Dataset ({len(angel_imgs)} images - Native 50cm GSD -> Upscaling to 30cm & 15cm)...")
    
    for img_p in angel_imgs:
        stem = Path(img_p).stem
        lbl_p = find_label_file(angel_base, stem)
        img = cv2.imread(img_p)
        if img is None: 
            continue
        h, w = img.shape[:2]

        targets = []
        if lbl_p and lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    box = convert_angel_label(line, w, h)
                    if box: 
                        targets.append(box)

        variants = build_angel_variants(img, targets)
        save_variants(SANDBOX_DIR / "Angel", stem, variants)

    # 2. COWC (Native 15cm GSD)
    cowc_base = RAW_DATA_DIR / "COWC"
    cowc_imgs = get_image_files(cowc_base / "images") if (cowc_base / "images").exists() else get_image_files(cowc_base)
    print(f"Processing COWC Dataset ({len(cowc_imgs)} images - Native 15cm Centerpoints)...")
    
    for img_p in cowc_imgs:
        stem = Path(img_p).stem
        lbl_p = find_label_file(cowc_base, stem)
        img = cv2.imread(img_p)
        if img is None: 
            continue

        targets = []
        if lbl_p and lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    pt = convert_cowc_label(line)
                    if pt: 
                        targets.append(pt)

        variants = build_degradation_variants(img, targets, native_gsd=15.0)
        save_variants(SANDBOX_DIR / "COWC", stem, variants)

    # 3. VEDAI (Native 12.5cm GSD -> Color & IR Downscaling)
    vedai_base = RAW_DATA_DIR / "VEDAI"
    color_dir = vedai_base / "vedai_color_obb"
    ir_dir = vedai_base / "vedai_ir_obb"
    
    vedai_imgs = get_image_files(color_dir) + get_image_files(ir_dir)
    if len(vedai_imgs) == 0:
        vedai_imgs = [
            p for p in get_image_files(vedai_base) 
            if not any(x in p.lower() for x in ["annfiles", "annotations", "labels"])
        ]

    print(f"Processing VEDAI Dataset ({len(vedai_imgs)} images across Color & IR modalities)...")
    
    for img_p in vedai_imgs:
        raw_stem = Path(img_p).stem
        anno_stem = raw_stem.replace("_co", "").replace("_ir", "")
        
        lbl_p = find_label_file(vedai_base, raw_stem, anno_stem)
        img = cv2.imread(img_p)
        if img is None: 
            continue

        targets = []
        if lbl_p and lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    box = convert_vedai_label(line)
                    if box: 
                        targets.append(box)

        variants = build_degradation_variants(img, targets, native_gsd=12.5)
        save_variants(SANDBOX_DIR / "VEDAI", raw_stem, variants)

    print("\n==================================================")
    print(f" SUCCESS! Test data generated in: '{SANDBOX_DIR.resolve()}'")
    print("==================================================")

if __name__ == "__main__":
    run_preprocessing()