import os
import cv2

# Define paths matching your layout
RAW_LABELS_DIR = "data/vedai/vedai_color_obb/labels/"  # Change this to where your raw labels are currently stored
COLOR_IMGS_DIR = "data/vedai/vedai_color_obb/images/"
OUTPUT_ANNO_DIR = "data/vedai/annfiles/"

os.makedirs(OUTPUT_ANNO_DIR, exist_ok=True)

label_files = [f for f in os.listdir(RAW_LABELS_DIR) if f.endswith('.txt')]

print(f"[-] Processing {len(label_files)} VEDAI label files...")

for label_file in label_files:
    # Match the label text file to its image counterpart to get absolute dimensions
    img_base = os.path.splitext(label_file)[0]
    img_name = img_base + ".png" # Change extension if your VEDAI images are .jpg/.tiff
    img_path = os.path.join(COLOR_IMGS_DIR, img_name)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Warning: Corresponding image not found for {label_file} at {img_path}. Skipping.")
        continue
        
    h, w = img.shape[:2]
    converted_lines = []
    
    with open(os.path.join(RAW_LABELS_DIR, label_file), 'r') as infile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) < 9: continue
            
            # Extract normalized coordinates, ignoring the leading class index
            # YOLO format: class x1 y1 x2 y2 x3 y3 x4 y4
            norm_coords = [float(x) for x in parts[1:9]]
            
            # Un-normalize back to true absolute pixel values
            abs_coords = [
                norm_coords[0] * w, norm_coords[1] * h,  # x1, y1
                norm_coords[2] * w, norm_coords[3] * h,  # x2, y2
                norm_coords[4] * w, norm_coords[5] * h,  # x3, y3
                norm_coords[6] * w, norm_coords[7] * h   # x4, y4
            ]
            
            # Format as a space-separated string line
            str_line = " ".join([f"{coord:.2f}" for coord in abs_coords])
            converted_lines.append(str_line)
            
    # Write the un-normalized absolute points to the shared validation annotation folder
    with open(os.path.join(OUTPUT_ANNO_DIR, label_file), 'w') as outfile:
        outfile.write("\n".join(converted_lines))

print("[+] Dataset labels successfully converted and structured!")