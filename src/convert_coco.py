import json
import os
from collections import defaultdict

# 1. Configuration - Change these to match your setup
json_path = "data/Angel/labels_my-project-name_2026-07-15-11-50-54.json"  # Path to your exported COCO JSON file
output_dir = "data/Angel/labels"       # Where you want the .txt files saved

os.makedirs(output_dir, exist_ok=True)

# 2. Load the COCO data
with open(json_path, 'r') as f:
    coco_data = json.load(f)

# Create lookup dictionaries for speed
images = {img['id']: img for img in coco_data['images']}
categories = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}

# 3. Process annotations (collected per file, then written once so reruns don't duplicate labels)
label_lines = defaultdict(list)
skipped_non_quad = 0

for anno in coco_data['annotations']:
    image_id = anno['image_id']
    category_id = anno['category_id']

    # Get image metadata to normalize coordinates
    img_meta = images.get(image_id)
    if not img_meta:
        continue

    img_w = img_meta['width']
    img_h = img_meta['height']
    img_name = img_meta['file_name']

    # Get the base filename without extension (e.g., "image_59a66e")
    base_name = os.path.splitext(img_name)[0]

    # Map COCO category ID to a zero-indexed YOLO class ID
    yolo_class = categories.get(category_id, 0)

    # Extract and normalize polygon segmentation points
    # COCO stores segments as a flat list: [x1, y1, x2, y2, ...]
    if 'segmentation' in anno and anno['segmentation']:
        # Taking the first polygon segment path
        segment = anno['segmentation'][0]

        # Downstream parsers expect exactly 4 corners (8 values); anything else would be silently truncated
        if len(segment) != 8:
            skipped_non_quad += 1
            print(f"[WARNING] Skipping annotation {anno.get('id')} in {img_name}: "
                  f"{len(segment) // 2} points (expected a 4-corner box)")
            continue

        normalized_coords = []
        for i in range(0, len(segment), 2):
            x = segment[i] / img_w
            y = segment[i+1] / img_h
            normalized_coords.append(f"{x:.6f} {y:.6f}")

        # Format line: <class> <x1> <y1> <x2> <y2> ...
        label_lines[base_name].append(f"{yolo_class} " + " ".join(normalized_coords) + "\n")

# 4. Write each label file fresh (overwrites any previous run's output)
for base_name, lines in label_lines.items():
    with open(os.path.join(output_dir, f"{base_name}.txt"), 'w') as txt_file:
        txt_file.writelines(lines)

if skipped_non_quad:
    print(f"[WARNING] {skipped_non_quad} non-4-corner polygons were skipped; re-annotate them as rotated boxes.")
print(f"Done! YOLO polygon labels saved to the '{output_dir}' directory.")
