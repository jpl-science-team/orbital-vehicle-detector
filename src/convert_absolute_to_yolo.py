import os
from pathlib import Path

# --- CONFIGURATION ---
INPUT_DIR = "data/cowc_test/refinedboxes"      # Folder containing your 8-coordinate label files
OUTPUT_DIR = "data/cowc_test/labels"  # Where the clean center-point labels will be saved
# ---------------------

def convert_obb_to_centers():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    label_files = list(input_path.glob("*.txt"))
    print(f"Found {len(label_files)} OBB label files to convert.")
    
    for file_path in label_files:
        center_lines = []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Expecting 1 class ID + 8 coordinates (4 points) = 9 parts
                if len(parts) < 9:
                    continue
                
                class_id = parts[0]
                coords = [float(val) for val in parts[1:]]
                
                # Extract the four x and y coordinates
                xs = coords[0::2]  # Takes index 0, 2, 4, 6
                ys = coords[1::2]  # Takes index 1, 3, 5, 7
                
                # Calculate the exact center mass of the box
                x_center = sum(xs) / len(xs)
                y_center = sum(ys) / len(ys)
                
                # Keep coordinates clamped safely between 0.0 and 1.0
                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                
                # Output format: class_id x_center y_center
                center_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f}\n")
        
        # Save the new file with the same name
        out_file_path = output_path / file_path.name
        with open(out_file_path, 'w') as f_out:
            f_out.writelines(center_lines)

    print(f"✔ Conversion complete! Clean center points saved to: {output_path.resolve()}")

if __name__ == "__main__":
    convert_obb_to_centers()