"""
================================================================================
RQ1 Test Data Explorer (Universal OBB/AABB Version)
================================================================================
Description:
    An interactive desktop GUI to browse the RQ1 test data (README Step 5.1) with
    a model's predictions drawn on it. Two dropdowns pick the dataset folder
    (COWC, COWC_point, VEDAI, Angel) and its GSD variant (15cm_base, 30cm_naive,
    ...) under <test-dir>/<Dataset>/<variant>/images/. Images are run at the same
    scale as yolo_eval.py (enlarged by train-imgsz / 512). Strips class text names
    entirely so detections stay perfectly clean.

    Dynamically scales bounding box line widths depending on the variant's GSD
    to prevent thick borders from drowning out tiny vehicles on degraded chips.

Prerequisites:
    yolo_env (Pillow, ultralytics) and a display (a local screen or ssh -X)

How to Run (from the repo root):
    $ python src/viewer.py \
        --model models/weights/test_weights/YOLO11n_15GSD_640.pt \
        --test-dir test_data --train-imgsz 640
================================================================================
"""

import os
import math
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

from eval.config import DATASET_VARIANTS
from eval.parsers import get_pixel_gsd

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
TRAIN_TILE_SIZE = 512  # cowc_512 training tiles, as in yolo_eval.py --train-tile-size

class DynamicInferenceViewer:
    def __init__(self, root, model_path, test_dir, train_imgsz, conf):
        self.root = root
        self.root.title(f"YOLO Resolution Explorer | Model: {os.path.basename(model_path)}")
        self.root.geometry("1000x950")

        self.test_dir = test_dir
        self.train_scale = train_imgsz / TRAIN_TILE_SIZE
        self.conf = conf
        self.datasets = sorted(d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))
        if not self.datasets:
            raise SystemExit(f"No dataset folders found in {test_dir}; see README Step 5.1")

        print(f"Loading YOLO model from: {model_path}...")
        self.model = YOLO(model_path)

        self.image_files = []
        self.current_idx = 0
        self.current_dataset = self.datasets[0]
        self.current_variant = None

        self.create_widgets()
        self.setup_key_bindings()
        self.on_dataset_change()

    def create_widgets(self):
        # Top Panel
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        
        self.btn_prev = ttk.Button(top_frame, text="◀ Previous", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=5)
        
        self.btn_next = ttk.Button(top_frame, text="Next ▶", command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=5)
        
        # Dataset and Variant Selection Dropdowns
        ttk.Label(top_frame, text="Dataset:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(20, 5))
        self.dataset_var = tk.StringVar()
        self.dataset_dropdown = ttk.Combobox(top_frame, textvariable=self.dataset_var, state="readonly", width=12)
        self.dataset_dropdown['values'] = self.datasets
        self.dataset_dropdown.current(0)
        self.dataset_dropdown.pack(side=tk.LEFT, padx=5)
        self.dataset_dropdown.bind("<<ComboboxSelected>>", self.on_dataset_change)

        ttk.Label(top_frame, text="Variant:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(10, 5))
        self.variant_var = tk.StringVar()
        self.variant_dropdown = ttk.Combobox(top_frame, textvariable=self.variant_var, state="readonly", width=16)
        self.variant_dropdown.pack(side=tk.LEFT, padx=5)
        self.variant_dropdown.bind("<<ComboboxSelected>>", self.on_variant_change)
        
        self.lbl_status = ttk.Label(top_frame, font=("Helvetica", 11, "bold"))
        self.lbl_status.pack(side=tk.RIGHT, padx=20)
        
        ttk.Separator(self.root, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Workspace Area
        self.display_frame = ttk.Frame(self.root, padding=10)
        self.display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.title_lbl = ttk.Label(self.display_frame, font=("Helvetica", 12, "bold"))
        self.title_lbl.pack(pady=5)
        
        self.img_lbl = ttk.Label(self.display_frame)
        self.img_lbl.pack(expand=True)

    def setup_key_bindings(self):
        # Bind arrow keys for quick desktop keyboard navigation
        self.root.bind("<Left>", lambda event: self.prev_image())
        self.root.bind("<Right>", lambda event: self.next_image())

    def on_dataset_change(self, event=None):
        self.current_dataset = self.dataset_var.get() or self.datasets[0]
        ds_dir = os.path.join(self.test_dir, self.current_dataset)
        present = [v for v in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, v))]
        # COWC_point holds the same variants as COWC; list them in DATASET_VARIANTS order, then any others
        known = DATASET_VARIANTS.get(self.current_dataset.split("_")[0], [])
        variants = [v for v in known if v in present] + sorted(v for v in present if v not in known)

        self.variant_dropdown['values'] = variants
        if variants:
            self.variant_dropdown.current(0)
        else:
            self.variant_var.set('')
        self.on_variant_change()

    def on_variant_change(self, event=None):
        self.current_variant = self.variant_var.get() or None
        self.current_idx = 0
        self.update_dataset_list()

    def update_dataset_list(self):
        self.image_files = []
        self.active_dir = None

        if self.current_variant:
            d = os.path.join(self.test_dir, self.current_dataset, self.current_variant, "images")
            if os.path.exists(d):
                self.image_files = sorted([f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTENSIONS)])
                self.active_dir = d
                print(f"Switched to {self.current_dataset}/{self.current_variant}: Found {len(self.image_files)} images.")

        self.load_and_predict()

    def load_and_predict(self):
        if not self.image_files:
            self.lbl_status.config(text=f"No images found in {self.current_dataset}/{self.current_variant}!")
            self.img_lbl.config(image='')
            self.title_lbl.config(text="Directory Empty")
            return
            
        filename = self.image_files[self.current_idx]
        self.lbl_status.config(text=f"Image [{self.current_idx + 1}/{len(self.image_files)}]")
        
        img_path = os.path.normpath(os.path.join(self.active_dir, filename))
        
        if os.path.exists(img_path):
            # Force conversion to 3-channel RGB to keep grayscale images from dropping bounding box overlays
            with Image.open(img_path) as PIL_raw:
                rgb_img = PIL_raw.convert("RGB")
                w, h = rgb_img.size
                
            # Same inference size as yolo_eval.py: the image enlarged by the training scale, rounded up to a multiple of 32
            imgsz = int(math.ceil(max(w, h) * self.train_scale / 32.0) * 32)
            results = self.model.predict(rgb_img, imgsz=imgsz, conf=self.conf, verbose=False)
            
            # --- DYNAMIC TASK DETECTION ---
            is_obb = hasattr(results[0], 'obb') and results[0].obb is not None and len(results[0].obb) > 0
            is_aabb = hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0

            # Dynamic Metadata Class Safeguard
            if is_obb:
                detected_classes = set(results[0].obb.cls.int().tolist())
            elif is_aabb:
                detected_classes = set(results[0].boxes.cls.int().tolist())
            else:
                detected_classes = set()

            for c in detected_classes:
                if c not in results[0].names:
                    results[0].names[c] = f"ID {c}"

            # --- DYNAMIC THICKNESS & MINIMALIST LABELS ---
            # Thinner strokes and smaller fonts prevent overlays from dominating low resolution matrices
            if get_pixel_gsd(self.current_variant) <= 15:
                thickness = 2      # Full-resolution chips
                font_sz = 10       # Minimalist font
            else:
                thickness = 1      # Kept at 1 (Ultralytics minimum) on coarser GSDs
                font_sz = 6        # Micro font
                
            # Render frame using universal plotting engine with customized line thickness and font scaling
            annotated_frame = results[0].plot(labels=False, conf=True, line_width=thickness, font_size=font_sz)
            
            # Convert BGR array (OpenCV standard) back to RGB (PIL display standard)
            annotated_frame_rgb = annotated_frame[..., ::-1] 
            pil_img = Image.fromarray(annotated_frame_rgb)
            
            # High-fidelity display scaling
            display_img = pil_img.resize((800, 800), Image.NEAREST)
            tk_img = ImageTk.PhotoImage(display_img)
            
            self.title_lbl.config(text=f"{filename} | {w}x{h} px, inference at {imgsz} px")
            self.img_lbl.config(image=tk_img)
            self.img_lbl.image = tk_img
        else:
            self.img_lbl.config(image='')
            self.title_lbl.config(text="File Error")

    def next_image(self):
        if self.image_files and self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.load_and_predict()

    def prev_image(self):
        if self.image_files and self.current_idx > 0:
            self.current_idx -= 1
            self.load_and_predict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Resolution Dynamic YOLO Explorer")
    parser.add_argument('--model', type=str, required=True, help="YOLO .pt weights")
    parser.add_argument('--test-dir', type=str, required=True,
                        help="Test data folder from README Step 5.1, laid out as <Dataset>/<variant>/images/ (e.g. test_data)")
    parser.add_argument('--train-imgsz', type=int, required=True,
                        help="--img_size the model was trained at (train.py), so images are scaled as in yolo_eval.py")
    parser.add_argument('--conf', type=float, default=0.10,
                        help="Only draw boxes with at least this confidence (low, to catch faint vehicles at coarse GSDs)")

    args = parser.parse_args()
    root = tk.Tk()
    app = DynamicInferenceViewer(root, args.model, args.test_dir, args.train_imgsz, args.conf)
    root.mainloop()