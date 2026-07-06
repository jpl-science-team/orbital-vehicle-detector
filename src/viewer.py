"""
================================================================================
COWC Dynamic Resolution Dataset Explorer (Universal OBB/AABB Version)
================================================================================
Description:
    An interactive desktop GUI to inspect partitioned test datasets. The dropdown
    menu seamlessly switches your active directory between the three resolution
    splits (512, 256, 128) and loads the unique images for that specific folder.
    Strips class text names entirely so detections stay perfectly clean.
    
    Dynamically scales bounding box line widths depending on the split resolution 
    to prevent thick borders from drowning out tiny vehicles on degraded chips.

Prerequisites:
    $ pip install Pillow ultralytics

How to Run:
    $ python src/viewer.py \
        --model /Users/graddy/work/orbital-vehicle-detector/models/weights/sharioz_baseline_best.pt \
        --splits_dir /Users/graddy/work/orbital-vehicle-detector/data/cowc_split_test
================================================================================
"""

import os
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

DEFAULT_MODEL = "models/weights/20260618_256_best.pt"
DEFAULT_SPLITS_DIR = "data/cowc_split_test"

class DynamicInferenceViewer:
    def __init__(self, root, model_path, splits_dir):
        self.root = root
        self.root.title(f"YOLO Resolution Explorer | Model: {os.path.basename(model_path)}")
        self.root.geometry("1000x950")
        
        self.splits_dir = splits_dir
        
        print(f"Loading YOLO model from: {model_path}...")
        self.model = YOLO(model_path)
        
        self.image_files = []
        self.current_idx = 0
        self.current_split = 'test_512' # Default startup folder
        
        self.create_widgets()
        self.setup_key_bindings()
        self.update_dataset_list()

    def create_widgets(self):
        # Top Panel
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        
        self.btn_prev = ttk.Button(top_frame, text="◀ Previous", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=5)
        
        self.btn_next = ttk.Button(top_frame, text="Next ▶", command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=5)
        
        # Dataset Selection Dropdown
        ttk.Label(top_frame, text="Active Dataset:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(20, 5))
        self.dataset_var = tk.StringVar()
        self.dataset_dropdown = ttk.Combobox(top_frame, textvariable=self.dataset_var, state="readonly", width=18)
        self.dataset_dropdown['values'] = ('test_512 (Original)', 'test_256 (Degraded)', 'test_128 (Degraded)')
        self.dataset_dropdown.current(0)
        self.dataset_dropdown.pack(side=tk.LEFT, padx=5)
        
        self.dataset_dropdown.bind("<<ComboboxSelected>>", self.on_dropdown_change)
        
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

    def on_dropdown_change(self, event=None):
        selection = self.dataset_var.get()
        if '512' in selection:
            self.current_split = 'test_512'
        elif '256' in selection:
            self.current_split = 'test_256'
        else:
            self.current_split = 'test_128'
            
        self.current_idx = 0
        self.update_dataset_list()

    def update_dataset_list(self):
        search_dirs = [
            os.path.join(self.splits_dir, self.current_split, "images"),
            os.path.join(self.splits_dir, self.current_split, "images", "test"),
            os.path.join(self.splits_dir, self.current_split)
        ]
        
        self.image_files = []
        self.active_dir = None
        
        for d in search_dirs:
            if os.path.exists(d):
                files = sorted([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                if files:
                    self.image_files = files
                    self.active_dir = d
                    print(f"Switched to {self.current_split}: Found {len(self.image_files)} images.")
                    break
                    
        self.load_and_predict()

    def load_and_predict(self):
        if not self.image_files:
            self.lbl_status.config(text=f"No images found in {self.current_split}!")
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
                
            # Maintain a lower conf floor (0.10) to catch low-contrast vehicles on degraded resolutions
            results = self.model.predict(rgb_img, imgsz=512, conf=0.10, verbose=False)
            
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
            if '512' in self.current_split:
                thickness = 2      # Decreased from 3
                font_sz = 10       # Minimalist font
            elif '256' in self.current_split:
                thickness = 1      # Decreased from 2
                font_sz = 8        # Extra small font
            else:
                thickness = 1      # Kept at 1 (Ultralytics minimum)
                font_sz = 6        # Micro font for 128x128
                
            # Render frame using universal plotting engine with customized line thickness and font scaling
            annotated_frame = results[0].plot(labels=False, conf=True, line_width=thickness, font_size=font_sz)
            
            # Convert BGR array (OpenCV standard) back to RGB (PIL display standard)
            annotated_frame_rgb = annotated_frame[..., ::-1] 
            pil_img = Image.fromarray(annotated_frame_rgb)
            
            # High-fidelity display scaling
            display_img = pil_img.resize((800, 800), Image.NEAREST)
            tk_img = ImageTk.PhotoImage(display_img)
            
            self.title_lbl.config(text=f"{filename} | Native Resolution: {w}x{h} px")
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
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--splits_dir', type=str, default=DEFAULT_SPLITS_DIR)
    
    args = parser.parse_args()
    root = tk.Tk()
    app = DynamicInferenceViewer(root, args.model, args.splits_dir)
    root.mainloop()