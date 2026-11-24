import os
import argparse
from PIL import Image
from ultralytics import YOLO

def generate_prediction_image(model_path, image_path, output_name="annotated_output.jpg", conf=0.4):
    """
    Loads a YOLO model, converts a grayscale image to RGB, 
    runs inference, and saves the image with only bounding boxes drawn 
    (no text, no conf scores).
    """
    # Quick check to ensure your files exist
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return
    
    # 1. Load the YOLO model
    print(f"Loading model '{model_path}'...")
    model = YOLO(model_path)
    
    # 2. Load and convert the image to 3 channels (RGB)
    print(f"Loading and converting '{image_path}' to 3-channel RGB...")
    img = Image.open(image_path).convert("RGB")
    
    # 3. Run inference on the converted image object
    print("Running inference...")
    results = model.predict(source=img, conf=conf)
    
    # 4. Save the annotated image
    # results[0] grabs the data for our single image. 
    # labels=False removes the class name.
    # conf=False removes the confidence score.
    print("Drawing boxes and saving image...")
    results[0].save(filename=output_name, labels=False, conf=False)
    
    print(f"Success! Annotated image saved directly to: {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a YOLO model on one image and save it with the predicted boxes drawn")
    parser.add_argument("--model", type=str, required=True, help="YOLO .pt weights")
    parser.add_argument("--image", type=str, required=True, help="Image to run on")
    parser.add_argument("--out", type=str, required=True, help="Output image path (e.g. predictions_drawn.jpg)")
    parser.add_argument("--conf", type=float, default=0.4, help="Only draw boxes with at least this confidence")
    args = parser.parse_args()

    generate_prediction_image(args.model, args.image, args.out, args.conf)