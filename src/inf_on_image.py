import os
from PIL import Image
from ultralytics import YOLO

def generate_prediction_image(model_path, image_path, output_name="annotated_output.jpg"):
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
    results = model.predict(source=img, conf=0.4)
    
    # 4. Save the annotated image
    # results[0] grabs the data for our single image. 
    # labels=False removes the class name.
    # conf=False removes the confidence score.
    print("Drawing boxes and saving image...")
    results[0].save(filename=output_name, labels=False, conf=False)
    
    print(f"Success! Annotated image saved directly to: {output_name}")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # Update these variables with your actual file names/paths
    # ---------------------------------------------------------
    
    # The path to your PyTorch model
    MY_MODEL = "models/weights/test_weights/YOLO11n_15GSD.pt"  
    
    # The path to the image you want to test
    MY_IMAGE = "test_data/COWC/60cm_scalenorm/images/Potsdam_ISPRS_top_potsdam_2_13_RGB_patch_512_1280.png"
    
    # What you want the final saved image to be called
    MY_OUTPUT_IMAGE = "predictions_drawn.jpg" 
    
    generate_prediction_image(MY_MODEL, MY_IMAGE, MY_OUTPUT_IMAGE)