# conversion/convert_to_qnn.py
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Convert YOLOv11 PyTorch model to Qualcomm QNN")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to your PyTorch weights")
    parser.add_argument("--htp_arch", type=str, default="75", help="Hexagon architecture version (e.g., 73, 75, 79)")
    args = parser.parse_args()

    print(f"Loading PyTorch model: {args.weights}")
    model = YOLO(args.weights)

    print(f"Exporting to QNN format targeting HTP architecture version {args.htp_arch}...")
    # This automatically handles the ONNX QDQ quantization pipeline optimized for Qualcomm NPUs
    exported_path = model.export(format="qnn", name=args.htp_arch)
    
    print(f"Export successfully completed! Output saved near: {exported_path}")

if __name__ == "__main__":
    main()