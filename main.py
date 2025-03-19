from ultralytics import YOLO

# Load the YOLO8 model
model = YOLO("yolo8n.pt")

# Load the exported NCNN model (./best_ncnn_model)
ncnn_model = YOLO("your_model_path")

# Run inference
results = ncnn_model("https://ultralytics.com/images/bus.jpg")
