from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")

# Train the model
train_results = model.train(
    data="abdomen1k.yaml",  # path to dataset YAML
    epochs=500,  # number of training epochs
    imgsz=256,  # training image size
    batch=64,
    optimizer="SGD",
    momentum=0.9,
    project="abdomen1k",
    name="train_fake"
)

# Evaluate model performance on the validation set
metrics = model.val(split="test")
