from ultralytics import YOLO
import json

# Load a model
model = YOLO("/home/lab333/marcelinho/yolo/abdomen1k/train_fake3/weights/best.pt")

# Evaluate model performance on the validation set
metrics = model.val(split='test')
