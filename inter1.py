from ultralytics import YOLO
import torch

model = YOLO('yolov8n-obb.pt')

model.export(format='engine')