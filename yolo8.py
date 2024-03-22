from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("yolov8n.pt")

in_img = cv2.imread("dog.jpg")
scale_fr = cv2.resize(in_img, (640, 480))

results = model.predict(scale_fr)

result = results[0]

print(result)
'''
len(result.boxes)

box = result.boxes[0]

for box in result.boxes:
  class_id = result.names[box.cls[0].item()]
  cords = box.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  conf = round(box.conf[0].item(), 2)
 
  print("Object type:", class_id)
  print("Coordinates:", cords)
  print("Probability:", conf)
  print("Coor:", f"({cords})")
  print("---")


img = Image.fromarray(result.plot()[:,:,::-1])
img.show()
'''