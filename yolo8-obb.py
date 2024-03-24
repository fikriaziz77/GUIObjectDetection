<<<<<<< HEAD
from ultralytics import YOLO
from PIL import Image
import cv2, math, time
import numpy as np
import torch

#torch.cuda.set_device(0)
model = YOLO("PCB-Relay(3000).pt")

cap = cv2.VideoCapture(0)
prev_time = 0
curr_time = 0
frame_count = 0
fps = 0
font = cv2.FONT_HERSHEY_PLAIN
color = (255,0,0)

while True:
    ret, unscale_fr = cap.read()  
    
  
    if ret:

        unflip_frame = cv2.resize(unscale_fr, (640, 480))
        frame = cv2.flip(unflip_frame,1)

        results = model.predict(frame, imgsz=320, conf=0.5)
        result = results[0]
        box = result.obb

        for box in result.obb:
            class_id = result.names[box.cls[0].item()]
            cords = box.xywhr[0].tolist()
            cords_corner = box.xyxyxyxy[0].tolist()
            conf = round(box.conf[0].item(), 2)
            
            px1 = int(cords_corner[0][0])
            py1 = int(cords_corner[0][1])
            cv2.circle(frame, (px1,py1), 2, (0,0,255), 2)

            px2 = int(cords_corner[1][0])
            py2 = int(cords_corner[1][1])
            cv2.circle(frame, (px2,py2), 2, (0,255,0), 2)

            x = round(cords[0])
            y = round(cords[1])
            w = round(cords[2])
            h = round(cords[3])
            val = cords[4]

            r = round(math.degrees(val),2)
            rect = ((x, y), (w, h), r)
            box = cv2.boxPoints(rect) 
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,color,1)
            cv2.putText(frame, f"{class_id} :  {round(val,2)}rad-{r}deg",(x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

        # Calculate FPS
        curr_time = time.time()
        frame_count += 1
        if curr_time - prev_time >= 1:
            fps = frame_count / (curr_time - prev_time)
            frame_count = 0
            prev_time = curr_time

        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Camera FPS", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
=======
from ultralytics import YOLO
from PIL import Image
import cv2, math, time
import numpy as np

model = YOLO("yolov8n-obb.pt")

cap = cv2.VideoCapture(0)

prev_time = 0
curr_time = 0
frame_count = 0
fps = 0
font = cv2.FONT_HERSHEY_PLAIN
f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()  
    
  
    if ret:

        #unflip_frame = cv2.resize(unscale_fr, (640, 380))
        #frame = cv2.flip(unflip_frame,1)

        results = model.predict(frame, imgsz=320, conf=0.6)
        result = results[0]
        box = result.obb
        print(box)

        for box in result.obb:
            class_id = result.names[box.cls[0].item()]
            cords = box.xywhr[0].tolist()
            cords_corner = box.xyxyxyxy[0].tolist()
            conf = round(box.conf[0].item(), 2)
            

            x = round(cords[0])
            y = round(cords[1])
            w = round(cords[2])
            h = round(cords[3])
            val = cords[4]

            r = round(math.degrees(val),2)
            rect = ((x, y), (w, h), r)
            box = cv2.boxPoints(rect) 
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(255,0,0),1)

            cx = int(f_width/2)
            cy = int(f_height/2)
            cv2.circle(frame, (cx,cy), 2, (0,255,0), 2) #frame center
            cv2.circle(frame, (x,y), 2, (0,255,0), 2) #object center
            cv2.line(frame, (x,y), (cx,cy), (0, 0, 255), 2)
            cv2.line(frame, (x,y), (cx,y), (255,0,0),1) #dx
            cv2.line(frame, (x,y), (x,cy), (255,0,0),1) #dy



            cv2.putText(frame, f"{conf*100}% :  {r}deg",(x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)

        # Calculate FPS
        curr_time = time.time()
        frame_count += 1
        if curr_time - prev_time >= 1:
            fps = frame_count / (curr_time - prev_time)
            frame_count = 0
            prev_time = curr_time

        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow("Camera FPS", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
>>>>>>> 9b903e1667882f1a73e022a9a93d1dccb84092a0
