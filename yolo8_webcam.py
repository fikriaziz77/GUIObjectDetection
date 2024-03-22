import cv2,time

cap = cv2.VideoCapture(0)
prev_time = 0
curr_time = 0
frame_count = 0
fps = 0

while True:
    ret, unscale_fr = cap.read()  
    
    if ret:

        unflip_frame = cv2.resize(unscale_fr, (640, 480))
        frame = cv2.flip(unflip_frame,1)

        # Calculate FPS
        curr_time = time.time()
        frame_count += 1
        if curr_time - prev_time >= 1:
            fps = frame_count / (curr_time - prev_time)
            frame_count = 0
            prev_time = curr_time

        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera FPS", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()