import cv2

font = cv2.FONT_HERSHEY_PLAIN
color = (255,0,0)

frame = cv2.imread("dog.jpg")
scale_fr = cv2.resize(frame, (640, 480))
cv2.rectangle(scale_fr, (182, 113), (518, 398), color, 1)
cv2.putText(scale_fr, f"Reza, 999%", (182, 113-5), font, 1, color, 1)


cv2.imshow("Gambar Anjing",scale_fr)


cv2.waitKey(0) 
cv2.destroyAllWindows()