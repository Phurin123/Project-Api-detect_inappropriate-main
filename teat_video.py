import cv2
from ultralytics import YOLO

# โหลดโมเดลที่คุณ train ไว้
model = YOLO(r"C:\Users\khongkaphan\Music\โปเจ็ควิดีโอ\yolov7-object-blurring\best-weapon.pt")

# เปิดวิดีโอ
cap = cv2.VideoCapture(r"C:\Users\khongkaphan\Downloads\myvideo.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_blurred.mp4", fourcc, cap.get(5),
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detect
    results = model(frame, conf=0.25)

    # วาด blur ที่ bounding box
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                roi = cv2.GaussianBlur(roi, (51, 51), 30)
                frame[y1:y2, x1:x2] = roi

    out.write(frame)

cap.release()
out.release()
print("✅ เสร็จแล้ว! วิดีโออยู่ที่ output_blurred.mp4")
