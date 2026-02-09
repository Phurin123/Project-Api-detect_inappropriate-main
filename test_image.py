import os
import cv2
import numpy as np
import math
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. ตั้งค่าพื้นฐาน (Configuration)
# ==========================================
input_folder = r"C:\Users\lovew\Downloads\Test_violance"
model_path = (
    r"C:\Users\lovew\Downloads\istockphoto-1447876057-640_adpp_is.mp4"
)
output_dir = "all_results"
rows = 6
resize_to = (320, 240)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. เตรียมไฟล์รูปภาพ
# ==========================================
image_paths = []
if os.path.isdir(input_folder):
    image_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
elif os.path.isfile(input_folder):
    image_paths = (
        [input_folder]
        if input_folder.lower().endswith((".png", ".jpg", ".jpeg"))
        else []
    )

if not image_paths:
    print("Error: ไม่พบไฟล์รูปภาพ")
    exit()

# ==========================================
# 3. โหลดโมเดลและเริ่มตรวจจับ
# ==========================================
print(f"กำลังโหลดโมเดล...")
model = YOLO(model_path)
detected_images = []

count_found = 0
count_not_found = 0

print(f"กำลังประมวลผล {len(image_paths)} รูป...")

for path in image_paths:
    results = model(path)

    # ตรวจสอบว่าในรูปนี้มีวัตถุ (boxes) หรือไม่
    if len(results[0].boxes) > 0:
        count_found += 1
    else:
        count_not_found += 1

    result_img = results[0].plot()
    small_img = cv2.resize(result_img, resize_to)
    detected_images.append(small_img)

# ==========================================
# 4. จัดทำ Grid และส่วนสรุปผล
# ==========================================
total_imgs = len(detected_images)
cols = math.ceil(total_imgs / rows)

# สร้างแถบสีขาวด้านบนเพื่อเขียนสรุป (Header)
header_h = 60
header_w = cols * resize_to[0]
header = np.full((header_h, header_w, 3), 255, dtype=np.uint8)

text_summary = (
    f"Total: {total_imgs} | Found: {count_found} | Not Found: {count_not_found}"
)
cv2.putText(header, text_summary, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# เติมภาพสีดำให้เต็ม Grid
blank = np.zeros((resize_to[1], resize_to[0], 3), dtype=np.uint8)
while len(detected_images) < rows * cols:
    detected_images.append(blank)

# รวมภาพตามแนวตั้งและแนวนอน
row_imgs = []
for i in range(0, len(detected_images), cols):
    row = cv2.hconcat(detected_images[i : i + cols])
    row_imgs.append(row)

grid_body = cv2.vconcat(row_imgs)

# นำ Header มาต่อกับ Grid Body
final_image = cv2.vconcat([header, grid_body])

# ==========================================
# 5. บันทึกผลลัพธ์
# ==========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"result_{timestamp}.png"
output_path = os.path.join(output_dir, filename)

cv2.imwrite(output_path, final_image)

print("-" * 30)
print(f"ประมวลผลเสร็จสิ้น!")
print(f"พบวัตถุ: {count_found} รูป / ไม่พบ: {count_not_found} รูป")
print(f"บันทึกไฟล์ไปที่: {output_path}")
print("-" * 30)

os.startfile(output_path)
