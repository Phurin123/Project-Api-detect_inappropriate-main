from PIL import Image
import os

# โฟลเดอร์ที่เก็บรูป
folder_path = r"C:\Users\lovew\OneDrive\เอกสาร\open huuse"  # เปลี่ยนเป็น path ของคุณ

# โหลดรูปทั้งหมดในโฟลเดอร์ (เฉพาะ .jpg, .png)
images = []
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(folder_path, file_name)
        images.append(Image.open(img_path))

# ถ้าอยากเอาแค่ 8 รูปแรก
images = images[:8]

# กำหนด grid
cols = 4
rows = 2

# ขนาดรูปแต่ละรูป (resize ให้เท่ากัน)
width, height = 200, 200
images = [img.resize((width, height)) for img in images]

# สร้าง canvas สำหรับภาพรวม
result = Image.new("RGB", (cols * width, rows * height))

# วางรูปลง canvas
for index, img in enumerate(images):
    x = (index % cols) * width
    y = (index // cols) * height
    result.paste(img, (x, y))

# บันทึกภาพรวม
result.save(os.path.join(folder_path, "merged_image.jpg"))
result.show()
