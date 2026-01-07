import cv2


def check_qrcode(image_path: str) -> bool:
    image = cv2.imread(image_path)
    if image is None:
        return False
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(image)
    return points is not None and bool(data)


# ==== ตัวอย่างการใช้งาน ====
image_path = r"C:\Users\lovew\OneDrive\รูปภาพ\slip-test\retest 1.png"  # เปลี่ยนเป็น path ของภาพที่คุณอยากทดสอบ
if check_qrcode(image_path):
    print("เจอ QR Code!")
else:
    print("ไม่พบ QR Code")
