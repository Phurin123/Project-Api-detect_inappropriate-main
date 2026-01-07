import easyocr
reader = easyocr.Reader(['th','en'], gpu=True)
print("EasyOCR device:", reader.device)
