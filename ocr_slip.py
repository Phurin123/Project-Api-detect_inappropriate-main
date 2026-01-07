import cv2
import numpy as np
import re
from fuzzywuzzy import fuzz
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import hashlib
import threading
from collections import OrderedDict
import requests
import json
from dotenv import load_dotenv
import tempfile

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedSlipOCR:
    def __init__(self):
        """เริ่มต้นระบบ OCR ที่ทันสมัย"""
        # โหลด environment variables
        load_dotenv()
        self.api_key = os.getenv("TYPHOON_OCR_API_KEY")
        if not self.api_key:
            logger.warning("TYPHOON_OCR_API_KEY not found in environment variables")
        else:
            logger.info("Typhoon OCR API initialized successfully")
        
        # Typhoon OCR API configuration
        self.ocr_url = "https://api.opentyphoon.ai/v1/ocr"
        self.model = "typhoon-ocr"
        self.task_type = "default"
        self.max_tokens = 16384
        self.temperature = 0.1
        self.top_p = 0.6
        self.repetition_penalty = 1.2

        self.thai_months = {
            # มกราคม (January)
            "มกราคม": "01",
            "ม.ค.": "01",
            "ม.ค": "01",
            "มค": "01",
            # กุมภาพันธ์ (February)
            "กุมภาพันธ์": "02",
            "ก.พ.": "02",
            "ก.พ": "02",
            "กพ": "02",
            # มีนาคม (March)
            "มีนาคม": "03",
            "มี.ค.": "03",
            "มี.ค": "03",
            "มีค": "03",
            # เมษายน (April)
            "เมษายน": "04",
            "เม.ย.": "04",
            "เม.ย": "04",
            "เมย": "04",
            # พฤษภาคม (May)
            "พฤษภาคม": "05",
            "พ.ค.": "05",
            "พ.ค": "05",
            "พค": "05",
            # มิถุนายน (June)
            "มิถุนายน": "06",
            "มิ.ย.": "06",
            "มิ.ย": "06",
            "มิย": "06",
            # กรกฎาคม (July)
            "กรกฎาคม": "07",
            "ก.ค.": "07",
            "ก.ค": "07",
            "กค": "07",
            # สิงหาคม (August)
            "สิงหาคม": "08",
            "ส.ค.": "08",
            "ส.ค": "08",
            "สค": "08",
            # กันยายน (September)
            "กันยายน": "09",
            "ก.ย.": "09",
            "ก.ย": "09",
            "กย": "09",
            # ตุลาคม (October)
            "ตุลาคม": "10",
            "ต.ค.": "10",
            "ต.ค": "10",
            "ตค": "10",
            # พฤศจิกายน (November)
            "พฤศจิกายน": "11",
            "พ.ย.": "11",
            "พ.ย": "11",
            "พย": "11",
            # ธันวาคม (December)
            "ธันวาคม": "12",
            "ธ.ค.": "12",
            "ธ.ค": "12",
            "ธค": "12",
        }
        self._cache_enabled = os.getenv("OCR_CACHE_ENABLED", "true").lower() == "true"
        self._cache_max_entries = max(int(os.getenv("OCR_CACHE_SIZE", "8")), 1)
        self._cache_lock = threading.Lock()
        self._cache: "OrderedDict[str, Dict]" = OrderedDict()

    def _hash_image(self, image: np.ndarray) -> str:
        return hashlib.sha256(image.tobytes()).hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict]:
        if not self._cache_enabled:
            return None
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            self._cache.move_to_end(key)
            return dict(cached)

    def _cache_set(self, key: str, value: Dict) -> None:
        if not self._cache_enabled:
            return
        with self._cache_lock:
            self._cache[key] = dict(value)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max_entries:
                self._cache.popitem(last=False)



    def extract_text_with_typhoon_ocr(self, image: np.ndarray) -> str:
        """ใช้ Typhoon OCR API เพื่อดึงข้อความจากภาพ"""
        if not self.api_key:
            logger.error("Typhoon OCR API key not configured")
            return ""
        
        try:
            # บันทึกภาพเป็นไฟล์ชั่วคราว
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_path = tmp_file.name
                # แปลง numpy array เป็น PIL Image และบันทึก
                if len(image.shape) == 2:
                    # Grayscale
                    pil_image = Image.fromarray(image)
                else:
                    # BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                pil_image.save(temp_path, format='JPEG', quality=95)
            
            # เตรียมข้อมูลสำหรับ API request
            with open(temp_path, 'rb') as file:
                files = {'file': file}
                data = {
                    'model': self.model,
                    'task_type': self.task_type,
                    'max_tokens': str(self.max_tokens),
                    'temperature': str(self.temperature),
                    'top_p': str(self.top_p),
                    'repetition_penalty': str(self.repetition_penalty)
                }
                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }
                
                # ส่ง request ไปยัง API
                response = requests.post(self.ocr_url, files=files, data=data, headers=headers, timeout=30)
            
            # ลบไฟล์ชั่วคราว
            try:
                os.unlink(temp_path)
            except:
                pass
            
            if response.status_code == 200:
                result = response.json()
                
                # ดึงข้อความจาก response
                extracted_texts = []
                for page_result in result.get('results', []):
                    if page_result.get('success') and page_result.get('message'):
                        content = page_result['message']['choices'][0]['message']['content']
                        try:
                            # พยายาม parse เป็น JSON ถ้ามันเป็น structured output
                            parsed_content = json.loads(content)
                            text = parsed_content.get('natural_text', content)
                        except json.JSONDecodeError:
                            text = content
                        extracted_texts.append(text)
                    elif not page_result.get('success'):
                        error_msg = page_result.get('error', 'Unknown error')
                        logger.error(f"OCR processing error: {error_msg}")
                
                return '\n'.join(extracted_texts)
            else:
                logger.error(f"Typhoon OCR API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Typhoon OCR error: {e}")
            return ""

    def extract_time(self, text: str) -> Optional[str]:
        time_patterns = [
            r"(\d{1,2})[:\.\-](\d{2})",
            r"(\d{1,2})[:\.\-](\d{2})[:\.\-](\d{2})",
            r"(\d{1,2})[:\.\-](\d{2})\s*(น\.|นาที|min|hr|hour)",
        ]
        valid_times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2:
                    try:
                        hour, minute = int(match[0]), int(match[1])
                        if 0 <= hour <= 23 and 0 <= minute <= 59:
                            valid_times.append(f"{hour:02d}:{minute:02d}")
                    except ValueError:
                        continue
        for time_str in sorted(valid_times, reverse=True):
            if time_str not in ["00:00", "01:00"]:
                return time_str
        return valid_times[-1] if valid_times else None

    def extract_date(self, text: str) -> Optional[str]:
        date_patterns = [
            r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})",
            r"(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})",
            r"(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})",
        ]
        thai_date_patterns = [
            r"(\d{1,2})\s*([ก-๙]{1,3})\.?\s*(\d{2,4})",
            r"(\d{1,2})([ก-๙]{1,3})\.(\d{2,4})",
            r"(\d{1,2})\s*([ก-๙]{2,})\s*(\d{2,4})",
            r"(\d{1,2})\s*([ก-๙]{1,10})\.?\s*(\d{2,4})",
            r"(\d{1,2})\s*([ก-๙]{1,}\.?)\s*(\d{2,4})",
            r"\d{2}\s(?:ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.)\s\d{4}",
            r"(\d{1,2})\s*([ก-๙\.]{2,5})\s*(\d{2,4})",
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3 and self.is_valid_date(match):
                    return f"{match[0]}/{match[1]}/{match[2]}"

        for pattern in thai_date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3:
                    day, thai_month, year_str = match
                    clean_month = re.sub(r"[^\u0E00-\u0E7F\.]", "", thai_month)
                    month_num = None
                    for key, value in self.thai_months.items():
                        if key in clean_month or clean_month.startswith(
                            key.replace(".", "")
                        ):
                            month_num = value
                            break
                    if month_num:
                        year_int = int(year_str)
                        if year_int < 100:
                            year_ad = year_int + 1957
                        elif year_int >= 2500:
                            year_ad = year_int - 543
                        else:
                            year_ad = year_int
                        date_tuple = (day, month_num, str(year_ad))
                        if self.is_valid_date(date_tuple):
                            return f"{int(day):02d}/{month_num}/{year_ad}"
        return None

    def is_valid_date(self, date_parts: Tuple[str, str, str]) -> bool:
        try:
            day, month, year = map(int, date_parts)
            if not (1 <= month <= 12):
                return False
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if month == 2 and ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0):
                days_in_month[1] = 29
            if not (1 <= day <= days_in_month[month - 1]):
                return False
            return True
        except:
            return False

    def extract_amount(self, text: str) -> Optional[str]:
        amount_patterns = [
            r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(บาท|฿|THB|baht)",
            r"(?:ราคา|จำนวน|ยอด|รวม|Total|Amount|Price)[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
            r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:บาท|฿|THB|baht)?",
        ]
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    amount = match[0] if isinstance(match, tuple) else match
                    amount_clean = re.sub(r"[^\d,\.]", "", amount)
                    if amount_clean:
                        return amount_clean
        return None

    def extract_name(
        self, text: str, expected_names: List[str] = None
    ) -> Optional[str]:
        if expected_names is None:
            expected_names = ["ภูรินทร์สุขมั่น", "ภูรินทร์", "สุขมั่น"]
        thai_words = re.findall(r"[ก-๙]{2,}", text)
        best_match = None
        best_score = 0
        for word in thai_words:
            for expected_name in expected_names:
                score = fuzz.ratio(word, expected_name)
                partial_score = fuzz.partial_ratio(word, expected_name)
                token_score = fuzz.token_sort_ratio(word, expected_name)
                total_score = (score + partial_score + token_score) / 3
                if total_score > best_score and total_score > 70:
                    best_score = total_score
                    best_match = expected_name
        return best_match

    def process_image(
        self, image: np.ndarray, expected_names: List[str] = None
    ) -> Dict:
        try:
            # ใช้ Typhoon OCR API โดยตรง (API จัดการ preprocessing เอง)
            all_text = self.extract_text_with_typhoon_ocr(image)
            all_text = re.sub(r"\s+", " ", all_text).strip()

            result = {
                "time": self.extract_time(all_text),
                "date": self.extract_date(all_text),
                "amount": self.extract_amount(all_text),
                "full_name": self.extract_name(all_text, expected_names),
                "full_text": all_text,
            }

            logger.info("OCR processing completed with Typhoon OCR API")
            return result

        except Exception as e:
            logger.error(f"Error in process_image: {e}")
            return {
                "error": str(e),
                "full_text": "",
                "time": None,
                "date": None,
                "amount": None,
                "full_name": None,
            }



    def extract_info(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        result = self.process_image(image)

        formatted_result = {
            "time": result.get("time"),
            "date": result.get("date"),
            "amount": result.get("amount"),
            "full_name": result.get("full_name"),
            "time_receipts": result.get("time"),
            "full_text": result.get("full_text", ""),
        }

        return formatted_result


# ส่วน main() ไม่เปลี่ยนมาก — แต่ลบการพิมพ์ receipt_number, merchant, confidence
def main():
    ocr = AdvancedSlipOCR()
    kasikor = r"C:\Users\khongkaphan\Downloads\สื่อ (12) (1).jpeg"
    krungthai = r"C:\Users\khongkaphan\Downloads\สื่อ (1).jpg"
    aomsin = r"C:\Users\khongkaphan\Downloads\สื่อ (5).jpg"
    krungthep = r"C:\Users\khongkaphan\Downloads\สื่อ (3).jpg"
    krungsri = r"C:\Users\lovew\OneDrive\รูปภาพ\slip-test\retest 1.png"
    thaipanit = r"C:\Users\khongkaphan\Downloads\สื่อ (8).jpg"
    gg = r"C:\Users\lovew\OneDrive\รูปภาพ\slip-test\retest 1.png"

    image_path = gg

    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is None:
            try:
                image = np.array(Image.open(image_path))
                print("อ่านภาพด้วย Pillow สำเร็จ")
            except Exception as e:
                print(f"ไม่สามารถเปิดไฟล์ด้วย Pillow ได้: {e}")
                image = None

        if image is not None:
            result = ocr.process_image(image)

            print("=== ผลการประมวลผล OCR ===")
            print(f"เวลา: {result.get('time', 'ไม่พบ')}")
            print(f"วันที่: {result.get('date', 'ไม่พบ')}")
            print(f"จำนวนเงิน: {result.get('amount', 'ไม่พบ')}")
            print(f"ชื่อ: {result.get('full_name', 'ไม่พบ')}")
            print(f"\nข้อความทั้งหมด:\n{result.get('full_text', 'ไม่พบ')}")

            # # บันทึกเฉพาะภาพสุดท้าย (Otsu)
            # ocr.save_final_processed_image(image, "final_otsu.jpg")
        else:
            print("ไม่สามารถอ่านภาพได้")
    else:
        print(f"ไม่พบไฟล์ภาพ: {image_path}")


if __name__ == "__main__":
    main()
