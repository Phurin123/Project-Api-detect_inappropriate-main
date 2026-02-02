import base64
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import concurrent.futures
import json
from json import JSONDecodeError
import mimetypes
import os
import struct
import random
import re
import secrets
import shutil
import smtplib
import tempfile
import uuid
import zipfile
from urllib.parse import urlparse
import cv2
import jwt
import numpy as np
import requests
import torch
import imageio
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image
from promptpay import qrcode
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from werkzeug.security import check_password_hash, generate_password_hash
from zoneinfo import ZoneInfo
from ocr_slip import AdvancedSlipOCR
from ultralytics import YOLO

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

HOMEPAGE_DIR = BASE_DIR / "homepage"

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".jfif",
}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CONTENT_TYPE_EXTENSION_MAP = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
    "image/tiff": ".tiff",
    "image/x-icon": ".ico",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "application/zip": ".zip",
}

ALLOWED_OUTPUT_MODES = {"blur", "bbox"}

try:
    MAX_CONCURRENT_ANALYSIS = max(int(os.getenv("MAX_CONCURRENT_ANALYSIS", "2")), 1)
except ValueError:
    MAX_CONCURRENT_ANALYSIS = 2

try:
    MAX_ANALYSIS_QUEUE = int(os.getenv("MAX_ANALYSIS_QUEUE", "10"))
    if MAX_ANALYSIS_QUEUE < 0:
        MAX_ANALYSIS_QUEUE = 0
except ValueError:
    MAX_ANALYSIS_QUEUE = 10


class ConcurrencyLimiter:
    def __init__(self, max_concurrency: int, max_waiting: int):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_waiting = max_waiting
        self._waiting = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        waiter_registered = False
        if self._max_waiting > 0:
            async with self._lock:
                if self._waiting >= self._max_waiting:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="ระบบกำลังประมวลผลคำขอจำนวนมาก กรุณาลองใหม่อีกครั้ง",
                    )
                self._waiting += 1
                waiter_registered = True
        try:
            await self._semaphore.acquire()
        finally:
            if waiter_registered:
                async with self._lock:
                    self._waiting = max(self._waiting - 1, 0)

    def release(self) -> None:
        self._semaphore.release()

    async def __aenter__(self) -> "ConcurrencyLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


analysis_concurrency_limiter = ConcurrencyLimiter(
    MAX_CONCURRENT_ANALYSIS, MAX_ANALYSIS_QUEUE
)

API_KEY_SECRET = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY") or "secret"
API_BASE_URL = os.getenv("API_BASE_URL", "")

MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT = int(os.getenv("MAIL_PORT", "587"))
MAIL_USE_TLS = os.getenv("MAIL_USE_TLS", "true").lower() == "true"
MAIL_USERNAME = os.getenv("EMAIL_USER")
MAIL_PASSWORD = os.getenv("EMAIL_PASS")
MAIL_DEFAULT_SENDER = os.getenv(
    "MAIL_DEFAULT_SENDER", MAIL_USERNAME or "no-reply@example.com"
)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client: MongoClient = MongoClient(MONGO_URI)
db: Database = client["api_database"]
users_collection: Collection = db["users"]
api_keys_collection: Collection = db["api_keys"]
orders_collection: Collection = db["orders"]
otp_collection: Collection = db["otp_reset"]
uploaded_files_collection: Collection = db["uploaded_files"]
api_key_usage_collection: Collection = db["api_key_usage"]

uploaded_files_collection.create_index([("created_at", 1)], expireAfterSeconds=3600)
api_key_usage_collection.create_index([("api_key", 1), ("created_at", -1)])
api_key_usage_collection.create_index([("email", 1), ("created_at", -1)])
orders_collection.create_index([("email", 1), ("paid", 1), ("created_time", -1)])
api_keys_collection.create_index([("expires_at", 1)], expireAfterSeconds=0)

TEST_PLAN_DURATION_DAYS = 7
PREMIUM_PLAN_PACKAGES: Dict[str, Dict[str, Any]] = {
    "image": {"media_access": ["image"], "monthly_price": 79},
    "video": {"media_access": ["video"], "monthly_price": 119},
    "both": {"media_access": ["image", "video"], "monthly_price": 159},
}

# Video processing optimization: process every Nth frame
try:
    VIDEO_FRAME_SKIP = max(int(os.getenv("VIDEO_FRAME_SKIP", "2")), 1)
except ValueError:
    VIDEO_FRAME_SKIP = 2


def sanitize_filename(filename: str) -> str:
    return Path(filename or "upload").name


def allowed_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def send_email_message(subject: str, body: str, recipients: List[str]) -> None:
    if not MAIL_USERNAME or not MAIL_PASSWORD:
        raise RuntimeError("Email credentials are not configured.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = MAIL_DEFAULT_SENDER
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
        if MAIL_USE_TLS:
            server.starttls()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.send_message(msg)


async def extract_request_payload(request: Request) -> Dict[str, Any]:
    """
    Safely extract JSON or form payloads regardless of Content-Type header.
    """
    try:
        return await request.json()
    except (JSONDecodeError, ValueError):
        form = await request.form()
        return {key: form.get(key) for key in form.keys()}


def generate_token(email: str) -> str:
    payload = {"email": email, "exp": datetime.utcnow() + timedelta(hours=1)}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    print(f"Generated token for {email}: {token}")
    return token


def decode_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])


def save_bytes_to_uploads(
    data: bytes, extension: str, original_name: str
) -> Dict[str, str]:
    suffix = extension if extension.startswith(".") else f".{extension}"
    filename = f"{uuid.uuid4()}{suffix.lower()}"
    file_path = UPLOAD_FOLDER / filename
    with open(file_path, "wb") as fh:
        fh.write(data)
    uploaded_files_collection.insert_one(
        {"filename": filename, "created_at": datetime.utcnow()}
    )
    return {
        "file_path": str(file_path),
        "stored_filename": filename,
        "original_filename": original_name,
    }


async def save_upload_file(
    upload: UploadFile, original_name: Optional[str] = None
) -> Dict[str, str]:
    original_name = original_name or upload.filename or "upload"
    ext = Path(original_name).suffix.lower()
    if not ext:
        ext = Path(upload.filename or "").suffix.lower()
    if not ext:
        ext = ".bin"
    filename = f"{uuid.uuid4()}{ext}"
    file_path = UPLOAD_FOLDER / filename
    content = await upload.read()
    with open(file_path, "wb") as fh:
        fh.write(content)
    await upload.close()
    uploaded_files_collection.insert_one(
        {"filename": filename, "created_at": datetime.utcnow()}
    )
    return {
        "file_path": str(file_path),
        "stored_filename": filename,
        "original_filename": original_name,
    }


def remove_stored_file(file_record: Dict[str, Any]) -> None:
    stored_filename = file_record.get("stored_filename")
    file_path = file_record.get("file_path")
    if stored_filename:
        uploaded_files_collection.delete_one({"filename": stored_filename})
    if file_path:
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            pass


STREAM_CHUNK_SIZE = 1024 * 1024


def parse_range_header(range_header: str, file_size: int) -> Optional[Tuple[int, int]]:
    if not range_header or not range_header.startswith("bytes="):
        return None
    ranges = range_header.replace("bytes=", "", 1).split(",")[0].strip()
    if "-" not in ranges:
        return None
    start_str, end_str = ranges.split("-", 1)
    try:
        if start_str:
            start = int(start_str)
        else:
            length = int(end_str)
            if length <= 0:
                return None
            start = max(file_size - length, 0)
        if end_str:
            end = int(end_str)
        else:
            end = file_size - 1
    except ValueError:
        return None

    if start < 0 or end < start or end >= file_size:
        return None
    return start, end


def iter_file_chunks(path: Path, start: int, end: int) -> Iterable[bytes]:
    with path.open("rb") as file_obj:
        file_obj.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = file_obj.read(min(STREAM_CHUNK_SIZE, remaining))
            if not chunk:
                break
            yield chunk
            remaining -= len(chunk)


class ImageIOVideoWriter:
    """Video writer that uses imageio/ffmpeg to produce H.264 compatible files."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        if imageio is None:
            raise RuntimeError("imageio is not available")
        if fps <= 0:
            fps = 25.0
        self._writer = imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
            output_params=["-movflags", "+faststart"],
        )
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._writer.append_data(rgb_frame)

    def release(self) -> None:
        if not self._closed:
            self._writer.close()
            self._closed = True


def create_video_writer(path: Path, fps: float, width: int, height: int):
    if imageio is not None:
        try:
            return ImageIOVideoWriter(path, fps, width, height)
        except Exception as exc:
            print(f"[video-writer] imageio fallback for {path.name}: {exc}")
    for codec in ("mp4v", "XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Unable to initialize video writer for {path.name}")


def patch_moov_offsets_inplace(buffer: memoryview, shift: int) -> None:
    container_atoms = {
        b"moov",
        b"trak",
        b"mdia",
        b"minf",
        b"stbl",
        b"edts",
        b"udta",
        b"mvex",
    }
    offset = 0
    length = len(buffer)
    while offset + 8 <= length:
        atom_size = struct.unpack(">I", buffer[offset : offset + 4])[0]
        atom_type = bytes(buffer[offset + 4 : offset + 8])
        header_size = 8
        if atom_size == 1:
            if offset + 16 > length:
                raise ValueError("Invalid extended atom header in moov atom")
            atom_size = struct.unpack(">Q", buffer[offset + 8 : offset + 16])[0]
            header_size = 16
        if atom_size == 0:
            atom_size = length - offset
        atom_end = offset + atom_size
        if atom_end > length:
            raise ValueError("Corrupted atom size inside moov atom")
        data_start = offset + header_size
        data_view = buffer[data_start:atom_end]
        if atom_type in container_atoms:
            patch_moov_offsets_inplace(data_view, shift)
        elif atom_type == b"stco":
            if len(data_view) < 8:
                raise ValueError("Invalid stco atom length")
            entry_count = struct.unpack(">I", data_view[4:8])[0]
            pos = 8
            for _ in range(entry_count):
                if pos + 4 > len(data_view):
                    raise ValueError("Invalid stco entry")
                value = struct.unpack(">I", data_view[pos : pos + 4])[0] + shift
                data_view[pos : pos + 4] = struct.pack(">I", value)
                pos += 4
        elif atom_type == b"co64":
            if len(data_view) < 8:
                raise ValueError("Invalid co64 atom length")
            entry_count = struct.unpack(">I", data_view[4:8])[0]
            pos = 8
            for _ in range(entry_count):
                if pos + 8 > len(data_view):
                    raise ValueError("Invalid co64 entry")
                value = struct.unpack(">Q", data_view[pos : pos + 8])[0] + shift
                data_view[pos : pos + 8] = struct.pack(">Q", value)
                pos += 8
        offset += atom_size


def optimize_mp4_faststart(path: Path) -> None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        print(f"[faststart] unable to read {path.name}: {exc}")
        return

    atoms: List[Tuple[bytes, bytes]] = []
    offset = 0
    moov_atom: Optional[bytes] = None
    moov_index = -1
    mdat_index = -1

    while offset + 8 <= len(data):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        atom_type = data[offset + 4 : offset + 8]
        header_size = 8
        if size == 1:
            if offset + 16 > len(data):
                print(f"[faststart] invalid extended atom in {path.name}")
                return
            size = struct.unpack(">Q", data[offset + 8 : offset + 16])[0]
            header_size = 16
        if size == 0:
            size = len(data) - offset
        atom_end = offset + size
        if atom_end > len(data):
            print(f"[faststart] atom extends beyond file end in {path.name}")
            return
        atom_bytes = data[offset:atom_end]
        atoms.append((atom_type, atom_bytes))
        if atom_type == b"moov":
            moov_atom = atom_bytes
            moov_index = len(atoms) - 1
        elif atom_type == b"mdat" and mdat_index == -1:
            mdat_index = len(atoms) - 1
        offset = atom_end

    if not atoms or moov_atom is None or mdat_index == -1:
        return
    if moov_index < mdat_index:
        return

    moov_bytes = bytearray(moov_atom)
    shift = len(moov_bytes)
    if shift <= 0:
        return
    try:
        patch_moov_offsets_inplace(memoryview(moov_bytes), shift)
    except ValueError as exc:
        print(f"[faststart] skip {path.name}: {exc}")
        return

    new_atoms: List[bytes] = []
    inserted = False
    for index, (atom_type, atom_bytes) in enumerate(atoms):
        if atom_type == b"moov":
            continue
        if not inserted and index == mdat_index:
            new_atoms.append(bytes(moov_bytes))
            inserted = True
        new_atoms.append(atom_bytes)
    if not inserted:
        new_atoms.append(bytes(moov_bytes))

    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("wb") as output:
            for chunk in new_atoms:
                output.write(chunk)
        temp_path.replace(path)
    except OSError as exc:
        print(f"[faststart] failed to rewrite {path.name}: {exc}")
        temp_path.unlink(missing_ok=True)


def is_image(file_path: str) -> bool:
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False


def log_api_key_usage_event(
    api_key: str,
    email: str,
    analysis_types: List[str],
    thresholds: Dict[str, float],
    result: Dict[str, Any],
) -> None:
    thresholds_log = {k: float(v) for k, v in (thresholds or {}).items()}
    media_type = str(result.get("media_type") or "image").lower()
    payload = {
        "api_key": api_key,
        "email": email,
        "original_filename": result.get("original_filename"),
        "stored_filename": result.get("stored_filename"),
        "processed_filename": result.get("processed_filename"),
        "blurred_filename": result.get("blurred_filename"),
        "status": result.get("status"),
        "detections": result.get("detections", []),
        "analysis_types": analysis_types or [],
        "thresholds": thresholds_log,
        "media_type": media_type,
        "output_modes": result.get("output_modes", []),
        "media_access": result.get("media_access", []),
        "created_at": datetime.utcnow(),
    }
    api_key_usage_collection.insert_one(payload)


def load_model(model_name: str) -> YOLO:
    model_path = BASE_DIR / "models" / model_name
    return YOLO(str(model_path))


models = {
    "porn": load_model("โป๊เปลือยดีจัดเลียๆๆๆๆ.pt"),
    "weapon": load_model("อาวุธดีจัดปั้งงงงงๆ.pt"),
    "cigarette": load_model("บุหรี่ของดีจัดสูดๆๆๆ.pt"),
    "violence": load_model("ความรุนเเรงต่อยๆๆๆๆ.pt"),
}


def run_models_on_frame(
    image_bgr: np.ndarray, model_types: List[str], thresholds: Dict[str, float]
) -> List[Dict[str, Any]]:
    def run_model(model_type: str) -> Tuple[str, List[Dict[str, Any]]]:
        model = models[model_type]
        threshold = float(thresholds.get(model_type, 0.5))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = model.predict(
            source=image_bgr,
            imgsz=640,
            device=device,
            conf=threshold,
            verbose=False,
            save=False,
            stream=False,
        )
        detections_local: List[Dict[str, Any]] = []
        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf)
                if confidence < threshold:
                    continue
                label_name = model.names[int(box.cls)].lower()
                x1, y1, x2, y2 = box.xyxy[0]
                bbox = [round(float(coord), 2) for coord in [x1, y1, x2, y2]]
                detections_local.append(
                    {
                        "label": label_name,
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                        "model_type": model_type,
                    }
                )
        return model_type, detections_local

    detections: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_model, model_type) for model_type in model_types]
        for future in concurrent.futures.as_completed(futures):
            _, model_detections = future.result()
            detections.extend(model_detections)
    return detections


def draw_bounding_boxes_np(
    image_bgr: np.ndarray, detections: List[Dict[str, Any]]
) -> np.ndarray:
    output = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        label = detection.get("label", "")
        confidence = detection.get("confidence", 0.0)
        h, w = output.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({confidence:.2f})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Calculate text position
        text_w, text_h = text_size
        if y1 - text_h - 10 < 0:
            # Draw text inside the box if it doesn't fit above
            text_origin = (x1, y1 + text_h + 5)
            rect_pt1 = (x1, y1)
            rect_pt2 = (x1 + text_w, y1 + text_h + 10)
        else:
            # Draw text above the box
            text_origin = (x1, y1 - 5)
            rect_pt1 = (x1, y1 - text_h - 10)
            rect_pt2 = (x1 + text_w, y1)

        # Draw text background
        cv2.rectangle(
            output,
            rect_pt1,
            rect_pt2,
            (0, 255, 0),
            -1,
        )
        # Draw text
        cv2.putText(
            output,
            text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
    return output


def blur_detected_areas_np(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    blur_ksize: Tuple[int, int] = (51, 51),
) -> np.ndarray:
    blurred_image = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        h, w = blurred_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        roi = blurred_image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi_blurred = cv2.GaussianBlur(roi, blur_ksize, 0)
        blurred_image[y1:y2, x1:x2] = roi_blurred
    return blurred_image


def process_selected_models(
    image: Image.Image, model_types: List[str], thresholds: Dict[str, float]
) -> Tuple[Image.Image, Image.Image, List[Dict[str, Any]]]:
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detections = run_models_on_frame(image_bgr, model_types, thresholds)
    output_bbox = draw_bounding_boxes_np(image_bgr, detections)
    output_blur = blur_detected_areas_np(image_bgr, detections)
    output_bbox_image = Image.fromarray(cv2.cvtColor(output_bbox, cv2.COLOR_BGR2RGB))
    output_blur_image = Image.fromarray(cv2.cvtColor(output_blur, cv2.COLOR_BGR2RGB))
    return output_bbox_image, output_blur_image, detections


def process_image_file_for_models(
    file_path: str, model_types: List[str], thresholds: Dict[str, float]
) -> Tuple[Image.Image, Image.Image, List[Dict[str, Any]]]:
    path = Path(file_path)
    with Image.open(path) as raw_image:
        image_rgb = raw_image.convert("RGB")
    return process_selected_models(image_rgb, model_types, thresholds)


def process_video_media(
    video_path: Path,
    model_types: List[str],
    thresholds: Dict[str, float],
    include_bbox: bool,
    include_blur: bool,
) -> Tuple[Optional[Path], Optional[Path], List[Dict[str, Any]], Dict[str, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open video file")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    processed_filename: Optional[str] = None
    blurred_filename: Optional[str] = None
    processed_path: Optional[Path] = None
    blurred_path: Optional[Path] = None
    writer_processed: Optional[Any] = None
    writer_blurred: Optional[Any] = None
    try:
        if include_bbox:
            processed_filename = f"processed_{uuid.uuid4()}.mp4"
            processed_path = UPLOAD_FOLDER / processed_filename
            writer_processed = create_video_writer(processed_path, fps, width, height)
        if include_blur:
            blurred_filename = f"blurred_{uuid.uuid4()}.mp4"
            blurred_path = UPLOAD_FOLDER / blurred_filename
            writer_blurred = create_video_writer(blurred_path, fps, width, height)
    except Exception as writer_exc:
        capture.release()
        if writer_processed:
            writer_processed.release()
        if writer_blurred:
            writer_blurred.release()
        raise RuntimeError(
            f"Failed to initialize video writers: {writer_exc}"
        ) from writer_exc

    detections_per_frame: List[Dict[str, Any]] = []
    aggregated: Dict[str, int] = defaultdict(int)

    frame_index = 0
    last_bbox_frame = None
    last_blurred_frame = None
    last_detections = []

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            # Process frame only if it's the Nth frame (based on VIDEO_FRAME_SKIP)
            should_process = (frame_index % VIDEO_FRAME_SKIP) == 0

            if should_process:
                # Run AI detection on this frame
                detections = run_models_on_frame(frame, model_types, thresholds)
                last_detections = detections  # Cache for skipped frames

                if writer_processed is not None:
                    bbox_frame = draw_bounding_boxes_np(frame, detections)
                    last_bbox_frame = bbox_frame
                    writer_processed.write(bbox_frame)

                if writer_blurred is not None:
                    blurred_frame = blur_detected_areas_np(frame, detections)
                    last_blurred_frame = blurred_frame
                    writer_blurred.write(blurred_frame)
            else:
                # Skip AI processing, reuse previous frame's detection results
                detections = last_detections

                # Write the raw frame with previous detections overlaid
                if writer_processed is not None:
                    if last_bbox_frame is not None:
                        # Apply previous detections to current frame
                        bbox_frame = draw_bounding_boxes_np(frame, detections)
                        writer_processed.write(bbox_frame)
                    else:
                        writer_processed.write(frame)

                if writer_blurred is not None:
                    if last_blurred_frame is not None:
                        # Apply previous blur to current frame
                        blurred_frame = blur_detected_areas_np(frame, detections)
                        writer_blurred.write(blurred_frame)
                    else:
                        writer_blurred.write(frame)

            # Aggregate detections for summary
            summary = []
            for detection in detections:
                label = detection.get("label")
                if label:
                    aggregated[label] += 1
                summary.append(
                    {
                        "label": detection.get("label"),
                        "confidence": detection.get("confidence"),
                        "bbox": detection.get("bbox"),
                        "model_type": detection.get("model_type"),
                    }
                )
            detections_per_frame.append({"frame": frame_index, "detections": summary})
            frame_index += 1
    finally:
        capture.release()
        if writer_processed:
            writer_processed.release()
        if writer_blurred:
            writer_blurred.release()

    try:
        if processed_path is not None:
            optimize_mp4_faststart(processed_path)
        if blurred_path is not None:
            optimize_mp4_faststart(blurred_path)
    except Exception as exc:
        print(
            f"[faststart] optimization error for {processed_filename} / {blurred_filename}: {exc}"
        )

    if processed_filename:
        uploaded_files_collection.insert_one(
            {"filename": processed_filename, "created_at": datetime.utcnow()}
        )
    if blurred_filename:
        uploaded_files_collection.insert_one(
            {"filename": blurred_filename, "created_at": datetime.utcnow()}
        )

    return processed_path, blurred_path, detections_per_frame, dict(aggregated)


app = FastAPI(title="Objexify API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if HOMEPAGE_DIR.exists():
    app.mount("/homepage", StaticFiles(directory=HOMEPAGE_DIR), name="homepage_static")


@app.middleware("http")
async def block_browser_api_key_usage(request: Request, call_next):
    path = request.url.path
    if path not in ["/analyze-image", "/analyze-video"]:
        return await call_next(request)

    origin = request.headers.get("origin")
    user_agent = request.headers.get("user-agent", "").lower()
    is_browser = any(x in user_agent for x in ["mozilla", "webkit", "gecko"])

    # If the request comes from a browser
    if is_browser:
        if origin in ALLOWED_DEMO_ORIGINS:
            request.state.is_demo_mode = True
            return await call_next(request)
        else:
            # Block browser access even with x-api-key
            return JSONResponse(
                {
                    "error": "Browser access not allowed. Use authorized demo or server-to-server with API key."
                },
                status_code=403,
            )

    # If not from a browser, require x-api-key
    x_api_key = request.headers.get("x-api-key")
    if not x_api_key:
        return JSONResponse({"error": "Missing x-api-key"}, status_code=401)

    api_key_data = api_keys_collection.find_one({"api_key": x_api_key})
    if not api_key_data:
        return JSONResponse({"error": "Invalid API Key"}, status_code=401)

    request.state.api_key_data = api_key_data
    return await call_next(request)


async def get_current_user(
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is missing"
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )

    token = parts[1]
    try:
        data = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    current_user = users_collection.find_one({"email": data["email"]})
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return current_user


async def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
) -> Dict[str, Any]:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API Key"
        )

    api_key_data = api_keys_collection.find_one({"api_key": x_api_key})
    if not api_key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )

    expires_at = api_key_data.get("expires_at")
    if expires_at:
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires_at:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key expired"
            )

    return api_key_data


ALLOWED_DEMO_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.getenv("API_BASE_URL", "").split(",")
    if origin.strip()
}


@app.get("/")
def home() -> FileResponse:
    index_path = HOMEPAGE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Homepage not found"
        )
    return FileResponse(index_path)


@app.get("/homepage/{filename:path}")
def serve_homepage_assets(filename: str) -> FileResponse:
    asset_path = HOMEPAGE_DIR / filename
    if not asset_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
        )
    return FileResponse(asset_path)


@app.get("/uploads/{filename:path}", name="uploaded_file")
def get_uploaded_file(
    filename: str, range_header: Optional[str] = Header(None, alias="Range")
) -> StreamingResponse:
    file_path = UPLOAD_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
        )

    file_size = file_path.stat().st_size
    media_type, _ = mimetypes.guess_type(str(file_path))
    media_type = media_type or "application/octet-stream"

    if range_header:
        byte_range = parse_range_header(range_header, file_size)
        if not byte_range:
            raise HTTPException(
                status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                detail="Invalid range",
            )
        start, end = byte_range
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        }
        return StreamingResponse(
            iter_file_chunks(file_path, start, end),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            media_type=media_type,
            headers=headers,
        )

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
    }
    return StreamingResponse(
        iter_file_chunks(file_path, 0, file_size - 1),
        media_type=media_type,
        headers=headers,
    )


@app.post("/signup")
async def signup(request: Request) -> JSONResponse:
    payload = await extract_request_payload(request)
    email = payload.get("email") or payload.get("username")
    username = payload.get("username") or payload.get("email")
    password = payload.get("password")

    if not email or not username or not password:
        return JSONResponse(
            {"message": "All fields are required"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if users_collection.find_one({"email": email}):
        return JSONResponse(
            {"message": "Email already exists"}, status_code=status.HTTP_400_BAD_REQUEST
        )

    hashed_password = generate_password_hash(
        password, method="pbkdf2:sha256", salt_length=8
    )
    users_collection.insert_one(
        {"email": email, "username": username, "password": hashed_password}
    )
    return JSONResponse(
        {"message": "Signup successful"}, status_code=status.HTTP_201_CREATED
    )


@app.post("/login")
async def login(request: Request) -> JSONResponse:
    payload = await extract_request_payload(request)
    email = payload.get("email")
    password = payload.get("password")

    if not email or not password:
        return JSONResponse(
            {"error": "Email and password are required"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    user = users_collection.find_one({"email": email})
    if not user:
        return JSONResponse(
            {"error": "User not found"}, status_code=status.HTTP_404_NOT_FOUND
        )

    stored_password = user.get("password")
    if stored_password is None:
        return JSONResponse(
            {"error": "This account uses Google login only. Please login with Google."},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if not check_password_hash(stored_password, password):
        return JSONResponse(
            {"error": "Incorrect password"}, status_code=status.HTTP_400_BAD_REQUEST
        )

    token = generate_token(email)
    return JSONResponse({"message": "Login successful", "token": token})


def parse_analysis_types_value(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value if item]
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return []
        try:
            decoded = json.loads(raw_value)
            if isinstance(decoded, list):
                return [str(item) for item in decoded if item]
        except json.JSONDecodeError:
            return [raw_value]
    return [str(raw_value)]


def parse_thresholds_value(raw_value: Any) -> Dict[str, float]:
    if not raw_value:
        return {}
    if isinstance(raw_value, dict):
        result = {}
        for key, value in raw_value.items():
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                continue
        return result
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return {}
        try:
            decoded = json.loads(raw_value)
            if isinstance(decoded, dict):
                return {k: float(v) for k, v in decoded.items()}
        except (json.JSONDecodeError, ValueError, TypeError):
            return {}
    return {}


def parse_output_modes_value(raw_value: Any) -> List[str]:
    if not raw_value:
        return []
    candidates: Iterable[str]
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return []
        try:
            decoded = json.loads(raw_value)
            if isinstance(decoded, list):
                candidates = (str(item) for item in decoded)
            else:
                candidates = [raw_value]
        except json.JSONDecodeError:
            candidates = [raw_value]
    elif (
        isinstance(raw_value, list)
        or isinstance(raw_value, set)
        or isinstance(raw_value, tuple)
    ):
        candidates = (str(item) for item in raw_value)
    else:
        return []
    seen: Set[str] = set()
    modes: List[str] = []
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized in ALLOWED_OUTPUT_MODES and normalized not in seen:
            seen.add(normalized)
            modes.append(normalized)
    return modes


def serialize_datetime(value: Any) -> Optional[str]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    images: List[UploadFile] = File(
        ...,
        description="ไฟล์ภาพหรือ .zip ที่มีภาพ (ส่งได้หลายไฟล์)",
        max_files=100,  # ต้องใช้ FastAPI >= 0.100.0
    ),
    analysis_types: Optional[str] = Form(
        None, description="เช่น `['porn','weapon']` หรือ `porn,weapon`"
    ),
    thresholds: Optional[str] = Form(
        None, description='เช่น `{"porn":0.3,"weapon":0.5}`'
    ),
    output_modes: Optional[str] = Form(
        None, description='เช่น `["bbox","blur"]` หรือ `bbox,blur`'
    ),
):
    # Check if the request is in demo mode
    is_demo = getattr(request.state, "is_demo_mode", False)

    if is_demo:
        # Demo mode: Use safe default values
        api_key_data = {
            "email": "demo@demo.com",
            "api_key": "demo",
            "analysis_types": list(models.keys()),
            "thresholds": {},
            "media_access": ["image", "video"],
            "output_modes": ["bbox", "blur"],
            "plan": "demo",
        }
    else:
        # Normal mode: Require x-api-key
        x_api_key = request.headers.get("x-api-key")
        if not x_api_key:
            raise HTTPException(401, "Missing x-api-key")
        api_key_data = api_keys_collection.find_one({"api_key": x_api_key})
        if not api_key_data:
            raise HTTPException(401, "Invalid API Key")

    # Call the internal image analysis logic
    return await _analyze_image_internal(
        request=request,
        api_key_data=api_key_data,
        images=images,
        analysis_types=analysis_types,
        thresholds=thresholds,
        output_modes=output_modes,
    )


async def _analyze_image_internal(
    request: Request,
    api_key_data: Dict[str, Any],
    images: List[UploadFile],
    analysis_types: Optional[str],
    thresholds: Optional[str],
    output_modes: Optional[str],
):
    files_payload: List[Dict[str, str]] = []
    skipped_entries: List[Dict[str, Any]] = []
    MAX_TOTAL_IMAGES = 100

    try:
        # --- 1. อ่านเนื้อหาไฟล์ทั้งหมดเข้า memory (ครั้งเดียว) ---
        prepared_files = []
        for upload in images:
            user_provided_name = upload.filename or "upload"
            original_name = sanitize_filename(user_provided_name)
            extension = Path(original_name).suffix.lower()

            # ดึง extension จาก content-type ถ้าไม่มี
            if not extension:
                extension = CONTENT_TYPE_EXTENSION_MAP.get(
                    (upload.content_type or "").lower(), ""
                )
                if extension and not original_name.lower().endswith(extension):
                    original_name = f"{original_name}{extension}"

            content = await upload.read()
            await upload.close()

            # ✅ สร้างชื่อสุ่มสำหรับบันทึกจริง
            random_name = f"img_{uuid.uuid4()}{extension}"

            prepared_files.append(
                {
                    "original_name": original_name,  # ชื่อเดิม (แสดงผล)
                    "random_name": random_name,  # ชื่อสุ่ม (บันทึกจริง)
                    "extension": extension,
                    "content": content,
                }
            )

        # --- 2. นับจำนวนภาพรวม (รวมใน ZIP) ---
        total_image_count = 0
        for item in prepared_files:
            ext = item["extension"]
            if ext == ".zip":
                try:
                    with zipfile.ZipFile(BytesIO(item["content"])) as archive:
                        for member in archive.infolist():
                            if member.is_dir():
                                continue
                            member_ext = Path(member.filename).suffix.lower()
                            if member_ext in ALLOWED_IMAGE_EXTENSIONS:
                                total_image_count += 1
                except zipfile.BadZipFile:
                    pass  # ZIP เสีย → ไม่นับ
            else:
                if ext in ALLOWED_IMAGE_EXTENSIONS:
                    total_image_count += 1

        if total_image_count > MAX_TOTAL_IMAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Total number of images (including inside ZIP files) must not exceed {MAX_TOTAL_IMAGES}.",
            )

        if total_image_count == 0:
            return JSONResponse(
                {"error": "No valid image files provided", "skipped": skipped_entries},
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # --- 3. ประมวลผลไฟล์จริง ---
        for item in prepared_files:
            original_name = item["original_name"]
            random_name = item["random_name"]
            extension = item["extension"]
            content = item["content"]

            if extension == ".zip":
                try:
                    with zipfile.ZipFile(BytesIO(content)) as archive:
                        for member in archive.infolist():
                            if member.is_dir():
                                continue
                            member_original_name = (
                                Path(member.filename).name or member.filename
                            )
                            member_ext = Path(member_original_name).suffix.lower()
                            if (
                                not member_ext
                                or member_ext not in ALLOWED_IMAGE_EXTENSIONS
                            ):
                                skipped_entries.append(
                                    {
                                        "name": member.filename,
                                        "reason": "unsupported_extension",
                                    }
                                )
                                continue
                            # ✅ ใช้ชื่อสุ่มสำหรับไฟล์ใน ZIP ด้วย
                            member_random_name = f"img_{uuid.uuid4()}{member_ext}"
                            files_payload.append(
                                save_bytes_to_uploads(
                                    archive.read(member), member_ext, member_random_name
                                )
                            )
                except zipfile.BadZipFile:
                    skipped_entries.append(
                        {"name": original_name, "reason": "invalid_zip"}
                    )
                continue

            # --- กรณีไฟล์เดี่ยว ---
            if extension not in ALLOWED_IMAGE_EXTENSIONS:
                skipped_entries.append(
                    {"name": original_name, "reason": "unsupported_extension"}
                )
                continue

            # ✅ ใช้ชื่อสุ่มในการบันทึก
            temp_file = BytesIO(content)
            temp_upload = UploadFile(filename=random_name, file=temp_file)
            record = await save_upload_file(temp_upload, original_name=random_name)
            # เก็บ original_name ไว้ใน record สำหรับ response
            record["original_filename"] = original_name
            files_payload.append(record)

        # --- 4-6. วิเคราะห์ config, ประมวลผล, สรุปผล (เหมือนเดิม) ---
        resolved_analysis_types = parse_analysis_types_value(
            analysis_types
        ) or parse_analysis_types_value(api_key_data.get("analysis_types"))
        resolved_analysis_types = [m for m in resolved_analysis_types if m in models]
        if not resolved_analysis_types:
            for rec in files_payload:
                remove_stored_file(rec)
            raise HTTPException(
                status_code=400,
                detail="กรุณาเลือกโมเดลอย่างน้อย 1 โมเดล (เช่น porn, weapon)",
            )

        media_access = {
            str(x).lower() for x in api_key_data.get("media_access", []) if x
        }
        if media_access and "image" not in media_access:
            for rec in files_payload:
                remove_stored_file(rec)
            raise HTTPException(
                status_code=403, detail="API Key ไม่รองรับการวิเคราะห์รูปภาพ"
            )

        resolved_thresholds = parse_thresholds_value(api_key_data.get("thresholds"))
        if not resolved_thresholds:
            resolved_thresholds = parse_thresholds_value(thresholds)
        for model_type in resolved_analysis_types:
            resolved_thresholds.setdefault(model_type, 0.5)

        resolved_output_modes = parse_output_modes_value(
            api_key_data.get("output_modes")
        )
        if not resolved_output_modes:
            resolved_output_modes = parse_output_modes_value(output_modes)
        if not resolved_output_modes:
            resolved_output_modes = ["bbox", "blur"]

        include_bbox = "bbox" in resolved_output_modes
        include_blur = "blur" in resolved_output_modes

        results: List[Dict[str, Any]] = []
        processed_count = 0
        email = api_key_data.get("email")
        api_key = api_key_data.get("api_key")

        async with analysis_concurrency_limiter:
            for record in files_payload:
                file_path = record["file_path"]
                original_name = record["original_filename"]  # ใช้ชื่อเดิมจากผู้ใช้

                if not is_image(file_path):
                    remove_stored_file(record)
                    results.append(
                        {
                            "original_filename": original_name,
                            "status": "error",
                            "error": "Invalid or corrupted image",
                        }
                    )
                    continue

                try:
                    output_image, blurred_output, detections = await run_in_threadpool(
                        process_image_file_for_models,
                        file_path,
                        resolved_analysis_types,
                        resolved_thresholds,
                    )

                    model_summary = defaultdict(int)
                    for d in detections:
                        model_summary[d.get("model_type", "unknown")] += 1

                    processed_filename = blurred_filename = None
                    processed_url = blurred_url = None

                    if include_bbox:
                        processed_filename = f"processed_{uuid.uuid4()}.jpg"
                        processed_path = UPLOAD_FOLDER / processed_filename
                        output_image.save(processed_path)
                        uploaded_files_collection.insert_one(
                            {
                                "filename": processed_filename,
                                "created_at": datetime.utcnow(),
                            }
                        )
                        processed_url = str(
                            request.url_for(
                                "uploaded_file", filename=processed_filename
                            )
                        )

                    if include_blur:
                        blurred_filename = f"blurred_{uuid.uuid4()}.jpg"
                        blurred_path = UPLOAD_FOLDER / blurred_filename
                        blurred_output.save(blurred_path)
                        uploaded_files_collection.insert_one(
                            {
                                "filename": blurred_filename,
                                "created_at": datetime.utcnow(),
                            }
                        )
                        blurred_url = str(
                            request.url_for("uploaded_file", filename=blurred_filename)
                        )

                    status_result = "passed"
                    for d in detections:
                        th = float(resolved_thresholds.get(d.get("model_type"), 0.5))
                        if d.get("confidence", 0) > th:
                            status_result = "failed"
                            break

                    result_entry = {
                        "original_filename": original_name,
                        "status": status_result,
                        "detections": detections,
                        "model_summary": dict(model_summary),
                        "processed_image_url": processed_url,
                        "processed_blurred_image_url": blurred_url,
                    }
                    results.append(result_entry)
                    processed_count += 1

                    log_api_key_usage_event(
                        api_key=api_key,
                        email=email,
                        analysis_types=resolved_analysis_types,
                        thresholds=resolved_thresholds,
                        result={
                            "original_filename": original_name,
                            "stored_filename": record.get("stored_filename"),
                            "status": status_result,
                            "detections": detections,
                            "model_summary": dict(model_summary),
                            "processed_filename": processed_filename,
                            "blurred_filename": blurred_filename,
                            "media_type": "image",
                            "output_modes": resolved_output_modes,
                            "media_access": (
                                list(media_access)
                                if media_access
                                else ["image", "video"]
                            ),
                        },
                    )

                except Exception as e:
                    results.append(
                        {
                            "original_filename": original_name,
                            "status": "error",
                            "error": str(e),
                        }
                    )
                    remove_stored_file(record)
                finally:
                    Path(file_path).unlink(missing_ok=True)

        valid_results = [r for r in results if r["status"] in ("passed", "failed")]
        overall_status = (
            "failed"
            if any(r["status"] == "failed" for r in valid_results)
            else "passed" if valid_results else "error"
        )

        # --- สร้าง summary แบบรวมทุกภาพ ---
        aggregated_summary = defaultdict(int)
        for result in results:
            if result["status"] in ("passed", "failed") and "detections" in result:
                for det in result["detections"]:
                    label = det.get("label")
                    if label:
                        aggregated_summary[label] += 1

        summary_dict = dict(aggregated_summary)
        summary_labels = list(summary_dict.keys())

        response_payload = {
            "status": overall_status,
            "results": results,
            "skipped": skipped_entries,
            "processed_count": processed_count,
            "output_modes": resolved_output_modes,
            "summary": summary_dict,  # ← เพิ่มตรงนี้
            "summary_labels": summary_labels,  # ← เพิ่มตรงนี้
        }

        if len(valid_results) == 1:
            single = valid_results[0]
            response_payload.update(
                {
                    "detections": single["detections"],
                    "model_summary": single.get("model_summary"),
                    "processed_image_url": single.get("processed_image_url"),
                    "processed_blurred_image_url": single.get(
                        "processed_blurred_image_url"
                    ),
                }
            )

        return JSONResponse(response_payload)

    except HTTPException:
        raise
    except Exception as e:
        for rec in files_payload:
            remove_stored_file(rec)
        print(f"[analyze-image] unexpected error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during image analysis"
        )


@app.post("/analyze-video")
async def analyze_video(
    request: Request,
    video: UploadFile = File(...),
    analysis_types_form: Optional[str] = Form(None, alias="analysis_types"),
    thresholds_form: Optional[str] = Form(None, alias="thresholds"),
):
    is_demo = getattr(request.state, "is_demo_mode", False)

    if is_demo:
        api_key_data = {
            "email": "demo@demo.com",
            "api_key": "demo",
            "analysis_types": list(models.keys()),
            "thresholds": {},
            "media_access": ["image", "video"],
            "output_modes": ["bbox", "blur"],
            "plan": "demo",
        }
    else:
        x_api_key = request.headers.get("x-api-key")
        if not x_api_key:
            raise HTTPException(401, "Missing x-api-key")
        api_key_data = api_keys_collection.find_one({"api_key": x_api_key})
        if not api_key_data:
            raise HTTPException(401, "Invalid API Key")

    return await _analyze_video_internal(
        request=request,
        api_key_data=api_key_data,
        video=video,
        analysis_types_form=analysis_types_form,
        thresholds_form=thresholds_form,
    )


async def _analyze_video_internal(
    request: Request,
    api_key_data: Dict[str, Any],
    video: UploadFile,
    analysis_types_form: Optional[str],
    thresholds_form: Optional[str],
):
    original_name = sanitize_filename(video.filename)  # Define original_name
    if not allowed_video(original_name):
        await video.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported video format",
        )

    saved_record = await save_upload_file(
        video, original_name=original_name
    )  # Pass original_name
    temp_path = Path(saved_record["file_path"])

    # ✅ ตรวจสอบความยาววิดีโอ (จำกัด ≤ 60 วินาที)
    cap = cv2.VideoCapture(str(temp_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    if duration > 60:
        # ลบไฟล์ชั่วคราวก่อนตอบกลับ error
        remove_stored_file(saved_record)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded video exceeds the 1-minute limit.",
        )

    # prefer form-specified analysis_types when provided, otherwise use the api key's configured types
    if analysis_types_form:
        analysis_types = parse_analysis_types_value(analysis_types_form)
    else:
        analysis_types = parse_analysis_types_value(api_key_data.get("analysis_types"))

    if not analysis_types:
        remove_stored_file(saved_record)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No analysis_types provided"
        )

    media_access_config = {
        str(item).lower() for item in api_key_data.get("media_access", []) if item
    }
    if media_access_config and "video" not in media_access_config:
        remove_stored_file(saved_record)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key ไม่รองรับการวิเคราะห์วิดีโอ",
        )

    thresholds = parse_thresholds_value(api_key_data.get("thresholds"))
    if not thresholds:
        thresholds = parse_thresholds_value(thresholds_form)
    for model_type in analysis_types:
        thresholds.setdefault(model_type, 0.5)

    output_modes_config = set(
        parse_output_modes_value(api_key_data.get("output_modes"))
    )
    if not output_modes_config:
        output_modes_config = {"bbox", "blur"}

    include_bbox = not output_modes_config or "bbox" in output_modes_config
    include_blur = not output_modes_config or "blur" in output_modes_config

    try:
        async with analysis_concurrency_limiter:
            processed_path, blurred_path, detections, aggregated = (
                await run_in_threadpool(
                    process_video_media,
                    temp_path,
                    analysis_types,
                    thresholds,
                    include_bbox,
                    include_blur,
                )
            )

        processed_filename = processed_path.name if processed_path else None
        blurred_filename = blurred_path.name if blurred_path else None
        processed_url = (
            str(request.url_for("uploaded_file", filename=processed_filename))
            if processed_filename
            else None
        )
        blurred_url = (
            str(request.url_for("uploaded_file", filename=blurred_filename))
            if blurred_filename
            else None
        )

        status_result = "passed"
        for frame_info in detections:
            for detection in frame_info["detections"]:
                threshold = float(thresholds.get(detection.get("model_type"), 0.5))
                if detection.get("confidence", 0) > threshold:
                    status_result = "failed"
                    break
            if status_result == "failed":
                break

        api_key = api_key_data.get("api_key")
        email = api_key_data.get("email")
        summary_dict = dict(aggregated)
        log_api_key_usage_event(
            api_key=api_key,
            email=email,
            analysis_types=analysis_types,
            thresholds=thresholds,
            result={
                "original_filename": original_name,
                "stored_filename": saved_record.get("stored_filename"),
                "status": status_result,
                "detections": summary_dict,
                "processed_filename": processed_filename,
                "blurred_filename": blurred_filename,
                "media_type": "video",
                "output_modes": (
                    list(output_modes_config)
                    if output_modes_config
                    else ["bbox", "blur"]
                ),
                "media_access": (
                    list(media_access_config)
                    if media_access_config
                    else ["image", "video"]
                ),
            },
        )

        api_keys_collection.update_one(
            {"api_key": api_key},
            {"$set": {"last_used_at": datetime.utcnow()}, "$inc": {"usage_count": 1}},
        )

        Path(saved_record["file_path"]).unlink(missing_ok=True)
        uploaded_files_collection.delete_one(
            {"filename": saved_record["stored_filename"]}
        )

        return {
            "status": status_result,
            "original_filename": original_name,
            "processed_video_url": processed_url,
            "processed_blurred_video_url": blurred_url,
            "detections": detections,
            "summary": summary_dict,
            "summary_labels": list(summary_dict.keys()),
            "output_modes": (
                list(output_modes_config) if output_modes_config else ["bbox", "blur"]
            ),
        }
    except Exception as exc:
        # ในกรณี error ระหว่างประมวลผล → ลบไฟล์ชั่วคราว
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        uploaded_files_collection.delete_one(
            {"filename": saved_record["stored_filename"]}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc


@app.post("/request-api-key")
async def request_api_key(
    payload: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    raw_analysis_types = payload.get("analysis_types", [])
    analysis_types = parse_analysis_types_value(raw_analysis_types)
    analysis_types = [atype for atype in analysis_types if atype in models]

    thresholds = parse_thresholds_value(payload.get("thresholds"))
    output_modes = parse_output_modes_value(payload.get("output_modes"))
    plan_raw = (payload.get("plan") or "test").strip().lower()
    plan = "test" if plan_raw in {"test", "free"} else plan_raw

    if plan != "test":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid plan for this endpoint",
        )

    if not analysis_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one analysis type is required",
        )

    email = current_user["email"]

    existing_free_key = api_keys_collection.find_one({"email": email, "plan": "test"})
    if existing_free_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="คุณได้ขอ API Key ทดสอบไปแล้ว"
        )

    api_key = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(days=TEST_PLAN_DURATION_DAYS)
    api_keys_collection.insert_one(
        {
            "email": email,
            "api_key": api_key,
            "analysis_types": analysis_types,
            "thresholds": thresholds,
            "plan": "test",
            "media_access": ["image", "video"],
            "output_modes": output_modes,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "usage_count": 0,
            "last_used_at": None,
        }
    )
    return {
        "apiKey": api_key,
        "expires_at": serialize_datetime(expires_at),
        "plan": "test",
        "media_access": ["image", "video"],
    }


@app.post("/report-issue")
async def report_issue(payload: Dict[str, Any]):
    issue = payload.get("issue")
    category = payload.get("category")

    if not issue or not category:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Issue and category are required",
        )

    subject = f"[รายงานปัญหา] หมวดหมู่: {category}"
    body = f"หมวดหมู่: {category}\nรายละเอียดปัญหา: {issue}"

    try:
        send_email_message(subject, body, ["Phurinsukman3@gmail.com"])
        return {"success": True}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending email: {exc}",
        ) from exc


@app.get("/get-api-keys")
async def get_api_keys(current_user: Dict[str, Any] = Depends(get_current_user)):
    email = current_user.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required"
        )

    try:
        api_keys = list(api_keys_collection.find({"email": email}))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc}",
        ) from exc

    if not api_keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No API keys found for this email",
        )

    formatted_keys = []
    for key in api_keys:
        formatted_key = {
            "api_key": key.get("api_key"),
            "analysis_types": key.get("analysis_types", []),
            "thresholds": key.get("thresholds", {}),
            "plan": key.get("plan"),
            "package": key.get("package"),
            "media_access": key.get("media_access", []),
            "output_modes": key.get("output_modes", []),
            "created_at": serialize_datetime(key.get("created_at")),
            "last_used_at": serialize_datetime(key.get("last_used_at")),
            "usage_count": key.get("usage_count", 0),
            "expires_at": serialize_datetime(key.get("expires_at")),
        }
        formatted_keys.append(formatted_key)

    return {"api_keys": formatted_keys}


@app.get("/get-api-key-history")
async def get_api_key_history(
    request: Request,
    limit_param: Optional[int] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    email = current_user.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required"
        )

    key_cursor = api_keys_collection.find({"email": email}, {"api_key": 1})
    api_key_values = [doc.get("api_key") for doc in key_cursor if doc.get("api_key")]
    if not api_key_values:
        return {"history": []}

    limit = 50
    if limit_param is not None:
        try:
            limit = max(1, min(int(limit_param), 200))
        except (TypeError, ValueError):
            limit = 50

    history_cursor = (
        api_key_usage_collection.find({"api_key": {"$in": api_key_values}})
        .sort("created_at", -1)
        .limit(limit)
    )

    history: List[Dict[str, Any]] = []
    for entry in history_cursor:
        created_at = serialize_datetime(entry.get("created_at"))

        processed_filename = entry.get("processed_filename")
        blurred_filename = entry.get("blurred_filename")
        media_type = str(entry.get("media_type") or "").lower()

        inferred_media_type = media_type if media_type else None
        if not inferred_media_type:
            extension = (
                Path(processed_filename).suffix.lower() if processedFilename else ""
            ) or ""
            if extension in ALLOWED_VIDEO_EXTENSIONS:
                inferred_media_type = "video"
            elif extension in ALLOWED_IMAGE_EXTENSIONS:
                inferred_media_type = "image"
        if not inferred_media_type:
            inferred_media_type = "image"
        media_type = inferred_media_type

        processed_url = (
            str(request.url_for("uploaded_file", filename=processed_filename))
            if processed_filename
            else None
        )
        blurred_url = (
            str(request.url_for("uploaded_file", filename=blurred_filename))
            if blurred_filename
            else None
        )

        detections = entry.get("detections", [])
        detection_summary: List[str] = []
        seen_labels: Set[str] = set()
        if isinstance(detections, dict):
            for label in detections.keys():
                if label is None:
                    continue
                label_str = str(label).strip()
                if not label_str or label_str in seen_labels:
                    continue
                seen_labels.add(label_str)
                detection_summary.append(label_str)
        else:
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                label = detection.get("label")
                if label is None:
                    continue
                label_str = str(label).strip()
                if not label_str or label_str in seen_labels:
                    continue
                seen_labels.add(label_str)
                detection_summary.append(label_str)

        history_entry = {
            "api_key": entry.get("api_key"),
            "original_filename": entry.get("original_filename"),
            "status": entry.get("status"),
            "analysis_types": entry.get("analysis_types", []),
            "thresholds": entry.get("thresholds", {}),
            "detections": detections,
            "detection_summary": detection_summary,
            "media_type": media_type,
            "media_access": entry.get("media_access", []),
            "output_modes": entry.get("output_modes", []),
            "created_at": created_at,
        }

        if media_type == "video":
            history_entry["processed_video_url"] = processed_url
            history_entry["processed_blurred_video_url"] = blurred_url
            history_entry.setdefault("processed_image_url", None)
            history_entry.setdefault("processed_blurred_image_url", None)
        else:
            history_entry["processed_image_url"] = processed_url
            history_entry["processed_blurred_image_url"] = blurred_url

        history.append(history_entry)

    return {"history": history}


@app.get("/get-username")
async def get_username(current_user: Dict[str, Any] = Depends(get_current_user)):
    email = current_user.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Missing email parameter"
        )

    user = users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return {"username": user.get("username")}


@app.get("/manual")
def download_manual() -> FileResponse:
    file_path = BASE_DIR / "manual.pdf"
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Manual not found"
        )
    return FileResponse(file_path)


def generate_qr_code(promptpay_id: str, amount: float = 0) -> str:
    if amount > 0:
        payload = qrcode.generate_payload(promptpay_id, amount)
    else:
        payload = qrcode.generate_payload(promptpay_id)

    img = qrcode.to_image(payload)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


@app.post("/generate_qr")
async def generate_qr(
    payload: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    email = current_user["email"]
    plan_raw = (payload.get("plan") or "premium").strip().lower()
    if plan_raw != "premium":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="รองรับเฉพาะ Premium Plan"
        )

    package_raw = (payload.get("package") or "").strip().lower()
    package_config = PREMIUM_PLAN_PACKAGES.get(package_raw)
    if not package_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="แพ็กเกจไม่ถูกต้อง"
        )

    try:
        duration_months = int(
            payload.get("duration_months")
            or payload.get("duration")
            or payload.get("months")
            or 1
        )
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ระบุจำนวนเดือนไม่ถูกต้อง",
        )
    if duration_months < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="จำนวนเดือนต้องมากกว่าหรือเท่ากับ 1",
        )

    analysis_types = parse_analysis_types_value(payload.get("analysis_types"))
    analysis_types = [atype for atype in analysis_types if atype in models]
    if not analysis_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="กรุณาเลือกโมเดลอย่างน้อย 1 รายการ",
        )

    thresholds = parse_thresholds_value(payload.get("thresholds"))
    output_modes = parse_output_modes_value(payload.get("output_modes"))

    monthly_price = int(package_config["monthly_price"])
    amount = monthly_price * duration_months
    media_access = list(package_config["media_access"])
    promptpay_id = payload.get("promptpay_id", "66882884744")

    existing_unpaid_order = orders_collection.find_one({"email": email, "paid": False})
    if existing_unpaid_order:
        matches_request = (
            existing_unpaid_order.get("plan") == "premium"
            and existing_unpaid_order.get("package") == package_raw
            and int(existing_unpaid_order.get("duration_months", 1)) == duration_months
            and existing_unpaid_order.get("amount") == amount
            and existing_unpaid_order.get("analysis_types", []) == analysis_types
            and existing_unpaid_order.get("thresholds", {}) == thresholds
            and existing_unpaid_order.get("output_modes", []) == output_modes
        )
        if matches_request:
            ref_code = existing_unpaid_order["ref_code"]
            qr_base64 = generate_qr_code(promptpay_id, float(amount))
            return {
                "qr_code_url": qr_base64,
                "ref_code": ref_code,
                "amount": amount,
                "plan": "premium",
                "package": package_raw,
                "duration_months": duration_months,
                "media_access": media_access,
                "message": "ใช้งานคำสั่งซื้อเดิมที่ยังไม่ชำระ",
            }
        orders_collection.delete_one({"_id": existing_unpaid_order["_id"]})

    thai_time = datetime.now(ZoneInfo("Asia/Bangkok"))
    current_time = thai_time.strftime("%d/%m/%Y %H:%M:%S")
    timestamp = thai_time.strftime("%Y%m%d%H%M%S")
    random_str = secrets.token_hex(4).upper()
    ref_code = f"{current_time} {timestamp}{random_str}"

    orders_collection.insert_one(
        {
            "ref_code": ref_code,
            "email": email,
            "amount": amount,
            "plan": "premium",
            "package": package_raw,
            "duration_months": duration_months,
            "analysis_types": analysis_types,
            "thresholds": thresholds,
            "output_modes": output_modes,
            "media_access": media_access,
            "paid": False,
            "created_at": current_time,
            "created_time": datetime.now(timezone.utc),
        }
    )

    qr_base64 = generate_qr_code(promptpay_id, float(amount))
    return {
        "qr_code_url": qr_base64,
        "ref_code": ref_code,
        "amount": amount,
        "plan": "premium",
        "package": package_raw,
        "duration_months": duration_months,
        "media_access": media_access,
    }


@app.post("/cancel-order")
async def cancel_order(
    payload: Optional[Dict[str, Any]] = Body(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    payload = payload or {}
    ref_code = payload.get("ref_code")

    query: Dict[str, Any] = {"email": current_user["email"], "paid": False}
    if ref_code:
        query["ref_code"] = ref_code

    order = orders_collection.find_one(query, sort=[("created_time", -1)])
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ไม่พบคำสั่งซื้อที่สามารถยกเลิกได้",
        )

    orders_collection.delete_one({"_id": order["_id"]})
    return {
        "success": True,
        "message": "คำสั่งซื้อถูกยกเลิกแล้ว",
        "ref_code": order.get("ref_code"),
    }


def check_qrcode(image_path: str) -> bool:
    image = cv2.imread(image_path)
    if image is None:
        print(f"[DEBUG] โหลดภาพไม่ได้: {image_path}")
        return False
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(image)
    print(f"[DEBUG] QR points: {points is not None}")
    print(f"[DEBUG] QR data: {repr(data)}")
    return points is not None and bool(data)


@app.post("/upload-receipt")
async def upload_receipt(
    receipt: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    save_path: Optional[Path] = None
    try:
        filename = sanitize_filename(receipt.filename)
        save_path = UPLOAD_FOLDER / filename
        content = await receipt.read()
        with open(save_path, "wb") as fh:
            fh.write(content)
        await receipt.close()

        if not is_image(str(save_path)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="ไฟล์ไม่ใช่รูปภาพ"
            )

        if not check_qrcode(str(save_path)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="รูปเเบบใบเสร็จไม่ถูกต้อง"
            )

        try:
            ocr_engine = AdvancedSlipOCR()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ระบบ OCR ล้มเหลว",
            ) from exc

        try:
            image = Image.open(save_path).convert("RGB")
            ocr_data = ocr_engine.extract_info(image)
            print("=== OCR DATA ===")
            print(ocr_data)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ไม่สามารถประมวลผลรูปภาพได้",
            ) from exc

        required_fields = [
            "full_text",
            "date",
            "time",
            "amount",
            "full_name",
            "time_receipts",
        ]
        for field in required_fields:
            if not ocr_data.get(field):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"ข้อมูล {field} ขาดหายไปหรือเป็นค่าว่าง",
                )

        text = ocr_data["full_text"]
        date_text = ocr_data["date"]
        time_ocr = ocr_data["time"]
        amount = ocr_data["amount"]
        full_name = ocr_data["full_name"]
        time_receipts = ocr_data["time_receipts"]

        matched_order = orders_collection.find_one(
            {"email": current_user["email"], "paid": False},
            sort=[("created_time", -1)],
        )

        if not matched_order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="ไม่พบคำสั่งซื้อที่ยังไม่ชำระเงินสำหรับคุณ",
            )

        allowed_names = ["ภูรินทร์สุขมั่น", "ภูรินทร์", "สุขมั่น", "ภูรินทร์ สุขมั่น"]
        full_name_clean = full_name.strip().replace(" ", "").lower()
        allowed_names_clean = [name.replace(" ", "").lower() for name in allowed_names]
        if not any(name in full_name_clean for name in allowed_names_clean):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="ชื่อผู้รับเงินไม่ถูกต้อง"
            )

        try:
            created_datetime = datetime.strptime(
                matched_order["created_at"], "%d/%m/%Y %H:%M:%S"
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ข้อมูลวันที่ในฐานข้อมูลผิดพลาด",
            ) from exc

        if date_text:
            try:
                # แปลงปี พ.ศ. เป็น ค.ศ. ก่อนเปรียบเทียบ
                parts = date_text.split("/")
                day, month, year_str = parts[0], parts[1], parts[2]
                year_int = int(year_str)
                if year_int < 100:  # เช่น 68
                    year_ad = year_int + 1957  # 68 + 1957 = 2025
                elif year_int >= 2500:  # เช่น 2568
                    year_ad = year_int - 543  # 2568 - 543 = 2025
                else:  # ถ้าเป็นปี ค.ศ. อยู่แล้ว เช่น 2025
                    year_ad = year_int

                date_from_ocr = datetime(int(year_ad), int(month), int(day)).date()

                if date_from_ocr != created_datetime.date():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="วันที่ในสลิปไม่ตรงกับวันที่สร้างออร์เดอร์",
                    )

            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="วันที่ในสลิปไม่ตรงกับวันที่สร้างออร์เดอร์",
                ) from exc

        if time_receipts:
            try:
                time_from_ocr = datetime.strptime(time_receipts, "%H:%M")
                time_from_ocr_full = datetime.combine(
                    created_datetime.date(), time_from_ocr.time()
                )
                time_diff = abs((created_datetime - time_from_ocr_full).total_seconds())
                if time_diff > 300:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="เวลาในสลิปห่างกันเกิน 5 นาที",
                    )
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="เวลาในสลิปห่างกันเกิน 5 นาที",
                ) from exc

        if amount:
            try:
                amount_clean = float(amount.replace(",", ""))
                if float(matched_order.get("amount", 0)) != amount_clean:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST, detail="ยอดเงินไม่ตรงกัน"
                    )
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="ยอดเงินไม่สามารถแปลงได้",
                ) from exc

        orders_collection.update_one(
            {"_id": matched_order["_id"]},
            {
                "$set": {
                    "paid": True,
                    "paid_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                }
            },
        )

        api_key = str(uuid.uuid4())
        plan = matched_order.get("plan", "premium")
        package = matched_order.get("package")
        duration_months_raw = matched_order.get(
            "duration_months", matched_order.get("duration", 1)
        )
        try:
            duration_months = max(int(duration_months_raw), 1)
        except (TypeError, ValueError):
            duration_months = 1

        if plan in {"paid", "monthly"}:
            plan = "premium"

        raw_thresholds = matched_order.get("thresholds", {})
        thresholds_payload = {}
        if isinstance(raw_thresholds, dict):
            for key, value in raw_thresholds.items():
                try:
                    thresholds_payload[key] = float(value)
                except (TypeError, ValueError):
                    continue

        media_access = matched_order.get("media_access") or []
        if not media_access and package in PREMIUM_PLAN_PACKAGES:
            media_access = list(PREMIUM_PLAN_PACKAGES[package]["media_access"])
        if not media_access:
            media_access = ["image", "video"]
        output_modes = matched_order.get("output_modes") or []
        if not output_modes:
            output_modes = ["bbox", "blur"]

        insert_data: Dict[str, Any] = {
            "email": matched_order.get("email", ""),
            "api_key": api_key,
            "analysis_types": matched_order.get("analysis_types", []),
            "thresholds": thresholds_payload,
            "plan": plan,
            "package": package,
            "media_access": media_access,
            "output_modes": output_modes,
            "created_at": datetime.utcnow(),
            "usage_count": 0,
            "last_used_at": None,
        }

        if plan == "premium":
            insert_data["expires_at"] = datetime.now(timezone.utc) + relativedelta(
                months=+duration_months
            )

        api_keys_collection.insert_one(insert_data)
        orders_collection.delete_one({"_id": matched_order["_id"]})

        return {
            "success": True,
            "message": "อัปโหลดสำเร็จ",
            "api_key": api_key,
            "ocr_data": {
                "date": date_text,
                "time": time_ocr,
                "amount": amount,
                "fullname": full_name,
                "full_text": text,
            },
        }
    finally:
        if save_path and save_path.exists():
            try:
                save_path.unlink()
            except Exception:
                pass


@app.get("/auth/google")
async def auth_google() -> RedirectResponse:
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={GOOGLE_REDIRECT_URI}&"
        f"response_type=code&"
        f"scope=openid email profile"
    )
    return RedirectResponse(google_auth_url)


@app.get("/auth/google/callback")
async def google_callback(request: Request, code: Optional[str] = None):
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code not found",
        )

    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    token_response = requests.post(token_url, data=token_data)
    token_json = token_response.json()

    access_token = token_json.get("access_token")
    user_info_url = "https://www.googleapis.com/oauth2/v1/userinfo"
    user_info_response = requests.get(
        user_info_url, headers={"Authorization": f"Bearer {access_token}"}
    )
    user_info = user_info_response.json()

    email = user_info.get("email")
    user = users_collection.find_one({"email": email})
    if not user:
        users_collection.insert_one(
            {
                "email": email,
                "username": user_info.get("name"),
                "password": None,
            }
        )

    token = generate_token(email)
    base_url_from_request = str(request.base_url).rstrip("/")
    parsed_env_url = urlparse(API_BASE_URL) if API_BASE_URL else None
    env_host = parsed_env_url.hostname if parsed_env_url else None
    request_host = (
        request.base_url.hostname if hasattr(request.base_url, "hostname") else None
    )

    use_env_base = False
    if API_BASE_URL:
        if env_host and request_host:
            use_env_base = env_host == request_host
        elif not request_host:
            use_env_base = True

    base_url = (
        API_BASE_URL.rstrip("/")
        if use_env_base and API_BASE_URL
        else base_url_from_request
    )
    redirect_url = (
        f"{base_url}/apikey/view-api-keys.html?token={token}"
        if base_url
        else f"/?token={token}"
    )
    return RedirectResponse(redirect_url)


@app.post("/reset-request")
async def reset_request(
    payload: Dict[str, Any],
):
    email = payload.get("email")
    if not users_collection.find_one({"email": email}):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ไม่พบอีเมลนี้")

    otp = str(random.randint(100000, 999999))
    expiration = datetime.utcnow() + timedelta(minutes=5)

    otp_collection.update_one(
        {"email": email},
        {"$set": {"otp": otp, "otp_expiration": expiration, "used": False}},
        upsert=True,
    )

    send_email_message("OTP สำหรับรีเซ็ตรหัสผ่าน", f"รหัส OTP ของคุณคือ: {otp}", [email])
    return {"message": "ส่ง OTP แล้ว"}


@app.post("/verify-otp")
async def verify_otp(payload: Dict[str, Any]):
    email = payload.get("email")
    otp = payload.get("otp")

    record = otp_collection.find_one({"email": email, "otp": otp, "used": False})
    if not record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="OTP ไม่ถูกต้อง"
        )

    if record["otp_expiration"] < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="OTP หมดอายุแล้ว"
        )

    # Mark OTP as used to prevent reuse
    otp_collection.update_one({"email": email, "otp": otp}, {"$set": {"used": True}})

    return {"message": "OTP ถูกต้อง"}


@app.post("/reset-password")
async def reset_password(payload: Dict[str, Any]):
    email = payload.get("email")
    otp = payload.get("otp")
    password = payload.get("password")
    confirm_password = payload.get("confirm_password")

    print(f"[DEBUG] Reset password request for email: {email}")

    if password != confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="รหัสผ่านไม่ตรงกัน"
        )

    # Don't check "used": False because it was already marked as used in verify-otp
    record = otp_collection.find_one({"email": email, "otp": otp})
    if not record or record["otp_expiration"] < datetime.utcnow():
        print(f"[DEBUG] OTP record not found or expired for {email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="OTP ไม่ถูกต้องหรือหมดอายุ"
        )

    print(f"[DEBUG] OTP verified for {email}")

    try:
        hashed_password = generate_password_hash(
            password, method="pbkdf2:sha256", salt_length=8
        )
        print(f"[DEBUG] Generated password hash: {str(hashed_password)[:50]}...")

        # Check if user exists
        user_before = users_collection.find_one({"email": email})
        if not user_before:
            print(f"[DEBUG] User not found: {email}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="ไม่พบผู้ใช้นี้ในระบบ"
            )

        old_pass = user_before.get("password", "")
        print(
            f"[DEBUG] User found, old password hash: {str(old_pass)[:50] if old_pass else 'None'}..."
        )

        result = users_collection.update_one(
            {"email": email}, {"$set": {"password": hashed_password}}
        )
        print(
            f"[DEBUG] Update result - matched: {result.matched_count}, modified: {result.modified_count}"
        )

        # Verify the update
        user_after = users_collection.find_one({"email": email})
        new_pass = user_after.get("password", "") if user_after else ""
        print(
            f"[DEBUG] New password hash: {str(new_pass)[:50] if new_pass else 'None'}..."
        )

        # Clean up OTP after successful password reset
        otp_collection.delete_one({"email": email, "otp": otp})
        print(f"[DEBUG] OTP deleted for {email}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to update password: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"เกิดข้อผิดพลาดในการรีเซ็ตรหัสผ่าน: {str(e)}",
        )

    return {"message": "รีเซ็ตรหัสผ่านเรียบร้อยแล้ว"}


@app.get("/{filename:path}")
def serve_other_files(filename: str) -> FileResponse:
    file_path = BASE_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
        )
    return FileResponse(file_path)


def cleanup_expired_files() -> None:
    try:
        current_files = set(os.listdir(UPLOAD_FOLDER))
        active_files = set(doc["filename"] for doc in uploaded_files_collection.find())
        expired_files = current_files - active_files
        for fname in expired_files:
            try:
                (UPLOAD_FOLDER / fname).unlink()
                print(f"Deleted expired file: {fname}")
            except PermissionError:
                print(f"Skip deleting (in use): {fname}")
                continue
            except Exception as exc:
                print(f"Error deleting {fname}: {exc}")
    except Exception as exc:
        print(f"Cleanup system error: {exc}")


def start_cleanup_scheduler() -> None:
    import threading
    import time

    def run():
        while True:
            cleanup_expired_files()
            time.sleep(300)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


start_cleanup_scheduler()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)
