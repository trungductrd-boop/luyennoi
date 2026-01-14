import os
import uuid
import json
import re
import subprocess
from typing import Optional
from datetime import datetime
import time
import librosa
import numpy as np
import logging

# --- Configuration ---
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
SAMPLES_DIR = "data/samples"
USERS_DIR = "data/users"
STATIC_DIR = "data"
VOCAB_STORE_FILE = "data/vocab_store.json"
TMP_DIR = "data/tmp"
FFMPEG_BIN = "ffmpeg"
ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".ogg"}

# ensure directories exist
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(VOCAB_STORE_FILE) or ".", exist_ok=True)

# create empty store if missing
if not os.path.exists(VOCAB_STORE_FILE):
    with open(VOCAB_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump({"lessons": {}, "samples": {}}, f, ensure_ascii=False, indent=2)

# Built-in lessons & vocab (fallback/demo)
LESSONS = [
    {"id": "1", "title": "Chào hỏi cơ bản", "description": "Học cách chào hỏi", "progress": 80},
    {"id": "2", "title": "Gia đình", "description": "Từ vựng về gia đình", "progress": 60},
    {"id": "3", "title": "Thức ăn & Đồ uống", "description": "Từ vựng đồ ăn", "progress": 45},
    {"id": "4", "title": "Số đếm", "description": "Từ vựng về số và cách đếm", "progress": 0},
    {"id": "5", "title": "Màu sắc", "description": "Từ vựng các màu cơ bản", "progress": 0},
    {"id": "6", "title": "Động vật", "description": "Từ vựng về các động vật thường gặp", "progress": 0},
]

VOCAB = {
    "1": [
        {"id": "1_1", "word": "Xin chào", "meaning": "Hello", "example": "Xin chào bạn!", "audio_filename": "xin_chao.mp3"},
        {"id": "1_2", "word": "Tạm biệt", "meaning": "Goodbye", "example": "Tạm biệt nhé!", "audio_filename": "tam_biet.mp3"},
    ],
    "2": [
        {"id": "2_1", "word": "Mẹ", "meaning": "Mother", "example": "Mẹ của tôi...", "audio_filename": "me.mp3"},
        {"id": "2_2", "word": "Cha", "meaning": "Father", "example": "Cha làm nghề...", "audio_filename": "cha.mp3"},
        {"id": "2_3", "word": "Anh trai", "meaning": "Brother", "example": "Anh trai tôi...", "audio_filename": "anh_trai.mp3"},
    ],
    "3": [
        {"id": "3_1", "word": "Phở", "meaning": "Pho (noodle soup)", "example": "Tôi thích phở.", "audio_filename": "pho_dc1c9090-192c-4529-a74b-2325b488b86e.mp3"},
    ],
    "5": [
        {"id": "5_1", "word": "Đỏ", "meaning": "Red", "example": "Quả táo màu đỏ.", "audio_filename": "do_1c414482-56cd-43e5-a6cb-998a961d07a2.mp3"},
        {"id": "5_2", "word": "Vàng", "meaning": "Yellow", "example": "Hoa hướng dương màu vàng.", "audio_filename": "vang_1f4a223d-3ca2-4134-8894-b6a2070b6a1d.mp3"},
        {"id": "5_3", "word": "Xanh dương", "meaning": "Blue", "example": "Bầu trời xanh dương.", "audio_filename": "xanh_duong_a9788857-607d-430b-8e85-bd81fb7be85c.mp3"},
        {"id": "5_4", "word": "Xanh lá", "meaning": "Green", "example": "Lá cây màu xanh lá.", "audio_filename": "xanh_la_d5211260-62b1-4786-8dea-47057a94e172.mp3"},
    ],
    "6": [
        {"id": "6_1", "word": "Chó", "meaning": "Dog", "example": "Con chó của tôi.", "audio_filename": "cho_a7869b6f-d0b5-455b-ae21-43c628acc6cc.mp3"},
        {"id": "6_2", "word": "Mèo", "meaning": "Cat", "example": "Con mèo thích ngủ.", "audio_filename": "meo_dd324ca4-d60b-4ff0-9eba-4d7da40dc485.mp3"},
        {"id": "6_3", "word": "Gà", "meaning": "Chicken", "example": "Gà gáy vào buổi sáng.", "audio_filename": "ga_74dd0293-bbd8-4bc3-bdaf-ebdd76442d06.mp3"},
        {"id": "6_4", "word": "Heo", "meaning": "Pig", "example": "Heo sống trong chuồng.", "audio_filename": "heo_a731d482-0034-47e6-bab1-2616151e1c3b.mp3"},
        {"id": "6_5", "word": "Chim", "meaning": "Bird", "example": "Chim hót trên cây.", "audio_filename": "chim_a40a2c8e-6ea4-4583-8a0f-8e8bb0f831c4.mp3"},
    ],
}

# persisted store (file-backed)
PERSISTED_STORE = {"lessons": {}, "samples": {}}

# Internal flag: track whether the persisted store has been loaded at least once
_PERSISTED_STORE_LOADED = False

# Setup logging with better format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_persisted_store():
    global PERSISTED_STORE
    global _PERSISTED_STORE_LOADED
    try:
        with open(VOCAB_STORE_FILE, "r", encoding="utf-8") as f:
            PERSISTED_STORE = json.load(f)
        # Log at INFO only the first time, subsequent loads are routine and logged at DEBUG
        if not _PERSISTED_STORE_LOADED:
            logger.info(f"Loaded persisted store: {len(PERSISTED_STORE.get('samples', {}))} samples")
            _PERSISTED_STORE_LOADED = True
        else:
            logger.debug(f"Loaded persisted store (again): {len(PERSISTED_STORE.get('samples', {}))} samples")
        
        # Load progress back into LESSONS
        progress_data = PERSISTED_STORE.get("progress", {})
        for lesson in LESSONS:
            if lesson["id"] in progress_data:
                lesson["progress"] = progress_data[lesson["id"]]
        
        # Ensure all required keys exist
        PERSISTED_STORE.setdefault("lessons", {})
        PERSISTED_STORE.setdefault("samples", {})
        PERSISTED_STORE.setdefault("progress", {})
        
    except FileNotFoundError:
        PERSISTED_STORE = {"lessons": {}, "samples": {}, "progress": {}}
        logger.warning("vocab_store.json not found, created new store")
    except json.JSONDecodeError as e:
        PERSISTED_STORE = {"lessons": {}, "samples": {}, "progress": {}}
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        PERSISTED_STORE = {"lessons": {}, "samples": {}, "progress": {}}
        logger.error(f"Error loading store: {e}")

def save_persisted_store():
    try:
        with open(VOCAB_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(PERSISTED_STORE, f, ensure_ascii=False, indent=2)
        logger.info("Persisted store saved successfully")
    except Exception as e:
        logger.error(f"Error saving store: {e}")
        raise

def merged_vocab_for_lesson(lesson_id: str):
    result = []
    if lesson_id in VOCAB:
        for v in VOCAB[lesson_id]:
            fn = v.get("audio_filename")
            v2 = {**v}
            v2["audio_url"] = f"/static/samples/{fn}" if fn else None
            result.append(v2)
    lessons_store = PERSISTED_STORE.get("lessons", {})
    if lesson_id in lessons_store:
        for v in lessons_store[lesson_id]:
            fn = v.get("audio_filename")
            v2 = {**v}
            v2["audio_url"] = f"/static/samples/{fn}" if fn else None
            result.append(v2)
    return result

def convert_to_wav16_mono(src_path: str, dst_path: str) -> None:
    try:
        cmd = [FFMPEG_BIN, "-y", "-i", src_path, "-ac", "1", "-ar", "16000", "-vn", dst_path]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Converted {src_path} to {dst_path}")
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode() if e and getattr(e, 'stderr', None) else str(e)
        logger.error(f"FFmpeg conversion failed: {stderr_text}")
        # write stderr to a temp log for easier debugging from API
        try:
            os.makedirs(TMP_DIR, exist_ok=True)
            stamp = int(time.time() * 1000)
            err_path = os.path.join(TMP_DIR, f"ffmpeg_convert_error_{stamp}.log")
            with open(err_path, "w", encoding="utf-8") as ef:
                ef.write(stderr_text)
            logger.info(f"Wrote ffmpeg stderr to {err_path}")
        except Exception:
            pass
        raise RuntimeError(f"Audio conversion failed: {stderr_text}")
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install ffmpeg and add to PATH")
        raise RuntimeError("FFmpeg not installed")

def extract_features(path: str):
    try:
        # Suppress librosa warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path, sr=16000)
        
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        
        # More robust pitch extraction
        pitch, magnitude = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitch[magnitude > np.median(magnitude)]
        positive_pitch = pitch_values[pitch_values > 0]
        pitch_mean = float(np.mean(positive_pitch)) if positive_pitch.size > 0 else 0.0
        
        # Safer tempo extraction
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
        except Exception:
            tempo = 120.0  # Default BPM if extraction fails
        
        logger.info(f"Extracted features from {path}")
        return {"mfcc": mfcc.tolist(), "pitch": pitch_mean, "tempo": tempo}
    except Exception as e:
        logger.error(f"Feature extraction failed for {path}: {e}")
        raise

def compare_features_dicts(f1: dict, f2: dict):
    a = np.array(f1["mfcc"])
    b = np.array(f2["mfcc"])
    mfcc_dist = float(np.linalg.norm(a - b))
    pitch_diff = float(abs(f1["pitch"] - f2["pitch"]))
    tempo_diff = float(abs(f1["tempo"] - f2["tempo"]))
    return mfcc_dist, pitch_diff, tempo_diff

def rescan_samples():
    try:
        load_persisted_store()
        samples_meta = PERSISTED_STORE.setdefault("samples", {})
        lessons_store = PERSISTED_STORE.setdefault("lessons", {})

        existing_filenames = {m["filename"] for m in samples_meta.values()}
        
        if not os.path.exists(SAMPLES_DIR):
            logger.warning(f"Samples directory not found: {SAMPLES_DIR}")
            return {
                "total_files_on_disk": 0,
                "registered_samples_total": len(samples_meta),
                "new_samples_found": 0,
                "new_samples": [],
                "new_vocabs_added": 0,
                "new_vocabs": [],
            }
        
        files = sorted([f for f in os.listdir(SAMPLES_DIR) if os.path.isfile(os.path.join(SAMPLES_DIR, f))])
        new_samples = []
        new_vocabs = []

        for fn in files:
            _, ext = os.path.splitext(fn)
            if ext.lower() not in ALLOWED_EXTS:
                continue
            if fn in existing_filenames:
                continue
            sample_id = str(uuid.uuid4())[:8]
            samples_meta[sample_id] = {"filename": fn, "lesson_id": None, "vocab_id": None}
            new_samples.append({"sample_id": sample_id, "filename": fn})

            m = re.match(r'^(\d+)[-_](.+)$', fn)
            if m:
                lesson_id = m.group(1)
                remainder = os.path.splitext(m.group(2))[0]
                word = re.sub(r'[_\-]+', ' ', remainder).strip().capitalize()
                lesson_ids = {l["id"] for l in LESSONS}
                if lesson_id in lesson_ids:
                    lessons_store.setdefault(lesson_id, lessons_store.get(lesson_id, []))
                    exists = any(v.get("audio_filename") == fn for v in lessons_store[lesson_id])
                    if not exists:
                        vid = f"{lesson_id}_{len(lessons_store[lesson_id]) + 1}"
                        vocab_obj = {"id": vid, "word": word or vid, "meaning": "", "example": "", "audio_filename": fn}
                        lessons_store[lesson_id].append(vocab_obj)
                        samples_meta[sample_id]["lesson_id"] = lesson_id
                        samples_meta[sample_id]["vocab_id"] = vid
                        new_vocabs.append({"lesson_id": lesson_id, "vocab_id": vid, "word": word, "audio_filename": fn})

        save_persisted_store()
        report = {
            "total_files_on_disk": len(files),
            "registered_samples_total": len(samples_meta),
            "new_samples_found": len(new_samples),
            "new_samples": new_samples,
            "new_vocabs_added": len(new_vocabs),
            "new_vocabs": new_vocabs,
        }
        # Only log at INFO when there are new items; otherwise use DEBUG to reduce noise
        if len(new_samples) or len(new_vocabs):
            logger.info(f"Rescan completed: {len(new_samples)} new samples, {len(new_vocabs)} new vocabs")
        else:
            logger.debug(f"Rescan completed: {len(new_samples)} new samples, {len(new_vocabs)} new vocabs")
        return report
    except Exception as e:
        logger.error(f"Rescan failed: {e}")
        raise

def _get_timestamp() -> str:
    """Get current timestamp as ISO string"""
    return datetime.now().isoformat()

def _timestamp_to_str(timestamp: float) -> str:
    """Convert Unix timestamp to ISO string"""
    return datetime.fromtimestamp(timestamp).isoformat()

# Multimodal helpers are provided in a separate module helpers_multimodal.py.
# Attempt relative import first, fall back to top-level import, else set to None.
try:
    from . import helpers_multimodal as helpers_multimodal
except Exception:
    try:
        import helpers_multimodal as helpers_multimodal
    except Exception:
        helpers_multimodal = None
