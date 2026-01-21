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
import unicodedata
 
# Optional memory logging helper
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

def log_mem(tag: str):
    try:
        if psutil:
            mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            logger.info(f"[mem] {tag} RAM={mem:.1f}MB")
        else:
            logger.debug(f"[mem] {tag} psutil not available")
    except Exception:
        pass

# Optional: portalocker for cross-process file locking
try:
    import portalocker
except Exception:
    portalocker = None

# Try to import optional audio_features helper module (may not exist in minimal installs)
try:
    from . import audio_features
except Exception:
    try:
        import audio_features
    except Exception:
        audio_features = None

# --- Configuration ---
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
SAMPLES_DIR = "data/samples"
USERS_DIR = "data/users"
STATIC_DIR = "data"
# Allow overriding persisted store path via env var to keep it outside watched dirs
PERSIST_PATH = os.environ.get("PERSIST_PATH", "data/vocab_store.json")
VOCAB_STORE_FILE = PERSIST_PATH
TMP_DIR = "data/tmp"
FFMPEG_BIN = "ffmpeg"
ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".ogg"}

# Fast mode: enable aggressive speedups and disable non-essential features.
FAST_MODE = os.environ.get("FAST_MODE", "0") in ("1", "true", "True")

# ensure directories exist
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(VOCAB_STORE_FILE) or ".", exist_ok=True)

# Lock path used for coordinating readers/writers and leader election
LOCK_PATH = VOCAB_STORE_FILE + ".lock"

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

# Setup logging with better format. Lower verbosity in FAST_MODE to reduce overhead.
logging.basicConfig(
    level=(logging.WARNING if FAST_MODE else logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_persisted_store():
    global PERSISTED_STORE
    global _PERSISTED_STORE_LOADED
    try:
        # If portalocker is available, acquire a shared lock while reading to avoid races
        if portalocker:
            # ensure lock file exists
            open(LOCK_PATH, "a").close()
            with open(LOCK_PATH, "r+") as lockf:
                portalocker.lock(lockf, portalocker.LockFlags.SHARED)
                try:
                    with open(VOCAB_STORE_FILE, "r", encoding="utf-8") as f:
                        PERSISTED_STORE = json.load(f)
                finally:
                    try:
                        portalocker.unlock(lockf)
                    except Exception:
                        pass
        else:
            with open(VOCAB_STORE_FILE, "r", encoding="utf-8") as f:
                PERSISTED_STORE = json.load(f)
        # Avoid noisy INFO logs from multiple worker processes; use DEBUG for loads
        logger.debug(f"Loaded persisted store: {len(PERSISTED_STORE.get('samples', {}))} samples (pid={os.getpid()})")
        _PERSISTED_STORE_LOADED = True
        
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
        # Ensure destination dir exists
        os.makedirs(os.path.dirname(VOCAB_STORE_FILE) or ".", exist_ok=True)
        # If portalocker available, acquire exclusive lock while writing
        # Write atomically to avoid watchers seeing partial files and to reduce reload triggers.
        tmp_path = VOCAB_STORE_FILE + ".tmp"
        # Ensure destination dir exists
        os.makedirs(os.path.dirname(VOCAB_STORE_FILE) or ".", exist_ok=True)
        if portalocker:
            # ensure lock file exists
            open(LOCK_PATH, "a").close()
            with open(LOCK_PATH, "r+") as lockf:
                portalocker.lock(lockf, portalocker.LockFlags.EXCLUSIVE)
                try:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(PERSISTED_STORE, f, ensure_ascii=False, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    # atomic replace
                    os.replace(tmp_path, VOCAB_STORE_FILE)
                finally:
                    try:
                        portalocker.unlock(lockf)
                    except Exception:
                        pass
        else:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(PERSISTED_STORE, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, VOCAB_STORE_FILE)
        logger.info("Persisted store saved successfully (path=%s)", VOCAB_STORE_FILE)
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


def sanitize_filename(name: Optional[str]) -> str:
    """Sanitize an incoming filename for safe upload/storage.

    - Normalize Unicode (NFKD) and strip diacritics
    - Replace whitespace with underscores
    - Allow only a conservative subset of chars (alnum, dot, dash, underscore)
    - Collapse repeated underscores and trim length to 255
    Returns a safe ASCII filename; if input is falsy returns an empty string.
    """
    if not name:
        return ""
    try:
        # basename only
        base = os.path.basename(name)
        # Normalize and remove diacritics
        nk = unicodedata.normalize('NFKD', base)
        no_diac = ''.join(c for c in nk if not unicodedata.combining(c))
        # Replace spaces with underscore
        replaced = re.sub(r"\s+", "_", no_diac)
        # Keep only safe characters
        safe = re.sub(r"[^A-Za-z0-9._\-]", "", replaced)
        # Collapse repeated underscores/dots/dashes
        safe = re.sub(r"_+", "_", safe)
        safe = safe.strip('._-')
        if not safe:
            return ""
        # Limit length
        if len(safe) > 255:
            safe = safe[:255]
        return safe
    except Exception:
        return os.path.basename(name) or ""

def convert_to_wav16_mono(src_path: str, dst_path: str) -> None:
    # Use a temp filename that ends with .wav so ffmpeg can infer WAV muxer
    tmp = dst_path + ".tmp.wav"
    try:
        os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
        cmd = [FFMPEG_BIN, "-y", "-i", src_path, "-ac", "1", "-ar", "16000", "-vn", tmp]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            stderr_text = proc.stderr or proc.stdout or f"returncode={proc.returncode}"
            logger.error("FFmpeg conversion_failed: %s", stderr_text)
            try:
                os.makedirs(TMP_DIR, exist_ok=True)
                stamp = int(time.time() * 1000)
                err_path = os.path.join(TMP_DIR, f"ffmpeg_convert_error_{stamp}.log")
                with open(err_path, "w", encoding="utf-8") as ef:
                    ef.write(stderr_text)
                logger.info("Wrote ffmpeg stderr to %s", err_path)
            except Exception:
                pass
            # remove partial tmp file if present
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)

        # verify tmp exists and is non-empty
        if not os.path.exists(tmp):
            raise RuntimeError(f"ffmpeg reported success but output missing: {tmp}")
        try:
            st = os.stat(tmp)
            if st.st_size == 0:
                raise RuntimeError(f"ffmpeg produced empty output: {tmp}")
        except Exception:
            # propagate as runtime error
            raise

        # atomic replace to avoid half-written outputs being observed by other threads
        try:
            os.replace(tmp, dst_path)
        except Exception:
            # fallback to rename
            os.rename(tmp, dst_path)
        try:
            st2 = os.stat(dst_path)
            logger.info("Converted %s to %s (size=%d)", src_path, dst_path, st2.st_size)
        except Exception:
            logger.info("Converted %s to %s", src_path, dst_path)
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install ffmpeg and add to PATH")
        raise

def extract_features(path: str):
    try:
        # Suppress librosa warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Choose MFCC config depending on FAST_MODE
            n_mfcc = 13 if not FAST_MODE else 8

            # Always limit duration to avoid long files hanging the extractor
            load_kwargs = {"sr": 16000, "mono": True, "duration": 3.0}

            logger.info("Loading audio for feature extraction: %s", path)
            y, sr = librosa.load(path, **load_kwargs)
            logger.info("librosa.load done: sr=%s, samples=%d", sr, len(y))

            # Trim leading/trailing silence to focus on speech content
            try:
                y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                logger.info("Trimmed silence: original_samples=%d trimmed_samples=%d", len(y), len(y_trimmed))
                y = y_trimmed
            except Exception:
                logger.warning("librosa.effects.trim failed, continuing with original signal")

            # MFCC: prefer audio_features when available and not in FAST_MODE
            mfcc_vals = None
            if audio_features is not None and not FAST_MODE:
                try:
                    mfcc_vals = audio_features.extract_mfcc_mean(path, n_mfcc=n_mfcc, sr=sr, include_deltas=True)
                except Exception:
                    mfcc_vals = None

            if mfcc_vals is None:
                mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                # Limit number of frames to avoid very long matrices (keep ~3s * hop rate)
                max_frames = 300
                if mf.shape[1] > max_frames:
                    mf = mf[:, :max_frames]
                    logger.info("MFCC frames truncated to %d", max_frames)
                logger.info("mfcc computed, shape=%s", mf.shape)
                # Reduce to mean MFCC vector (time-averaged)
                mfcc_vals = np.mean(mf, axis=1)

            # Pitch estimation: attempt lightweight methods only
            pitch_mean = 0.0
            try:
                f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
                f0_pos = f0[np.isfinite(f0) & (f0 > 0)]
                if f0_pos.size > 0:
                    pitch_mean = float(np.median(f0_pos))
            except Exception:
                try:
                    pitch, magnitude = librosa.piptrack(y=y, sr=sr)
                    pitch_values = pitch[magnitude > np.median(magnitude)]
                    positive_pitch = pitch_values[pitch_values > 0]
                    pitch_mean = float(np.mean(positive_pitch)) if positive_pitch.size > 0 else 0.0
                except Exception:
                    pitch_mean = 0.0

            # Tempo: skip costly tempo estimation in FAST_MODE
            if FAST_MODE:
                tempo = 120.0
            else:
                try:
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    tempo = float(tempo)
                except Exception:
                    tempo = 120.0

        logger.info("Extracted features from %s", path)

        # Ensure mfcc is serializable list
        if isinstance(mfcc_vals, np.ndarray):
            mfcc_list = mfcc_vals.tolist()
        else:
            mfcc_list = list(mfcc_vals)
        return {"mfcc": mfcc_list, "pitch": pitch_mean, "tempo": tempo}
    except Exception as e:
        logger.exception("Feature extraction failed for %s", path)
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