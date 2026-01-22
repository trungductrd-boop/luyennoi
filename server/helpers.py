import os
import json
import unicodedata
import re
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
UPLOADS_DIR = BASE_DIR / "uploads"

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Sanitize filename: normalize unicode, strip diacritics, replace whitespace,
    remove unsafe chars and limit length."""
    if not name:
        return ""
    base = os.path.basename(name)
    nk = unicodedata.normalize("NFKD", base)
    no_diac = ''.join(c for c in nk if not unicodedata.combining(c))
    replaced = re.sub(r"\s+", "_", no_diac)
    safe = re.sub(r"[^A-Za-z0-9._\-]", "", replaced)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip('._-')
    if not safe:
        return ""
    if len(safe) > 255:
        safe = safe[:255]
    return safe


def job_file_path(job_id: str) -> str:
    return str(JOBS_DIR / f"{job_id}.json")


def save_job_result(job_id: str, payload: dict):
    try:
        jf = job_file_path(job_id)
        # Write atomically: write to temp file in same dir, fsync, then replace
        dirn = os.path.dirname(jf) or '.'
        tmp_path = os.path.join(dirn, f".{job_id}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                # fsync may not be available on some platforms; ignore but log
                logger.debug("fsync not available or failed for %s", tmp_path)
        # Atomic replace
        try:
            os.replace(tmp_path, jf)
        except Exception:
            # fallback to rename
            os.rename(tmp_path, jf)
        logger.info(f"Saved job result: {job_id}")
    except Exception as e:
        logger.exception("Failed to save job result: %s", e)


def load_job_result(job_id: str):
    try:
        jf = job_file_path(job_id)
        if not os.path.exists(jf):
            return None
        # Attempt to load JSON; if file was mid-write we may get JSONDecodeError.
        # Retry a few times with short sleep before treating as still processing.
        attempts = 3
        for i in range(attempts):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Partial/invalid job file read for %s, attempt %d/%d", job_id, i+1, attempts)
                time_sleep = 0.05
                try:
                    import time
                    time.sleep(time_sleep)
                except Exception:
                    pass
                continue
        # If still invalid after retries, assume worker is still processing and return processing status
        logger.warning("Job file %s appears incomplete; treating as processing", jf)
        return {"status": "processing"}
    except Exception as e:
        logger.exception("Failed to load job result: %s", e)
        return None


# --- audio id map helpers ---
_AUDIO_ID_MAP: Optional[Dict[str, str]] = None
_ASSET_TO_IDS: Optional[Dict[str, list]] = None


def _audio_id_map_path() -> str:
    # data/samples/audio_id_map.json relative to repository root
    return str(Path(__file__).resolve().parent.parent / "data" / "samples" / "audio_id_map.json")


def load_audio_id_map() -> Dict[str, str]:
    """Load and cache the audio id -> asset path mapping."""
    global _AUDIO_ID_MAP, _ASSET_TO_IDS
    if _AUDIO_ID_MAP is not None:
        return _AUDIO_ID_MAP
    path = _audio_id_map_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            _AUDIO_ID_MAP = json.load(f)
    except Exception:
        logger.exception("Failed to load audio_id_map from %s", path)
        _AUDIO_ID_MAP = {}

    # build reverse map asset -> [ids]
    _ASSET_TO_IDS = {}
    for k, v in _AUDIO_ID_MAP.items():
        _ASSET_TO_IDS.setdefault(v, []).append(k)
    return _AUDIO_ID_MAP


def get_canonical_id_for_asset(asset_path: str) -> Optional[str]:
    """Return a deterministic canonical id for a given asset path.

    If multiple ids map to the same asset, picks the lexicographically smallest id.
    """
    global _ASSET_TO_IDS
    if _ASSET_TO_IDS is None:
        load_audio_id_map()
    ids = _ASSET_TO_IDS.get(asset_path)
    if not ids:
        return None
    return sorted(ids)[0]


def resolve_vocab_id_from_request(req, allow_fallback: bool = True) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """Resolve `vocab_id` coming from a Flask `request`.

    Returns tuple (found, vocab_id, asset_path, error_message).
    - Accepts `vocab_id` from form, JSON body, or query string.
    - Converts numeric values to strings.
    - If vocab_id not found in map, will attempt fallback by `word` (basename of asset) when `allow_fallback`.
    - Logs lookup steps.
    """
    raw_vid = None
    # try form > json body > query
    try:
        if hasattr(req, "form") and req.form:
            raw_vid = req.form.get("vocab_id")
        if raw_vid is None:
            # attempt JSON
            try:
                jb = req.get_json(silent=True)
            except Exception:
                jb = None
            if jb and isinstance(jb, dict):
                raw_vid = jb.get("vocab_id")
        if raw_vid is None:
            raw_vid = req.args.get("vocab_id")
    except Exception:
        raw_vid = None

    # normalize to str when present
    if raw_vid is not None:
        try:
            vid = str(raw_vid)
        except Exception:
            vid = None
    else:
        vid = None

    audio_map = load_audio_id_map()
    logger.info("Incoming vocab lookup: raw=%r form=%r args=%r json_vocab=%r", raw_vid, getattr(req, 'form', None), getattr(req, 'args', None), (req.get_json(silent=True) if hasattr(req, 'get_json') else None))

    if vid:
        asset = audio_map.get(vid)
        if asset:
            # found by id
            logger.info("vocab_id resolved: %s -> %s", vid, asset)
            # return canonical id in case of duplicates
            canonical = get_canonical_id_for_asset(asset) or vid
            return True, canonical, asset, None
        else:
            logger.warning("vocab_id %s not found in audio map", vid)

    # fallback by word/basename
    if allow_fallback:
        # try form fields commonly used
        word = None
        try:
            if hasattr(req, 'form') and req.form:
                word = req.form.get('word') or req.form.get('basename')
        except Exception:
            word = None
        if not word:
            # if file uploaded, use filename
            try:
                f = getattr(req, 'files', None)
                if f:
                    ff = f.get('file')
                    if ff and getattr(ff, 'filename', None):
                        word = Path(ff.filename).stem
            except Exception:
                pass
        if not word:
            # try JSON 'word'
            try:
                jb = req.get_json(silent=True)
                if jb and isinstance(jb, dict):
                    word = jb.get('word')
            except Exception:
                pass

        if word:
            # canonicalize: look for asset paths that end with the basename
            cand = None
            for asset_path in (list(audio_map.values()) if audio_map else []):
                if Path(asset_path).stem == str(word):
                    cand = asset_path
                    break
            if cand:
                cid = get_canonical_id_for_asset(cand)
                logger.info("Fallback by word: %s -> %s (id=%s)", word, cand, cid)
                return True, cid, cand, None
        logger.warning("Fallback by word failed for word=%r", word)

    # not found
    return False, vid, None, "unknown vocab_id"
