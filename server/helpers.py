import os
import json
import unicodedata
import re
import logging
from pathlib import Path

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
