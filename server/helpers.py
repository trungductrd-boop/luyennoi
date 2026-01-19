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
        with open(jf, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        logger.info(f"Saved job result: {job_id}")
    except Exception as e:
        logger.exception("Failed to save job result: %s", e)


def load_job_result(job_id: str):
    try:
        jf = job_file_path(job_id)
        if not os.path.exists(jf):
            return None
        with open(jf, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to load job result: %s", e)
        return None
