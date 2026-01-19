#!/usr/bin/env python3
import os
import socket
import subprocess
import uuid
import json
import base64
import time
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import portalocker
except Exception:
    portalocker = None

# import shared helpers and router
# Local imports so `uvicorn main:app` works on deploy
import helpers
from audio_api import router as audio_router
from audio_api import warm_sample_cache_background
# reuse optimized helpers from audio_api for cached extraction & parallel compare
from audio_api import _extract_features_with_timeout, _build_match_entry, _get_cached_features

# --- FastAPI app setup ---
app = FastAPI(
    title="Vietnamese Pronunciation Learning API",
    description="API for Vietnamese pronunciation practice with audio comparison",
    version="1.0.0"
)

# CORS middleware with proper headers for audio streaming
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Range",
        "Accept-Ranges",
        "Content-Length",
        "Content-Type",
        "ETag",
        "Cache-Control"
    ]
)

# include audio-related router (uploads, samples, lessons...)
app.include_router(audio_router, prefix="/api", tags=["Audio & Vocab"])
# Also include without prefix for compatibility with clients hitting /analyze
app.include_router(audio_router)

# Serve static files for HTML/CSS/JS frontend
STATIC_DIR = Path("static").resolve()
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static directory (for frontend assets)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve audio samples (legacy support)
if os.path.exists(helpers.SAMPLES_DIR):
    app.mount("/samples", StaticFiles(directory=helpers.SAMPLES_DIR), name="samples")


def get_local_ip() -> Optional[str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def safe_filename(name: Optional[str]) -> str:
    """Return a sanitized filename (no path components / traversal)."""
    if not name:
        return ""
    base = os.path.basename(name)
    base = base.replace("\\", "/").split("/")[-1]
    return base.replace("..", "")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    # Load persisted store in all workers so in-memory data is available
    helpers.load_persisted_store()

    # Attempt to become the leader to run rescan/save (non-blocking lock)
    leader_lock_path = helpers.VOCAB_STORE_FILE + ".leader.lock"
    if portalocker:
        # ensure lock dir exists
        try:
            os.makedirs(os.path.dirname(leader_lock_path) or ".", exist_ok=True)
            open(leader_lock_path, "a").close()
        except Exception:
            pass
        try:
            with open(leader_lock_path, "r+") as lf:
                # Try to become leader, but DO NOT perform write operations during startup.
                # Writing at startup can cause file churn and restart loops when using file watchers.
                try:
                    portalocker.lock(lf, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING)
                    helpers.logger.info("Acquired leader lock; skipping rescan/save at startup to avoid writes (pid=%d)", os.getpid())
                finally:
                    try:
                        portalocker.unlock(lf)
                    except Exception:
                        pass
        except portalocker.exceptions.LockException:
            helpers.logger.info("Leader lock not acquired; skipping rescan/save in this worker (pid=%d)", os.getpid())
        except Exception as e:
            helpers.logger.warning(f"Leader lock attempt failed: {e}")
    else:
        # portalocker not available: do NOT run rescan here to avoid duplicated writes across workers.
        helpers.logger.warning("portalocker not installed; skipping rescan at startup to avoid writes")
    helpers.logger.info("Server started successfully")
    # NOTE: Do NOT warm the entire sample cache at startup. Warming all samples
    # can consume large amounts of RAM and cause issues on constrained hosts
    # (Render free). Warm cache on-demand (via admin endpoint or after rescan).


@app.get("/")
def index():
    """Serve index.html at root for easy access"""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        # Return API info if no index.html
        return {
            "message": "Vietnamese Pronunciation Learning API",
            "docs": "/docs",
            "health": "/health",
            "note": "Put index.html in static/ folder to serve frontend"
        }
    return FileResponse(str(index_path), media_type="text/html")


@app.head("/")
def head_root():
    """Respond to HEAD on root for health probes."""
    return Response(status_code=200)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "samples_count": len(helpers.PERSISTED_STORE.get("samples", {})),
        "lessons_count": len(helpers.LESSONS)
    }


@app.head("/health")
def head_health():
    """HEAD health check endpoint to satisfy probes."""
    return Response(status_code=200)


@app.get("/ping")
def ping():
    return {"pong": True}


@app.get("/whoami")
def whoami():
    return {"ip": get_local_ip(), "port": 8000}


# Note: initial load + rescan moved into the FastAPI startup event to avoid
# duplicate work during module import and when uvicorn spawns/reloads workers.

# -------------------------
# Comparison endpoints (keep in main)
# -------------------------
@app.post("/compare_features")
def api_compare_features(sample: dict, user: dict):
    # sample and user are JSON payloads matching FeaturePayload
    f1 = {"mfcc": sample.get("mfcc", []), "pitch": sample.get("pitch", 0.0), "tempo": sample.get("tempo", 0.0)}
    f2 = {"mfcc": user.get("mfcc", []), "pitch": user.get("pitch", 0.0), "tempo": user.get("tempo", 0.0)}
    mfcc_dist, pitch_diff, tempo_diff = helpers.compare_features_dicts(f1, f2)
    if mfcc_dist < 40:
        feedback = "Rất giống! Bạn phát âm tốt."
    elif mfcc_dist < 80:
        feedback = "Khá giống, cần điều chỉnh một chút."
    else:
        feedback = "Cần luyện thêm — âm khác nhiều so với mẫu."
    if pitch_diff > 30:
        feedback += "\nCao độ khác nhiều, hãy nói cao/trầm hơn."
    if tempo_diff > 20:
        feedback += "\nTốc độ nói khác, thử chậm hoặc nhanh hơn chút."
    return {"mfcc_distance": mfcc_dist, "pitch_diff": pitch_diff, "tempo_diff": tempo_diff, "feedback": feedback}


@app.post("/compare", summary="Compare user audio with sample (upload sample file or provide sample_id)")
async def api_compare(user: Optional[UploadFile] = File(None), file: Optional[UploadFile] = File(None), sample: Optional[UploadFile] = File(None), sample_id: Optional[str] = Form(None), background_tasks: BackgroundTasks = None): # pyright: ignore[reportUndefinedVariable]
    # Accept upload under either `user` or `file` field (client may send `file`)
    upload = user or file
    if not upload:
        raise HTTPException(status_code=400, detail="No user file provided")

    user_bytes = await upload.read()
    if len(user_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty user file")
    if len(user_bytes) > helpers.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="User file too large")

    user_name = safe_filename(upload.filename)
    user_ext = os.path.splitext(user_name)[1] or ".wav"
    user_temp_path = os.path.join(helpers.USERS_DIR, f"u_{uuid.uuid4().hex}{user_ext}")
    with open(user_temp_path, "wb") as f:
        f.write(user_bytes)

    sample_path = None
    sample_uploaded_temp = False
    if sample is not None:
        sample_bytes = await sample.read()
        if len(sample_bytes) > helpers.MAX_UPLOAD_BYTES:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=413, detail="Sample file too large")
        sample_name = safe_filename(sample.filename)
        sample_ext = os.path.splitext(sample_name)[1] or ".wav"
        sample_temp_path = os.path.join(helpers.SAMPLES_DIR, f"s_{uuid.uuid4().hex}{sample_ext}")
        with open(sample_temp_path, "wb") as f:
            f.write(sample_bytes)
        sample_path = sample_temp_path
        sample_uploaded_temp = True
    elif sample_id:
        helpers.load_persisted_store()
        sample_meta = helpers.PERSISTED_STORE.get("samples", {}).get(sample_id)
        if not sample_meta:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=404, detail="sample_id not found")
        fn = sample_meta.get("filename")
        if not fn:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=404, detail="No audio file available for this sample_id")
        sample_path = os.path.join(helpers.SAMPLES_DIR, fn)
        if not os.path.exists(sample_path):
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=404, detail=f"Sample file missing on server: {fn}")
    # Create job and offload heavy work to background task
    job_id = uuid.uuid4().hex
    jobs_dir = os.path.join(helpers.TMP_DIR, "jobs")
    try:
        os.makedirs(jobs_dir, exist_ok=True)
    except Exception:
        pass

    job_file = os.path.join(jobs_dir, f"{job_id}.json")
    # Seed job file
    try:
        with open(job_file, "w", encoding="utf-8") as jf:
            json.dump({"status": "processing", "created": time.time()}, jf)
    except Exception:
        pass

    # schedule background processing
    if background_tasks is None:
        raise HTTPException(status_code=500, detail="Server misconfiguration: BackgroundTasks unavailable")
    background_tasks.add_task(process_compare_job, sample_path, user_temp_path, sample_uploaded_temp, sample_id, job_id)

    return {"status": "processing", "job_id": job_id}


class CompareJSON(BaseModel):
    user_b64: str
    sample_b64: Optional[str] = None
    sample_id: Optional[str] = None
    user_filename: Optional[str] = None
    sample_filename: Optional[str] = None


@app.post("/compare_json", summary="Compare using JSON with base64 audio (user required)")
async def api_compare_json(payload: CompareJSON):
    # decode user audio
    try:
        user_bytes = base64.b64decode(payload.user_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 in user_b64")
    if len(user_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty user audio")
    if len(user_bytes) > helpers.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="User file too large")

    user_name = safe_filename(payload.user_filename or "user.wav")
    user_ext = os.path.splitext(user_name)[1] or ".wav"
    user_temp_path = os.path.join(helpers.USERS_DIR, f"u_{uuid.uuid4().hex}{user_ext}")
    with open(user_temp_path, "wb") as f:
        f.write(user_bytes)

    sample_path = None
    sample_uploaded_temp = False
    if payload.sample_b64:
        try:
            sample_bytes = base64.b64decode(payload.sample_b64)
        except Exception:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail="Invalid base64 in sample_b64")
        if len(sample_bytes) > helpers.MAX_UPLOAD_BYTES:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=413, detail="Sample file too large")
        sample_name = safe_filename(payload.sample_filename or "sample.wav")
        sample_ext = os.path.splitext(sample_name)[1] or ".wav"
        sample_temp_path = os.path.join(helpers.SAMPLES_DIR, f"s_{uuid.uuid4().hex}{sample_ext}")
        with open(sample_temp_path, "wb") as f:
            f.write(sample_bytes)
        sample_path = sample_temp_path
        sample_uploaded_temp = True
    elif payload.sample_id:
        helpers.load_persisted_store()
        sample_meta = helpers.PERSISTED_STORE.get("samples", {}).get(payload.sample_id)
        if not sample_meta:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=404, detail="sample_id not found")
        fn = sample_meta.get("filename")
        if not fn:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=404, detail="No audio file available for this sample_id")
        sample_path = os.path.join(helpers.SAMPLES_DIR, fn)
        if not os.path.exists(sample_path):
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=404, detail=f"Sample file missing on server: {fn}")
    else:
        # attempt to detect sample_id from provided user_filename
        helpers.load_persisted_store()
        samples_meta = helpers.PERSISTED_STORE.get("samples", {})
        user_name_safe = safe_filename(payload.user_filename or "")
        user_base = os.path.splitext(user_name_safe)[0]
        direct_sid = None
        if user_base in samples_meta:
            direct_sid = user_base
        else:
            for sid, meta in samples_meta.items():
                if meta.get("filename") == user_name_safe or meta.get("filename") == os.path.basename(user_name_safe):
                    direct_sid = sid
                    break
        if direct_sid:
            sample_meta = samples_meta.get(direct_sid)
            fn = sample_meta.get("filename") if sample_meta else None
            sample_path = os.path.join(helpers.SAMPLES_DIR, fn) if fn else None
            if not sample_path or not os.path.exists(sample_path):
                try:
                    os.remove(user_temp_path)
                except Exception:
                    pass
                raise HTTPException(status_code=404, detail=f"Matched sample file missing on server: {fn}")
            # delegate to compare helper
            return _compare_audio_paths(sample_path, user_temp_path, False)
        try:
            os.remove(user_temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Either sample_b64, sample_id, or user_filename matching a sample must be provided")

    return _compare_audio_paths(sample_path, user_temp_path, sample_uploaded_temp)


def _compare_audio_paths(sample_path: str, user_temp_path: str, sample_uploaded_temp: bool = False) -> Dict[str, Any]:
    """Convert files, extract features, compare and cleanup. Returns response dict."""
    sample_conv = sample_path + ".conv.wav"
    user_conv = user_temp_path + ".conv.wav"
    try:
        helpers.convert_to_wav16_mono(sample_path, sample_conv)
        helpers.convert_to_wav16_mono(user_temp_path, user_conv)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        try:
            os.remove(user_temp_path)
        except Exception:
            pass
        if sample_uploaded_temp:
            try:
                os.remove(sample_path)
            except Exception:
                pass
        for p in (user_conv, sample_conv):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        detail = "ffmpeg not found on server" if isinstance(e, FileNotFoundError) else "Error converting audio (ffmpeg required)"
        raise HTTPException(status_code=500, detail=detail)

    try:
        f1 = helpers.extract_features(sample_conv)
        f2 = helpers.extract_features(user_conv)
    except Exception:
        for p in (user_temp_path, user_conv, sample_conv):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        if sample_uploaded_temp:
            try:
                if os.path.exists(sample_path):
                    os.remove(sample_path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail="Failed to analyze audio. Ensure files are valid speech recordings.")

    mfcc_dist, pitch_diff, tempo_diff = helpers.compare_features_dicts(f1, f2)

    if mfcc_dist < 40:
        feedback = "Rất giống! Bạn phát âm tốt."
    elif mfcc_dist < 80:
        feedback = "Khá giống, cần điều chỉnh một chút."
    else:
        feedback = "Cần luyện thêm — âm khác nhiều so với mẫu."
    if pitch_diff > 30:
        feedback += "\nCao độ khác nhiều, hãy nói cao/trầm hơn."
    if tempo_diff > 20:
        feedback += "\nTốc độ nói khác, thử chậm hoặc nhanh hơn chút."

    # cleanup temp files
    for p in (user_temp_path, user_conv, sample_conv):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    if sample_uploaded_temp:
        try:
            if os.path.exists(sample_path):
                os.remove(sample_path)
        except Exception:
            pass

    # keep copies of extracted features for the response, then release originals
    features_sample = f1
    features_user = f2
    try:
        del f1, f2
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass

    return {
        "mfcc_distance": mfcc_dist,
        "pitch_diff": pitch_diff,
        "tempo_diff": tempo_diff,
        "feedback": feedback,
        "features_sample": features_sample,
        "features_user": features_user,
    }


# -------------------------
# Background job helpers
# -------------------------
def _job_file_path(job_id: str) -> str:
    jobs_dir = os.path.join(helpers.TMP_DIR, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    return os.path.join(jobs_dir, f"{job_id}.json")


def _save_job_result(job_id: str, payload: dict):
    try:
        jf = _job_file_path(job_id)
        dirn = os.path.dirname(jf) or '.'
        tmp_path = os.path.join(dirn, f".{job_id}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    helpers.logger.debug("fsync not available or failed for %s", tmp_path)
            try:
                os.replace(tmp_path, jf)
            except Exception:
                os.rename(tmp_path, jf)
        except Exception:
            # If writing tmp failed, attempt best-effort direct write
            with open(jf, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
    except Exception:
        try:
            helpers.logger.exception("Failed to save job result: %s", job_id)
        except Exception:
            pass


def _load_job_result(job_id: str) -> Optional[dict]:
    try:
        jf = _job_file_path(job_id)
        if not os.path.exists(jf):
            return None
        # Retry on JSON decode errors (partial write) before treating as processing
        attempts = 3
        for i in range(attempts):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                try:
                    helpers.logger.warning("Partial/invalid job file read for %s, attempt %d/%d", job_id, i+1, attempts)
                except Exception:
                    pass
                try:
                    import time
                    time.sleep(0.05)
                except Exception:
                    pass
                continue
        try:
            helpers.logger.warning("Job file %s appears incomplete; treating as processing", jf)
        except Exception:
            pass
        return {"status": "processing"}
    except Exception:
        return None


def process_compare_job(sample_path: Optional[str], user_temp_path: str, sample_uploaded_temp: bool, sample_id: Optional[str], job_id: str):
    """Background worker that performs the heavy compare work and writes job result file."""
    try:
        # If a sample_path or sample_id was provided, prefer direct compare via helper
        if sample_path or sample_id:
            # If sample_id provided but not sample_path, resolve it
            if not sample_path and sample_id:
                try:
                    helpers.load_persisted_store()
                    sample_meta = helpers.PERSISTED_STORE.get("samples", {}).get(sample_id)
                    if not sample_meta:
                        _save_job_result(job_id, {"status": "error", "error": "sample_id not found"})
                        try:
                            os.remove(user_temp_path)
                        except Exception:
                            pass
                        return
                    fn = sample_meta.get("filename")
                    if not fn:
                        _save_job_result(job_id, {"status": "error", "error": "sample file missing for sample_id"})
                        try:
                            os.remove(user_temp_path)
                        except Exception:
                            pass
                        return
                    sample_path = os.path.join(helpers.SAMPLES_DIR, fn)
                except Exception as e:
                    _save_job_result(job_id, {"status": "error", "error": str(e)})
                    try:
                        os.remove(user_temp_path)
                    except Exception:
                        pass
                    return

            # Perform compare using existing helper (handles conversion & cleanup)
            try:
                result = _compare_audio_paths(sample_path, user_temp_path, sample_uploaded_temp)
                _save_job_result(job_id, {"status": "done", "result": result})
                return
            except HTTPException as he:
                _save_job_result(job_id, {"status": "error", "error": he.detail, "code": he.status_code})
                return

        # Otherwise, perform auto-match flow (no sample provided)
        user_conv = user_temp_path + ".conv.wav"
        try:
            helpers.convert_to_wav16_mono(user_temp_path, user_conv)
        except Exception as e:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            _save_job_result(job_id, {"status": "error", "error": f"conversion_failed: {e}"})
            return

        try:
            f2 = _extract_features_with_timeout(user_conv, timeout=30)
        except Exception as e:
            for p in (user_temp_path, user_conv):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            _save_job_result(job_id, {"status": "error", "error": f"feature_extraction_failed: {e}"})
            return

        try:
            helpers.load_persisted_store()
            samples_meta = helpers.PERSISTED_STORE.get("samples", {})

            # Attempt direct match by filename/sample id from upload name
            user_base = os.path.splitext(os.path.basename(user_temp_path))[0]
            direct_sid = None
            if user_base in samples_meta:
                direct_sid = user_base
            else:
                for sid, meta in samples_meta.items():
                    if meta.get("filename") == os.path.basename(user_temp_path):
                        direct_sid = sid
                        break

            if direct_sid:
                sample_meta = samples_meta.get(direct_sid)
                fn = sample_meta.get("filename") if sample_meta else None
                sample_path = os.path.join(helpers.SAMPLES_DIR, fn) if fn else None
                if not sample_path or not os.path.exists(sample_path):
                    _save_job_result(job_id, {"status": "error", "error": "matched sample file missing"})
                    try:
                        os.remove(user_temp_path)
                    except Exception:
                        pass
                    return

                f1 = _get_cached_features(sample_path) or _extract_features_with_timeout(sample_path, timeout=30)
                mfcc_dist, pitch_diff, tempo_diff = helpers.compare_features_dicts(f1, f2)

                if mfcc_dist < 40:
                    feedback = "Rất giống! Bạn phát âm tốt."
                elif mfcc_dist < 80:
                    feedback = "Khá giống, cần điều chỉnh một chút."
                else:
                    feedback = "Cần luyện thêm — âm khác nhiều so với mẫu."
                if pitch_diff > 30:
                    feedback += "\nCao độ khác nhiều, hãy nói cao/trầm hơn."
                if tempo_diff > 20:
                    feedback += "\nTốc độ nói khác, thử chậm hoặc nhanh hơn chút."

                try:
                    gc.collect()
                except Exception:
                    pass

                resp = {
                    "matched_sample_id": direct_sid,
                    "matched_filename": os.path.basename(sample_path) if sample_path else None,
                    "mfcc_distance": mfcc_dist,
                    "pitch_diff": pitch_diff,
                    "tempo_diff": tempo_diff,
                    "feedback": feedback,
                    "features_sample": f1,
                    "features_user": f2,
                }
                _save_job_result(job_id, {"status": "done", "result": resp})
                return

            # Build candidates and compare in parallel
            candidates = []
            for sid, meta in samples_meta.items():
                fn = meta.get("filename")
                if not fn:
                    continue
                sp = os.path.join(helpers.SAMPLES_DIR, fn)
                if not os.path.exists(sp):
                    continue
                candidates.append((sid, meta, sp))

            matches = []
            if candidates:
                max_workers = min(8, len(candidates))
                with ThreadPoolExecutor(max_workers=max_workers) as comp_exec:
                    futures = [comp_exec.submit(_build_match_entry, s_id, meta, ref_path, f2) for (s_id, meta, ref_path) in candidates]
                    for fut in as_completed(futures):
                        try:
                            res = fut.result()
                            if res:
                                matches.append(res)
                        except Exception:
                            continue

            # cleanup user conv and temp user file
            for p in (user_temp_path, user_conv):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

            if not matches:
                _save_job_result(job_id, {"status": "error", "error": "no_matching_sample_found"})
                return

            matches.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
            best = matches[0]
            best_id = best.get("sample_id")
            best_path = os.path.join(helpers.SAMPLES_DIR, best.get("filename")) if best.get("filename") else None
            best_features = _get_cached_features(best_path) if best_path else None

            mfcc_dist = best.get("details", {}).get("mfcc_distance")
            pitch_diff = best.get("details", {}).get("pitch_difference_hz")
            tempo_diff = best.get("details", {}).get("tempo_difference_bpm")

            if mfcc_dist is None and best_features and f2:
                try:
                    mfcc_dist, pitch_diff, tempo_diff = helpers.compare_features_dicts(best_features, f2)
                except Exception:
                    mfcc_dist, pitch_diff, tempo_diff = None, None, None

            if mfcc_dist is not None:
                if mfcc_dist < 40:
                    feedback = "Rất giống! Bạn phát âm tốt."
                elif mfcc_dist < 80:
                    feedback = "Khá giống, cần điều chỉnh một chút."
                else:
                    feedback = "Cần luyện thêm — âm khác nhiều so với mẫu."
                if pitch_diff and pitch_diff > 30:
                    feedback += "\nCao độ khác nhiều, hãy nói cao/trầm hơn."
                if tempo_diff and tempo_diff > 20:
                    feedback += "\nTốc độ nói khác, thử chậm hoặc nhanh hơn chút."
            else:
                feedback = best.get("comment") or ""

            try:
                gc.collect()
            except Exception:
                pass

            resp = {
                "matched_sample_id": best_id,
                "matched_filename": os.path.basename(best_path) if best_path else None,
                "mfcc_distance": mfcc_dist,
                "pitch_diff": pitch_diff,
                "tempo_diff": tempo_diff,
                "feedback": feedback,
                "features_sample": best_features,
                "features_user": f2,
            }
            _save_job_result(job_id, {"status": "done", "result": resp})
            return

        except Exception as e:
            _save_job_result(job_id, {"status": "error", "error": str(e)})
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            return

    except Exception as e:
        try:
            _save_job_result(job_id, {"status": "error", "error": str(e)})
        except Exception:
            pass


@app.get("/result/{job_id}")
def get_result(job_id: str):
    data = _load_job_result(job_id)
    if data is None:
        return {"status": "processing"}
    return data


# -------------------------
# Mouth analysis endpoint (MediaPipe client will POST here)
# -------------------------

# Create log dir
LOG_DIR = getattr(helpers, "LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

class MouthPayload(BaseModel):
    ts: str
    features: Dict[str, Any]
    meta: Dict[str, Any] = {}

MAR_SPEAK_THRESHOLD = 0.18

def analyze_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Very small heuristic analysis based on normalized mouth opening (MAR)."""
    try:
        mar = float(features.get("normalized_mar", 0.0))
    except Exception:
        mar = 0.0
    mouth_h = float(features.get("mouth_height", 0.0) or 0.0)
    mouth_w = float(features.get("mouth_width", 0.0) or 0.0)
    result = {"normalized_mar": mar, "mouth_h": mouth_h, "mouth_w": mouth_w}
    if mar > MAR_SPEAK_THRESHOLD:
        result["likely_speaking"] = True
        result["note"] = "Mouth open beyond threshold — likely speaking or wide open."
    else:
        result["likely_speaking"] = False
        result["note"] = "Mouth small/closed."
    return result

if getattr(helpers, 'FAST_MODE', False):
    @app.post("/analyze-mouth")
    async def analyze_mouth_disabled(payload: MouthPayload, request: Request):
        return {"ok": False, "message": "analyze-mouth disabled in FAST_MODE"}
else:
    @app.post("/analyze-mouth")
    async def analyze_mouth(payload: MouthPayload, request: Request):
        rec = {"received_at": time.time(), "payload": payload.dict(), "client": request.client.host if request.client else None}
        fname = os.path.join(LOG_DIR, f"mouthlog_{time.strftime('%Y%m%d')}.ndjson")
        try:
            with open(fname, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print("Failed to write mouth log:", e)
        features = payload.features or {}
        analysis = analyze_features(features)
        return {"ok": True, "analysis": analysis, "received_ts": payload.ts}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)