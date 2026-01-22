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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from size_limit_middleware import MaxBodySizeMiddleware
from pydantic import BaseModel
import uvicorn
import gc
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as _np
try:
    import portalocker
except Exception:
    portalocker = None

# Global lock to limit concurrent heavy jobs to 1 to reduce memory spikes
PROCESS_LOCK = threading.Lock()

# import shared helpers and router
# Local imports so `uvicorn main:app` works on deploy
import helpers
from audio_api import router as audio_router
from audio_api import warm_sample_cache_background
# reuse optimized helpers from audio_api for cached extraction & parallel compare
from audio_api import _extract_features_with_timeout, _build_match_entry, _get_cached_features
# helpers used for streaming saves and vocab mapping
from audio_api import save_upload_streaming, AUDIO_ID_MAP, DEFAULT_UPLOAD_LIMIT, resolve_vocab_sample

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

# Limit request body size (also enforce at reverse proxy in production)
app.add_middleware(MaxBodySizeMiddleware, max_body_size=10 * 1024 * 1024)

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

# In-memory map of job_id -> list of waiting WebSocket clients
job_ws_clients: dict[str, list[WebSocket]] = {}

# Main asyncio event loop reference (set on startup) so background threads
# can schedule coroutines to notify websocket clients.
MAIN_LOOP: asyncio.AbstractEventLoop | None = None


@app.websocket("/ws/status/{job_id}")
async def ws_job_status(ws: WebSocket, job_id: str):
    await ws.accept()

    job_ws_clients.setdefault(job_id, []).append(ws)

    try:
        while True:
            # keep the connection alive; client may send pings
            await ws.receive_text()
    except WebSocketDisconnect:
        try:
            job_ws_clients[job_id].remove(ws)
        except Exception:
            pass


async def notify_ws(job_id: str, result: dict):
    clients = list(job_ws_clients.get(job_id, []))
    for ws in clients:
        try:
            await ws.send_json(result)
        except Exception:
            try:
                job_ws_clients.get(job_id, []).remove(ws)
            except Exception:
                pass


def schedule_notify_ws(job_id: str, result: dict):
    """Schedule notify_ws coroutine on the main event loop from sync/background threads."""
    try:
        loop = MAIN_LOOP
        if loop:
            try:
                asyncio.run_coroutine_threadsafe(notify_ws(job_id, result), loop)
            except Exception:
                pass
    except Exception:
        pass


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

    # Capture the running event loop so background threads can schedule coroutines
    try:
        global MAIN_LOOP
        MAIN_LOOP = asyncio.get_event_loop()
    except Exception:
        pass

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
    # Start job scanner to heal stale 'processing' jobs (multi-instance safety)
    try:
        # Start in a daemon thread so it won't block shutdown
        t = threading.Thread(target=_job_scanner_loop, daemon=True)
        t.start()
        helpers.logger.info("Job scanner thread started (pid=%d)", os.getpid())
    except Exception:
        helpers.logger.warning("Failed to start job scanner thread")
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
async def api_compare(
    request: Request,
    user: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    sample: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    type: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    vocab_id: Optional[str] = Form(None),
    word: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    wait: Optional[bool] = Form(False),
):
    # Accept upload under either `user` or `file` field (client may send `file`)
    upload = user or file

    # Generate a short request id for logging and error correlation
    request_id = uuid.uuid4().hex[:8]

    # Log minimal request info to help diagnose 400s (do not log file contents)
    try:
        client = None
        if hasattr(request, 'client') and request.client:
            client = f"{request.client.host}:{request.client.port}"
        helpers.logger.info("/compare request_id=%s from %s headers=%s filename=%s sample_present=%s sample_id=%s type=%s mode=%s vocab_id=%s",
                            request_id, client, dict(request.headers), getattr(upload, 'filename', None), bool(sample), sample_id, type, mode, vocab_id)
    except Exception:
        try:
            helpers.logger.warning("Failed to log /compare request details (request_id=%s)", request_id)
        except Exception:
            pass
    if not upload:
        # Keep responses JSON-shaped for client compatibility
        return JSONResponse(status_code=400, content={"error": "no_user_file", "detail": "No user file provided", "request_id": request_id})

    # Validate fields
    if type and type != "audio":
        return JSONResponse(status_code=400, content={"error": "invalid_input", "detail": "type must be 'audio'", "request_id": request_id})
    if mode == "vocab" and not vocab_id:
        return JSONResponse(status_code=400, content={"error": "invalid_input", "detail": "vocab_id required for mode=vocab", "request_id": request_id})

    # If mode=vocab, resolve vocabs centrally via resolve_vocab_sample.
    resolved_sample_path = None
    resolved_sample_id = None
    if mode == "vocab":
        allow_permissive = str(os.environ.get("ALLOW_PERMISSIVE_VOCAB", "0")) == "1"
        sid, spath = resolve_vocab_sample(vocab_id, request_id=request_id)
        if not sid or not spath:
            if allow_permissive:
                try:
                    helpers.logger.warning("Unknown vocab_id=%s (request_id=%s); permissive fallback enabled", vocab_id, request_id)
                except Exception:
                    pass
            else:
                return JSONResponse(status_code=400, content={"error": "invalid_input", "detail": "vocab_id not found or invalid", "request_id": request_id})
        else:
            resolved_sample_id = sid
            resolved_sample_path = spath

    # Stream-save user upload to avoid loading into RAM
    try:
        os.makedirs(helpers.USERS_DIR, exist_ok=True)
        user_temp_path, _ = save_upload_streaming(upload, helpers.USERS_DIR, getattr(helpers, "MAX_UPLOAD_BYTES", DEFAULT_UPLOAD_LIMIT))
    except ValueError:
        raise HTTPException(status_code=413, detail="User file too large")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store upload: {e}")

    sample_path = None
    sample_uploaded_temp = False
    # If mode=vocab resolved a sample path earlier, prefer that
    if resolved_sample_path:
        sample_path = resolved_sample_path
    if sample is not None:
        try:
            os.makedirs(helpers.SAMPLES_DIR, exist_ok=True)
            sample_temp_path, _ = save_upload_streaming(sample, helpers.SAMPLES_DIR, getattr(helpers, "MAX_UPLOAD_BYTES", DEFAULT_UPLOAD_LIMIT))
            sample_path = sample_temp_path
            sample_uploaded_temp = True
        except ValueError:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=413, detail="Sample file too large")
        except Exception as e:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to store sample upload: {e}")
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
    # If client requested synchronous handling, perform compare now and return result.
    if wait:
        if not (sample_path or sample_id):
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail="Synchronous compare requires a sample file or sample_id")
        try:
            result = _compare_audio_paths(sample_path, user_temp_path, sample_uploaded_temp)
            return result
        except HTTPException:
            raise
        except Exception as e:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Synchronous compare failed: {e}")

    # Create job and offload heavy work to background task (default behavior)
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

    # schedule background processing: prefer Redis queue if available (multi-instance safe)
    scheduled = False
    try:
        import audio_api as _audio_api
        rc = getattr(_audio_api, 'redis_client', None)
        if rc:
            try:
                qpayload = json.dumps({
                    'task': 'compare',
                    'sample_path': sample_path,
                    'user_temp_path': user_temp_path,
                    'sample_uploaded_temp': bool(sample_uploaded_temp),
                    'sample_id': sample_id,
                    'job_id': job_id,
                })
                # push to queue (use RPUSH so workers BRPOP)
                rc.rpush('job_queue', qpayload)
                try:
                    helpers.logger.info('Enqueued compare job %s to redis queue', job_id)
                except Exception:
                    pass
                scheduled = True
            except Exception:
                scheduled = False
    except Exception:
        scheduled = False

    if not scheduled:
        # fallback to in-process BackgroundTasks
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


def _compare_audio_paths(sample_path: str, user_temp_path: str, sample_uploaded_temp: bool = False, auto_cleanup: bool = True) -> Dict[str, Any]:
    """Convert files, extract features, compare and cleanup. Returns response dict."""
    sample_conv = sample_path + ".conv.wav"
    user_conv = user_temp_path + ".conv.wav"
    try:
        _safe_convert(sample_path, sample_conv)
        _safe_convert(user_temp_path, user_conv)
    except Exception as e:
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
        # Distinguish missing ffmpeg vs conversion error when possible
        if isinstance(e, FileNotFoundError):
            detail = "ffmpeg not found on server"
        else:
            detail = f"conversion_failed: {e}"
        raise HTTPException(status_code=500, detail=detail)

    try:
        f1 = _safe_extract(sample_conv, attempts=3, delay=0.2, timeout=30)
        f2 = _safe_extract(user_conv, attempts=3, delay=0.2, timeout=30)
    except Exception as e:
        try:
            helpers.logger.error("Feature extraction failed for %s and/or %s: %s", sample_conv, user_conv, e, exc_info=True)
        except Exception:
            pass
        # dump listing of directory for debugging
        try:
            d = os.path.dirname(user_conv) or '.'
            helpers.logger.error("Dir listing for %s: %s", d, os.listdir(d))
        except Exception:
            pass
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
        raise HTTPException(status_code=400, detail=f"feature_extraction_failed: {e}")

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

    # cleanup temp files (only if auto_cleanup requested). When called from
    # background workers we pass auto_cleanup=False and perform cleanup after
    # the job result is persisted to avoid race conditions and to retain files
    # for post-mortem debugging.
    if auto_cleanup:
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
                try:
                    helpers.logger.info("Saved job result for job_id=%s path=%s", job_id, jf)
                except Exception:
                    pass
            except Exception:
                os.rename(tmp_path, jf)
                try:
                    helpers.logger.info("Saved job result for job_id=%s path=%s", job_id, jf)
                except Exception:
                    pass
        except Exception:
            # If writing tmp failed, attempt best-effort direct write
            with open(jf, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            try:
                helpers.logger.info("Saved job result for job_id=%s path=%s", job_id, jf)
            except Exception:
                pass
        # Propagate to centralized job store (Redis) if audio_api.job_set available
        try:
            try:
                import audio_api as _audio_api
                if hasattr(_audio_api, 'job_set'):
                    _audio_api.job_set(job_id, payload)
            except Exception:
                pass
        except Exception:
            pass
        # Log important lifecycle events so operator can trace jobs
        try:
            st = payload.get("status") if isinstance(payload, dict) else None
            if st == "done":
                helpers.logger.info("JOB DONE job_id=%s", job_id)
            elif st == "error":
                # include error message if available
                try:
                    err = payload.get("error") or payload.get("message")
                except Exception:
                    err = None
                helpers.logger.error("JOB ERROR job_id=%s error=%s", job_id, err)
        except Exception:
            pass
    except Exception:
        try:
            helpers.logger.exception("Failed to save job result: %s", job_id)
        except Exception:
            pass


def _load_job_result(job_id: str) -> Optional[dict]:
    try:
        # First prefer centralized store (Redis) if available to support multi-instance
        try:
            import audio_api as _audio_api
            j = _audio_api.job_get(job_id)
            if j:
                return j
        except Exception:
            pass

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


def _safe_convert(src: str, dst: str, attempts: int = 3, delay: float = 0.5) -> None:
    """Attempt conversion with retries and verify output exists.

    Uses helpers.convert_to_wav16_mono and verifies that the dst file exists
    before returning. Raises the last exception on failure.
    """
    last_exc = None
    for i in range(attempts):
        try:
            helpers.logger.info("Converting %s -> %s (attempt %d/%d)", src, dst, i + 1, attempts)
            helpers.convert_to_wav16_mono(src, dst)
            if os.path.exists(dst):
                try:
                    st = os.stat(dst)
                    helpers.logger.info("Conversion succeeded, output size=%d bytes", st.st_size)
                except Exception:
                    helpers.logger.info("Conversion succeeded (size unknown)")
                return
            else:
                helpers.logger.warning("convert returned but output missing: %s", dst)
        except Exception as e:
            last_exc = e
            helpers.logger.warning("convert attempt %d failed: %s", i + 1, e)
        try:
            time.sleep(delay)
        except Exception:
            pass
    raise RuntimeError(f"convert_to_wav16_mono failed for {src} -> {dst}: last_exc={last_exc}")


def _safe_extract(path: str, attempts: int = 3, delay: float = 0.2, timeout: int = 30):
    """Wrapper an toàn cho `helpers.extract_features`.

    - Thử nhiều lần nếu file vừa mới được tạo (filesystem visibility).
    - Ghi đầy đủ stacktrace và lưu vào file log khi thất bại.
    - Nếu `helpers.extract_features` trả None/empty thì coi là lỗi.
    """
    last_exc = None
    for i in range(attempts):
        if not path or not os.path.exists(path):
            last_exc = FileNotFoundError(path)
            try:
                helpers.logger.warning("Expected file missing before extract: %s (attempt %d/%d)", path, i + 1, attempts)
            except Exception:
                pass
            try:
                time.sleep(delay)
            except Exception:
                pass
            continue
        try:
            helpers.logger.info("Starting feature extract for %s (attempt %d/%d)", path, i + 1, attempts)
            feats = helpers.extract_features(path)
            if not feats:
                raise RuntimeError("helpers.extract_features returned empty/None")
            helpers.logger.info("Extract succeeded for %s", path)
            return feats
        except Exception as e:
            last_exc = e
            try:
                helpers.logger.warning("Extract attempt %d/%d failed for %s: %s", i + 1, attempts, path, repr(e))
            except Exception:
                pass
            # write full traceback to a diagnostic file for post-mortem
            try:
                errdir = os.path.join("data", "tmp")
                os.makedirs(errdir, exist_ok=True)
                errfile = os.path.join(errdir, f"extract_error_{int(time.time()*1000)}.log")
                with open(errfile, "w", encoding="utf-8") as ef:
                    ef.write("Exception repr:\n")
                    ef.write(repr(e) + "\n\n")
                    ef.write("Stacktrace:\n")
                    traceback.print_exc(file=ef)
                helpers.logger.info("Wrote extract stacktrace to %s", errfile)
            except Exception:
                try:
                    helpers.logger.exception("Failed to write extract stacktrace")
                except Exception:
                    pass
            try:
                time.sleep(delay * (i + 1))
            except Exception:
                pass
            continue

    msg = f"extract_features failed for {path}: last_exc={repr(last_exc)}"
    raise RuntimeError(msg)


def _job_scanner_loop():
    """Background loop that scans local job files and heals stale 'processing' jobs.

    Marks jobs as error if they've been in 'processing' state longer than JOB_STALE_SECONDS.
    This helps in multi-instance deployments where a worker may die before persisting final state.
    """
    try:
        STALE = int(os.environ.get("JOB_STALE_SECONDS", "600"))
        INTERVAL = int(os.environ.get("JOB_SCAN_INTERVAL", "60"))
    except Exception:
        STALE = 600
        INTERVAL = 60

    jobs_dir = os.path.join(helpers.TMP_DIR, "jobs")
    while True:
        try:
            if not os.path.isdir(jobs_dir):
                time.sleep(INTERVAL)
                continue
            for fname in os.listdir(jobs_dir):
                if not fname.endswith('.json'):
                    continue
                jid = fname[:-5]
                try:
                    data = _load_job_result(jid)
                    # If centralized store reports done/error, skip
                    if not data:
                        continue
                    status = data.get('status') if isinstance(data, dict) else None
                    if status == 'processing':
                        created = data.get('created') or data.get('created_at') or 0
                        try:
                            created = float(created)
                        except Exception:
                            created = 0
                        age = time.time() - created if created else None
                        if age is not None and age > STALE:
                            helpers.logger.warning("Job scanner: marking stale job %s as error (age=%s)", jid, age)
                            _save_job_result(jid, {"status": "error", "error": "stale-job-marked-error"})
                except Exception:
                    continue
        except Exception:
            try:
                helpers.logger.exception("Job scanner loop error")
            except Exception:
                pass
        try:
            time.sleep(INTERVAL)
        except Exception:
            break


def process_compare_job(sample_path: Optional[str], user_temp_path: str, sample_uploaded_temp: bool, sample_id: Optional[str], job_id: str):
    """Background worker that performs the heavy compare work and writes job result file."""
    saved = False
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
                        saved = True
                        try:
                            os.remove(user_temp_path)
                        except Exception:
                            pass
                        return
                    fn = sample_meta.get("filename")
                    if not fn:
                        _save_job_result(job_id, {"status": "error", "error": "sample file missing for sample_id"})
                        saved = True
                        try:
                            os.remove(user_temp_path)
                        except Exception:
                            pass
                        return
                    sample_path = os.path.join(helpers.SAMPLES_DIR, fn)
                except Exception as e:
                    _save_job_result(job_id, {"status": "error", "error": str(e)})
                    saved = True
                    try:
                        os.remove(user_temp_path)
                    except Exception:
                        pass
                    return

            # Perform compare using existing helper (handles conversion & cleanup)
            try:
                # Defer cleanup until after job result is persisted so operators can
                # inspect files if something goes wrong.
                result = _compare_audio_paths(sample_path, user_temp_path, sample_uploaded_temp, auto_cleanup=False)
                payload = {"status": "done", "result": result}
                _save_job_result(job_id, payload)
                try:
                    schedule_notify_ws(job_id, payload)
                except Exception:
                    pass
                # Now that result is saved, perform cleanup of temp files owned by this job.
                try:
                    sample_conv = sample_path + ".conv.wav"
                    user_conv = user_temp_path + ".conv.wav"
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
                except Exception:
                    pass
                saved = True
                return
            except HTTPException as he:
                _save_job_result(job_id, {"status": "error", "error": he.detail, "code": he.status_code})
                saved = True
                return

        # Otherwise, perform auto-match flow (no sample provided)
        user_conv = user_temp_path + ".conv.wav"
        try:
            _safe_convert(user_temp_path, user_conv)
        except Exception as e:
            try:
                helpers.logger.error("Conversion failed for %s -> %s: %s", user_temp_path, user_conv, e, exc_info=True)
            except Exception:
                pass
            try:
                # list dir to aid debugging
                helpers.logger.error("Dir listing for %s: %s", os.path.dirname(user_conv) or '.', os.listdir(os.path.dirname(user_conv) or '.'))
            except Exception:
                pass
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            _save_job_result(job_id, {"status": "error", "error": f"conversion_failed: {e}"})
            saved = True
            return

        try:
            f2 = _safe_extract(user_conv, attempts=3, delay=0.2, timeout=30)
        except Exception as e:
            try:
                helpers.logger.error("Feature extraction for %s failed: %s", user_conv, e, exc_info=True)
            except Exception:
                pass
            for p in (user_temp_path, user_conv):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            _save_job_result(job_id, {"status": "error", "error": f"feature_extraction_failed: {e}"})
            saved = True
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
                    saved = True
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
                payload = {"status": "done", "result": resp}
                _save_job_result(job_id, payload)
                try:
                    schedule_notify_ws(job_id, payload)
                except Exception:
                    pass
                saved = True
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
                max_workers = min(4, len(candidates))
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
                saved = True
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
            payload = {"status": "done", "result": resp}
            _save_job_result(job_id, payload)
            try:
                schedule_notify_ws(job_id, payload)
            except Exception:
                pass
            saved = True
            return

        except Exception as e:
            _save_job_result(job_id, {"status": "error", "error": str(e)})
            saved = True
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            return

    except Exception as e:
        try:
            # Log exception with stacktrace for operator visibility
            try:
                helpers.logger.exception("JOB FAILED job_id=%s", job_id)
            except Exception:
                pass
            _save_job_result(job_id, {"status": "error", "error": str(e)})
        except Exception:
            pass
        saved = True
    finally:
        # Final-guard: if the job completed the heavy work but failed to persist
        # a final state (e.g., process returned early or an unexpected path),
        # ensure we do not leave the job in 'processing' forever.
        try:
            if not saved:
                cur = _load_job_result(job_id)
                # If no data or still processing, mark as error to unblock clients
                if not cur or (isinstance(cur, dict) and cur.get("status") == "processing"):
                    helpers.logger.warning("Final-guard: job %s exited without saving; marking as error", job_id)
                    _save_job_result(job_id, {"status": "error", "error": "worker-exited-without-saving"})
        except Exception:
            pass


# TEST-ONLY: trigger notify for a job_id with provided JSON payload.
# This endpoint is intended for local manual testing and should be removed
# or protected in production.
@app.post("/__test_notify/{job_id}")
async def _test_notify(job_id: str, payload: dict):
    try:
        data = {"status": "done", "result": payload}
        _save_job_result(job_id, data)
        # schedule notify on MAIN_LOOP
        try:
            schedule_notify_ws(job_id, data)
        except Exception:
            pass
        return {"notified": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/result/{job_id}")
def get_result(job_id: str):
    data = _load_job_result(job_id)
    if data is None:
        return {"status": "processing"}
    return data


@app.get("/result/analysis/{job_id}")
def get_result_analysis(job_id: str):
    """Return detailed analysis metrics for a completed job using saved features.

    Requires the job result to include `result.features_sample` and `result.features_user`
    (the mean-MFCC vectors plus pitch and tempo). Returns MFCC euclidean distance,
    MFCC cosine similarity, pitch diff, tempo diff and the original result payload.
    """
    data = _load_job_result(job_id)
    if data is None:
        return {"status": "processing"}
    if not isinstance(data, dict):
        return {"status": "error", "error": "invalid job data"}
    if data.get("status") != "done":
        return {"status": data.get("status", "processing")}
    res = data.get("result") or {}
    f_sample = res.get("features_sample")
    f_user = res.get("features_user")
    if not f_sample or not f_user:
        return {"status": "done", "message": "No feature vectors available for analysis", "result": res}

    try:
        a = _np.array(f_sample.get("mfcc") if isinstance(f_sample, dict) else f_sample)
        b = _np.array(f_user.get("mfcc") if isinstance(f_user, dict) else f_user)
        # Ensure 1D vectors
        a = a.flatten()
        b = b.flatten()
        # Pad shorter vector with zeros if lengths differ
        if a.shape[0] != b.shape[0]:
            mx = max(a.shape[0], b.shape[0])
            a = _np.pad(a, (0, mx - a.shape[0]))
            b = _np.pad(b, (0, mx - b.shape[0]))

        euclid = float(_np.linalg.norm(a - b))
        # cosine similarity (1 means identical)
        denom = float(_np.linalg.norm(a) * _np.linalg.norm(b))
        cos_sim = float(_np.dot(a, b) / denom) if denom > 0 else 0.0
    except Exception as e:
        return {"status": "done", "error": f"failed to compute mfcc metrics: {e}", "result": res}

    pitch_s = None
    tempo_s = None
    try:
        pitch_s = float(res.get("pitch_diff")) if res.get("pitch_diff") is not None else None
    except Exception:
        try:
            # Some results embed pitch under details
            pitch_s = float(res.get("details", {}).get("pitch_difference_hz"))
        except Exception:
            pitch_s = None
    try:
        tempo_s = float(res.get("tempo_diff")) if res.get("tempo_diff") is not None else None
    except Exception:
        try:
            tempo_s = float(res.get("details", {}).get("tempo_difference_bpm"))
        except Exception:
            tempo_s = None

    analysis = {
        "mfcc_euclidean": euclid,
        "mfcc_cosine_similarity": cos_sim,
        "pitch_difference_hz": pitch_s,
        "tempo_difference_bpm": tempo_s,
        "original_result": res,
    }
    return {"status": "done", "analysis": analysis}


# Mouth/face analysis endpoint removed.
# Previously this app accepted consumer MediaPipe snapshots at `/analyze-mouth`.
# That functionality has been intentionally removed to disable facial/mouth analysis.


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)