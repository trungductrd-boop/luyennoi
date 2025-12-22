#!/usr/bin/env python3
import os
import socket
import subprocess
import uuid
import json
import time
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# import shared helpers and router
# Local imports so `uvicorn main:app` works on deploy
import helpers
from audio_api import router as audio_router

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
    helpers.load_persisted_store()
    helpers.logger.info("Server started successfully")


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


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "samples_count": len(helpers.PERSISTED_STORE.get("samples", {})),
        "lessons_count": len(helpers.LESSONS)
    }


@app.get("/ping")
def ping():
    return {"pong": True}


@app.get("/whoami")
def whoami():
    return {"ip": get_local_ip(), "port": 8000}


# perform initial load + rescan at startup
helpers.load_persisted_store()
try:
    startup_report = helpers.rescan_samples()
    print("startup rescan report:", startup_report)
except Exception as e:
    print("startup rescan failed:", e)

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
async def api_compare(user: UploadFile = File(...), sample: Optional[UploadFile] = File(None), sample_id: Optional[str] = Form(None)):
    user_bytes = await user.read()
    if len(user_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty user file")
    if len(user_bytes) > helpers.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="User file too large")

    user_name = safe_filename(user.filename)
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
    else:
        try:
            os.remove(user_temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Either sample file or sample_id must be provided")

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

    return {
        "mfcc_distance": mfcc_dist,
        "pitch_diff": pitch_diff,
        "tempo_diff": tempo_diff,
        "feedback": feedback,
        "features_sample": f1,
        "features_user": f2,
    }


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