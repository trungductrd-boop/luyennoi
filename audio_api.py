from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, Response, JSONResponse
from typing import Optional
from pathlib import Path
import os
import uuid
import zipfile
import shutil
import mimetypes
import re

# audio helpers & globals
import wave as _wave
import helpers
from pydantic import BaseModel, field_validator

router = APIRouter()

# Constants for streaming
CHUNK_SIZE = 1024 * 1024  # 1MB

import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Background executor for light async processing
executor = ThreadPoolExecutor(max_workers=8)

# In-memory cache for extracted sample features to avoid repeated costly extraction
SAMPLE_FEATURES: dict = {}
SAMPLE_FEATURES_LOCK = threading.Lock()


def _get_cached_features(path: str):
	"""Return cached features for `path` or extract and cache them. Returns None on failure."""
	if not path:
		return None
	key = os.path.abspath(path)
	with SAMPLE_FEATURES_LOCK:
		if key in SAMPLE_FEATURES:
			return SAMPLE_FEATURES[key]
	try:
		feats = helpers.extract_features(path)
	except Exception:
		return None
	with SAMPLE_FEATURES_LOCK:
		SAMPLE_FEATURES[key] = feats
	return feats


def _build_match_entry(sample_id: str, meta: dict, ref_path: str, user_features: dict):
	"""Compute comparison between `user_features` and reference at `ref_path` and return match dict or None."""
	try:
		ref_features = _get_cached_features(ref_path)
		if not ref_features:
			return None
		mfcc_dist, pitch_diff, tempo_diff = helpers.compare_features_dicts(user_features, ref_features)

		mfcc_score = max(0, 100 - mfcc_dist * 2)
		pitch_score = max(0, 100 - pitch_diff * 0.5)
		tempo_score = max(0, 100 - tempo_diff * 0.5)
		overall_score = (mfcc_score * 0.6 + pitch_score * 0.3 + tempo_score * 0.1)

		vocab_info = _get_vocab_info(sample_id, meta)

		return {
			"sample_id": sample_id,
			"filename": os.path.basename(ref_path),
			"lesson_id": meta.get("lesson_id"),
			"vocab_id": meta.get("vocab_id"),
			"word": vocab_info.get("word", "Unknown"),
			"meaning": vocab_info.get("meaning", ""),
			"overall_score": round(overall_score, 2),
			"scores": {
				"mfcc": round(mfcc_score, 2),
				"pitch": round(pitch_score, 2),
				"tempo": round(tempo_score, 2)
			},
			"details": {
				"mfcc_distance": round(mfcc_dist, 4),
				"pitch_difference_hz": round(pitch_diff, 2),
				"tempo_difference_bpm": round(tempo_diff, 2)
			},
			"comment": _get_review(overall_score, mfcc_dist, pitch_diff, tempo_diff)
		}
	except Exception:
		return None


# In-memory task store (task_id -> meta)
UPLOAD_TASKS: dict = {}

# Default upload limit (can be overridden per endpoint)
DEFAULT_UPLOAD_LIMIT = 10 * 1024 * 1024  # 10 MB


def _detect_audio_type_from_file(path: Path) -> Optional[str]:
	"""Basic magic-byte sniffing for common audio types. Returns MIME type or None."""
	try:
		with open(path, "rb") as f:
			head = f.read(64)
	except Exception:
		return None

	if head.startswith(b"RIFF") and b"WAVE" in head[8:16]:
		return "audio/wav"
	if head.startswith(b"ID3") or head[0:2] == b"\xff\xfb":
		return "audio/mpeg"
	if head.startswith(b"OggS"):
		return "audio/ogg"
	if head.startswith(b"fLaC"):
		return "audio/flac"
	# MP4 family (including m4a, 3gp) contains 'ftyp' at offset 4
	if b"ftyp" in head[4:16]:
		return "audio/mp4"
	return None


def _require_auth(request: Request) -> bool:
	"""Simple API key auth: checks Authorization: Bearer <key> against env UPLOAD_API_KEY or helpers.UPLOAD_API_KEY"""
	auth = request.headers.get("authorization") or request.headers.get("Authorization")
	if not auth:
		return False
	parts = auth.split()
	if len(parts) != 2 or parts[0].lower() != "bearer":
		return False
	token = parts[1]
	expected = os.environ.get("UPLOAD_API_KEY") or getattr(helpers, "UPLOAD_API_KEY", None)
	return expected is not None and token == expected


def _enqueue_processing(task_id: str, src_path: str, original_filename: str, metadata: dict):
	"""Submit a background task to convert and move file to samples, update persisted store."""
	def job():
		try:
			sample_id = str(uuid.uuid4())[:8]
			_, ext = os.path.splitext(original_filename)
			ext = ext or ".wav"
			dst_fname = f"{sample_id}{ext}"
			dst_path = os.path.join(helpers.SAMPLES_DIR, dst_fname)

			# If source is already WAV 16k mono, skip conversion and copy
			try:
				need_convert = True
				if src_path.lower().endswith('.wav'):
					try:
						with _wave.open(src_path, 'rb') as wf:
							channels = wf.getnchannels()
							sr = wf.getframerate()
						if channels == 1 and sr == 16000:
							need_convert = False
					except Exception:
						need_convert = True

				if not need_convert:
					shutil.copy2(src_path, dst_path)
				else:
					try:
						conv_path = dst_path
						helpers.convert_to_wav16_mono(src_path, conv_path)
					except Exception:
						# If conversion fails, copy raw
						shutil.copy2(src_path, dst_path)
			except Exception:
				# ensure we still try to copy if anything unexpected
				try:
					shutil.copy2(src_path, dst_path)
				except Exception:
					pass

			# Register sample in persisted store
			helpers.load_persisted_store()
			helpers.PERSISTED_STORE.setdefault("samples", {})
			helpers.PERSISTED_STORE[sample_id] = {"filename": os.path.basename(dst_path), "lesson_id": metadata.get("lesson_id"), "vocab_id": None}
			helpers.save_persisted_store()

			UPLOAD_TASKS[task_id]["status"] = "done"
			UPLOAD_TASKS[task_id]["sample_id"] = sample_id
			UPLOAD_TASKS[task_id]["filename"] = os.path.basename(dst_path)
		except Exception as e:
			UPLOAD_TASKS[task_id]["status"] = "error"
			UPLOAD_TASKS[task_id]["error"] = str(e)
		finally:
			try:
				os.remove(src_path)
			except Exception:
				pass

	executor.submit(job)


# -------------------------
# Upload endpoint
# -------------------------
@router.post("/upload", summary="Upload audio file (multipart/form-data)")
async def api_upload(
	request: Request,
	file: UploadFile = File(...),
	user_id: Optional[str] = Form(None),
	lesson_id: Optional[str] = Form(None),
	metadata: Optional[str] = Form(None),
	max_size: Optional[int] = None,
):
	"""Stream-upload an audio file with chunked write, size limit, magic-byte validation, and async processing.

	Returns 202 with task_id for background processing.
	"""
	# Auth
	if not _require_auth(request):
		raise HTTPException(status_code=401, detail="Unauthorized")

	limit = int(max_size) if max_size else DEFAULT_UPLOAD_LIMIT
	limit = min(limit, 50 * 1024 * 1024)  # hard cap 50MB

	# Ensure temp dir
	os.makedirs(helpers.TMP_DIR, exist_ok=True)
	task_id = str(uuid.uuid4())
	tmp_fname = f"upload_{task_id}.tmp"
	tmp_path = os.path.join(helpers.TMP_DIR, tmp_fname)

	# Stream write
	total = 0
	try:
		with open(tmp_path, "wb") as out_f:
			while True:
				chunk = await file.read(8192)
				if not chunk:
					break
				total += len(chunk)
				if total > limit:
					out_f.close()
					try:
						os.remove(tmp_path)
					except Exception:
						pass
					raise HTTPException(status_code=413, detail="File too large")
				out_f.write(chunk)
	except HTTPException:
		raise
	except Exception as e:
		try:
			os.remove(tmp_path)
		except Exception:
			pass
		raise HTTPException(status_code=500, detail=f"Failed to upload: {e}")

	# Basic magic-byte validation
	detected = _detect_audio_type_from_file(Path(tmp_path))
	if not detected or not detected.startswith("audio/"):
		try:
			os.remove(tmp_path)
		except Exception:
			pass
		raise HTTPException(status_code=400, detail="Uploaded file is not a recognized audio format")

	# Parse metadata JSON if provided
	meta_obj = {}
	if metadata:
		try:
			meta_obj = json.loads(metadata)
		except Exception:
			meta_obj = {"raw": metadata}

	# Register task
	UPLOAD_TASKS[task_id] = {
		"status": "queued",
		"original_filename": file.filename,
		"size": total,
		"content_type": detected,
		"uploaded_at": time.time(),
		"user_id": user_id,
		"lesson_id": lesson_id,
		"metadata": meta_obj,
	}

	# Enqueue background processing
	_enqueue_processing(task_id, tmp_path, file.filename, {"lesson_id": lesson_id, **meta_obj})

	return JSONResponse(status_code=202, content={"status": "accepted", "task_id": task_id, "message": "processing"})


@router.get("/upload/status/{task_id}")
def api_upload_status(task_id: str):
	t = UPLOAD_TASKS.get(task_id)
	if not t:
		raise HTTPException(status_code=404, detail="Task not found")
	return {"task_id": task_id, **t}


class VocabItemIn(BaseModel):
	id: Optional[str] = None
	word: str
	meaning: Optional[str] = ""
	example: Optional[str] = ""
	audio_filename: Optional[str] = None
	sample_id: Optional[str] = None
	
	@field_validator('word')
	@classmethod
	def word_not_empty(cls, v):
		if not v or not v.strip():
			raise ValueError('Word cannot be empty')
		return v.strip()

# Helper functions for audio streaming
def get_mime_type(path: Path) -> str:
	"""Get MIME type for file"""
	mt, _ = mimetypes.guess_type(str(path))
	return mt or "application/octet-stream"

def safe_resolve_filename(filename: str) -> Path:
	"""Safely resolve filename and prevent path traversal"""
	if "/" in filename or "\\" in filename or ".." in filename:
		raise HTTPException(status_code=400, detail="Invalid filename")
	
	file_path = (Path(helpers.SAMPLES_DIR) / filename).resolve()
	samples_dir = Path(helpers.SAMPLES_DIR).resolve()
	
	if not str(file_path).startswith(str(samples_dir)):
		raise HTTPException(status_code=400, detail="Invalid file path")
	
	return file_path

# -------------------------
# API: lessons & vocab
# -------------------------
@router.get("/lessons")
def api_get_lessons():
	return helpers.LESSONS

@router.get("/lessons/{lesson_id}/vocab")
def api_get_vocab(lesson_id: str):
	return helpers.merged_vocab_for_lesson(lesson_id)

@router.put("/lessons/{lesson_id}/progress")
def api_update_progress(lesson_id: str, progress: int):
	# Persist progress to JSON instead of memory
	helpers.load_persisted_store()
	helpers.PERSISTED_STORE.setdefault("progress", {})
	helpers.PERSISTED_STORE["progress"][lesson_id] = max(0, min(100, int(progress)))
	helpers.save_persisted_store()
	
	# Also update in-memory for compatibility
	for l in helpers.LESSONS:
		if l["id"] == lesson_id:
			l["progress"] = helpers.PERSISTED_STORE["progress"][lesson_id]
			return {"ok": True, "lesson": l}
	raise HTTPException(status_code=404, detail="Lesson not found")

# -------------------------
# API: samples listing, upload, rescan
# -------------------------
@router.get("/samples", summary="List sample files available on server (auto-rescan is triggered)")
def api_list_samples():
	# ensure samples dir exists
	os.makedirs(helpers.SAMPLES_DIR, exist_ok=True)
	report = helpers.rescan_samples()
	files = []
	for fn in sorted(os.listdir(helpers.SAMPLES_DIR)):
		path = os.path.join(helpers.SAMPLES_DIR, fn)
		if os.path.isfile(path):
			files.append({"filename": fn, "audio_url": f"/static/samples/{fn}"})
	return {"count": len(files), "files": files, "rescan_report": report}

@router.post("/samples/upload-simple", summary="Upload sample and register (simple). Returns sample_id and audio_url.")
async def api_upload_simple(file: UploadFile = File(...), lesson_id: Optional[str] = Form(None), vocab_id: Optional[str] = Form(None)):
	contents = await file.read()
	if len(contents) == 0:
		raise HTTPException(status_code=400, detail="Empty file")
	if len(contents) > helpers.MAX_UPLOAD_BYTES:
		raise HTTPException(status_code=413, detail="File too large")
	_, ext = os.path.splitext(file.filename)
	ext = ext or ".wav"
	sample_id = str(uuid.uuid4())[:8]
	raw_fname = f"{sample_id}{ext}"
	# ensure samples dir exists before writing
	os.makedirs(helpers.SAMPLES_DIR, exist_ok=True)
	raw_path = os.path.join(helpers.SAMPLES_DIR, raw_fname)
	with open(raw_path, "wb") as f:
		f.write(contents)

	conv_fname = f"{sample_id}.wav"
	conv_path = os.path.join(helpers.SAMPLES_DIR, conv_fname)
	try:
		helpers.convert_to_wav16_mono(raw_path, conv_path)
		if raw_path != conv_path:
			try:
				os.remove(raw_path)
			except Exception:
				pass
	except Exception:
		conv_fname = raw_fname
		conv_path = raw_path

	helpers.load_persisted_store()
	helpers.PERSISTED_STORE.setdefault("samples", {})
	helpers.PERSISTED_STORE["samples"][sample_id] = {"filename": conv_fname, "lesson_id": lesson_id, "vocab_id": vocab_id}
	helpers.save_persisted_store()
	audio_url = f"/static/samples/{conv_fname}"
	return {"ok": True, "sample_id": sample_id, "audio_filename": conv_fname, "audio_url": audio_url}

@router.post("/samples/rescan", summary="Rescan samples directory and auto-register new files")
def api_samples_rescan():
	report = helpers.rescan_samples()
	return {"ok": True, "report": report}


@router.post("/samples/link", summary="Link a sample to a lesson/vocab")
def api_samples_link(sample_id: str = Form(...), lesson_id: Optional[str] = Form(None), vocab_id: Optional[str] = Form(None)):
	"""Link an existing sample (by sample_id) to a lesson and/or vocab entry.

	- Updates `helpers.PERSISTED_STORE['samples'][sample_id]` with `lesson_id` and `vocab_id`.
	- Ensures the vocab entry exists in persisted lessons (creates minimal entry if missing) and sets its `audio_filename` to the sample filename.
	"""
	helpers.load_persisted_store()
	samples = helpers.PERSISTED_STORE.setdefault("samples", {})
	sample = samples.get(sample_id)
	if not sample:
		raise HTTPException(status_code=404, detail="sample_id not found")

	# Update lesson association if provided
	if lesson_id:
		sample["lesson_id"] = lesson_id

	# If vocab_id provided, ensure vocab exists in persisted lessons and set audio_filename
	lessons_store = helpers.PERSISTED_STORE.setdefault("lessons", {})
	target_lesson = lesson_id or sample.get("lesson_id")

	if vocab_id:
		# ensure lesson bucket exists
		if target_lesson:
			lessons_store.setdefault(target_lesson, lessons_store.get(target_lesson, []))

		found = False
		if target_lesson and target_lesson in lessons_store:
			for v in lessons_store[target_lesson]:
				if v.get("id") == vocab_id:
					v["audio_filename"] = sample.get("filename")
					found = True
					break

		# If not found in persisted lessons, try built-in VOCAB and import it (with audio)
		if not found and target_lesson and target_lesson in helpers.VOCAB:
			for v in helpers.VOCAB[target_lesson]:
				if v.get("id") == vocab_id:
					# copy into persisted store and attach audio_filename
					new_v = {**v}
					new_v["audio_filename"] = sample.get("filename")
					lessons_store[target_lesson].append(new_v)
					found = True
					break

		# If still not found, create a minimal vocab entry under target_lesson (or '0' bucket)
		if not found:
			bucket = target_lesson or "0"
			lessons_store.setdefault(bucket, lessons_store.get(bucket, []))
			new_v = {"id": vocab_id, "word": vocab_id, "meaning": "", "example": "", "audio_filename": sample.get("filename")}
			lessons_store[bucket].append(new_v)

		# Finally, set sample meta
		sample["vocab_id"] = vocab_id

	# Persist changes
	helpers.save_persisted_store()

	# Build response objects
	vocab_obj = None
	if vocab_id:
		bucket = target_lesson or "0"
		for v in helpers.PERSISTED_STORE.get("lessons", {}).get(bucket, []):
			if v.get("id") == vocab_id:
				vocab_obj = {**v, "audio_url": f"/static/samples/{v.get('audio_filename')}" if v.get('audio_filename') else None}
				break

	return {"ok": True, "sample_id": sample_id, "sample": sample, "vocab": vocab_obj}

@router.get("/samples/export-package", summary="Export samples + vocab_store.json as a zip for offline use")
def api_export_package():
	package_name = f"speech_package_{uuid.uuid4().hex[:8]}.zip"
	# ensure tmp dir exists
	os.makedirs(helpers.TMP_DIR, exist_ok=True)
	package_path = os.path.join(helpers.TMP_DIR, package_name)
	with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
		if os.path.exists(helpers.VOCAB_STORE_FILE):
			z.write(helpers.VOCAB_STORE_FILE, arcname=os.path.basename(helpers.VOCAB_STORE_FILE))
		if os.path.isdir(helpers.SAMPLES_DIR):
			for f in os.listdir(helpers.SAMPLES_DIR):
				full = os.path.join(helpers.SAMPLES_DIR, f)
				if os.path.isfile(full):
					z.write(full, arcname=os.path.join("samples", f))
	def iterfile():
		# stream in fixed-size chunks to avoid line-based iteration issues
		try:
			with open(package_path, "rb") as fp:
				while True:
					chunk = fp.read(8192)
					if not chunk:
						break
					yield chunk
		finally:
			try:
				os.remove(package_path)
			except Exception:
				pass
	# use proper Content-Disposition header
	headers = {"Content-Disposition": f'attachment; filename="{package_name}"'}
	return StreamingResponse(iterfile(), media_type="application/zip", headers=headers)

@router.post("/samples/import-package", summary="Import zip (samples + vocab_store.json). Overwrites persisted store and copies samples.")
async def api_import_package(file: UploadFile = File(...)):
	contents = await file.read()
	if not contents:
		raise HTTPException(status_code=400, detail="Empty upload")
	# ensure tmp dir exists
	os.makedirs(helpers.TMP_DIR, exist_ok=True)
	temp_zip = os.path.join(helpers.TMP_DIR, f"import_{uuid.uuid4().hex[:8]}.zip")
	with open(temp_zip, "wb") as f:
		f.write(contents)
	extract_dir = os.path.join(helpers.TMP_DIR, f"extract_{uuid.uuid4().hex[:8]}")
	os.makedirs(extract_dir, exist_ok=True)
	try:
		with zipfile.ZipFile(temp_zip, 'r') as z:
			z.extractall(extract_dir)
		maybe_vocab = os.path.join(extract_dir, os.path.basename(helpers.VOCAB_STORE_FILE))
		if os.path.exists(maybe_vocab):
			shutil.copy2(maybe_vocab, helpers.VOCAB_STORE_FILE)
		extracted_samples_dir = os.path.join(extract_dir, "samples")
		if os.path.isdir(extracted_samples_dir):
			# ensure samples dir exists before copying
			os.makedirs(helpers.SAMPLES_DIR, exist_ok=True)
			for f in os.listdir(extracted_samples_dir):
				src = os.path.join(extracted_samples_dir, f)
				dst = os.path.join(helpers.SAMPLES_DIR, f)
				shutil.copy2(src, dst)
		try:
			os.remove(temp_zip)
		except Exception:
			pass
		report = helpers.rescan_samples()
		return {"ok": True, "report": report}
	except zipfile.BadZipFile:
		raise HTTPException(status_code=400, detail="Invalid zip file")
	finally:
		try:
			shutil.rmtree(extract_dir)
		except Exception:
			pass

# -------------------------
# Persisted vocab management
# -------------------------
@router.post("/lessons/{lesson_id}/vocab", summary="Add a vocab item to persisted store and link to a sample")
def api_add_vocab(lesson_id: str, item: VocabItemIn):
	helpers.load_persisted_store()
	lessons_store = helpers.PERSISTED_STORE.setdefault("lessons", {})
	if lesson_id not in lessons_store:
		lessons_store[lesson_id] = []
	audio_filename = item.audio_filename
	if item.sample_id:
		sample_meta = helpers.PERSISTED_STORE.get("samples", {}).get(item.sample_id)
		if not sample_meta:
			raise HTTPException(status_code=404, detail="sample_id not found")
		audio_filename = sample_meta.get("filename")
	vid = item.id or f"{lesson_id}_{len(lessons_store[lesson_id]) + 1}"
	vocab_obj = {
		"id": vid,
		"word": item.word,
		"meaning": item.meaning or "",
		"example": item.example or "",
		"audio_filename": audio_filename
	}
	lessons_store[lesson_id].append(vocab_obj)
	helpers.save_persisted_store()
	vocab_with_url = {**vocab_obj, "audio_url": f"/static/samples/{audio_filename}" if audio_filename else None}
	return {"ok": True, "vocab": vocab_with_url}

@router.get("/lessons/{lesson_id}/vocab/store", summary="Get persisted vocab for lesson")
def api_get_vocab_store(lesson_id: str):
	helpers.load_persisted_store()
	lessons_store = helpers.PERSISTED_STORE.get("lessons", {})
	items = lessons_store.get(lesson_id, [])
	result = []
	for v in items:
		fn = v.get("audio_filename")
		result.append({**v, "audio_url": f"/static/samples/{fn}" if fn else None})
	return {"count": len(result), "vocab": result}

# -------------------------
# API: Pronunciation Analysis
# -------------------------
@router.post("/analyze/compare", summary="Compare user audio with reference sample")
async def api_analyze_compare(
	user_audio: Optional[UploadFile] = File(None, description="User's pronunciation audio"),
	file: Optional[UploadFile] = File(None, description="Alternative field name 'file'"),
	sample_id: Optional[str] = Form(None, description="Reference sample ID to compare against (optional)")
):
	"""
	Upload user audio and compare with a reference sample.
	Returns similarity scores for pronunciation accuracy.
	"""
	# Accept either 'user_audio' or 'file'
	upload = user_audio or file

	# Prepare user audio
	user_wav_path = None
	created_tmp = False
	try:
		if not upload:
			return _default_compare_response("No user audio provided")
		user_contents = await upload.read()
		if len(user_contents) == 0:
			return _default_compare_response("Empty user file")
		if hasattr(helpers, "MAX_UPLOAD_BYTES") and len(user_contents) > helpers.MAX_UPLOAD_BYTES:
			return _default_compare_response("File too large")

		os.makedirs(helpers.TMP_DIR, exist_ok=True)
		created_tmp = True
		user_id = str(uuid.uuid4())[:8]
		_, ext = os.path.splitext(upload.filename or "user.wav")
		user_raw_path = os.path.join(helpers.TMP_DIR, f"user_{user_id}{ext or '.wav'}")
		with open(user_raw_path, "wb") as f:
			f.write(user_contents)
		user_wav_path = os.path.join(helpers.TMP_DIR, f"user_{user_id}.wav")
		# If WAV and already 16k mono, skip conversion
		try:
			skip_conv = False
			if user_raw_path.lower().endswith('.wav'):
				try:
					with _wave.open(user_raw_path, 'rb') as wf:
						if wf.getnchannels() == 1 and wf.getframerate() == 16000:
							skip_conv = True
				except Exception:
					skip_conv = False

			if skip_conv:
				user_wav_path = user_raw_path
			else:
				helpers.convert_to_wav16_mono(user_raw_path, user_wav_path)
				if user_raw_path != user_wav_path:
					try:
						os.remove(user_raw_path)
					except Exception:
						pass
		except Exception as e:
			helpers.logger.error(f"Audio conversion failed: {e}")
			try:
				os.remove(user_raw_path)
			except Exception:
				pass
			return _default_compare_response("Audio conversion failed")

		# sample_id must be provided for explicit compare
		if not sample_id:
			try:
				if created_tmp and user_wav_path and os.path.exists(user_wav_path):
					os.remove(user_wav_path)
			except Exception:
				pass
			return _default_compare_response("No reference sample_id provided")

		# Load reference
		helpers.load_persisted_store()
		sample_meta = helpers.PERSISTED_STORE.get("samples", {}).get(sample_id)
		if not sample_meta:
			try:
				if created_tmp and user_wav_path and os.path.exists(user_wav_path):
					os.remove(user_wav_path)
			except Exception:
				pass
			return _default_compare_response(f"Reference sample '{sample_id}' not found")

		ref_filename = sample_meta.get("filename")
		ref_path = os.path.join(helpers.SAMPLES_DIR, ref_filename)
		if not os.path.exists(ref_path):
			try:
				if created_tmp and user_wav_path and os.path.exists(user_wav_path):
					os.remove(user_wav_path)
			except Exception:
				pass
			return _default_compare_response(f"Reference audio file not found: {ref_filename}")

		# Extract features and compare
		try:
			user_features = helpers.extract_features(user_wav_path)
		except Exception as e:
			helpers.logger.error(f"Feature extraction failed: {e}")
			try:
				if created_tmp and user_wav_path and os.path.exists(user_wav_path):
					os.remove(user_wav_path)
			except Exception:
				pass
			return _default_compare_response("Feature extraction failed")

		try:
			ref_features = helpers.extract_features(ref_path)
		except Exception as e:
			helpers.logger.error(f"Feature extraction (ref) failed: {e}")
			try:
				if created_tmp and user_wav_path and os.path.exists(user_wav_path):
					os.remove(user_wav_path)
			except Exception:
				pass
			return _default_compare_response("Reference feature extraction failed")

		try:
			mfcc_dist, pitch_diff, tempo_diff = helpers.compare_features_dicts(user_features, ref_features)
		except Exception as e:
			helpers.logger.error(f"Comparison failed: {e}")
			try:
				if created_tmp and user_wav_path and os.path.exists(user_wav_path):
					os.remove(user_wav_path)
			except Exception:
				pass
			return _default_compare_response("Comparison failed")

		mfcc_score = max(0, 100 - mfcc_dist * 2)
		pitch_score = max(0, 100 - pitch_diff * 0.5)
		tempo_score = max(0, 100 - tempo_diff * 0.5)
		overall_score = (mfcc_score * 0.6 + pitch_score * 0.3 + tempo_score * 0.1)

		# Cleanup
		try:
			if created_tmp and user_wav_path and os.path.exists(user_wav_path):
				os.remove(user_wav_path)
		except Exception:
			pass

		return {
			"ok": True,
			"sample_id": sample_id,
			"reference_file": ref_filename,
			"scores": {
				"overall": round(overall_score, 2),
				"mfcc": round(mfcc_score, 2),
				"pitch": round(pitch_score, 2),
				"tempo": round(tempo_score, 2)
			},
			"details": {
				"mfcc_distance": round(mfcc_dist, 4),
				"pitch_difference_hz": round(pitch_diff, 2),
				"tempo_difference_bpm": round(tempo_diff, 2)
			},
			"feedback": _get_feedback(overall_score),
			"comment": _get_review(overall_score, mfcc_dist, pitch_diff, tempo_diff)
		}

	except Exception as e:
		helpers.logger.error(f"Unexpected error in /analyze/compare: {e}")
		return _default_compare_response("Unexpected error preparing user audio")

@router.post("/analyze", summary="Analyze user audio (compare or auto)")
async def api_analyze(
	user_audio: Optional[UploadFile] = File(None, description="User's pronunciation audio"),
	file: Optional[UploadFile] = File(None, description="Alternative field name 'file'"),
	sample_id: Optional[str] = Form(None, description="Reference sample ID to compare against (optional)")
):
	"""
	Compatibility endpoint that either compares uploaded user audio against a
	reference sample (when `sample_id` provided) or automatically compares
	against all registered samples and returns the best match.
	Accepts multipart/form-data with field `user_audio` or `file`.
	"""
	# accept either 'user_audio' or 'file'
	upload = user_audio or file

	# Prepare user audio (allow fallback to default sample when missing)
	user_wav_path = None
	created_tmp = False
	try:
		if not upload:
			default_path = _get_default_sample_path()
			if not default_path:
				return _default_compare_response("No user audio provided and no default sample available")
			user_wav_path = default_path
		else:
			user_contents = await upload.read()
			if len(user_contents) == 0:
				default_path = _get_default_sample_path()
				if not default_path:
					return _default_compare_response("Empty user file and no default sample available")
				user_wav_path = default_path
			elif hasattr(helpers, "MAX_UPLOAD_BYTES") and len(user_contents) > helpers.MAX_UPLOAD_BYTES:
				return _default_compare_response("File too large")
			else:
				os.makedirs(helpers.TMP_DIR, exist_ok=True)
				created_tmp = True
				user_id = str(uuid.uuid4())[:8]
				_, ext = os.path.splitext(upload.filename or "user.wav")
				user_raw_path = os.path.join(helpers.TMP_DIR, f"user_{user_id}{ext or '.wav'}")
				with open(user_raw_path, "wb") as f:
					f.write(user_contents)
				user_wav_path = os.path.join(helpers.TMP_DIR, f"user_{user_id}.wav")
				# If WAV and already 16k mono, skip conversion
				try:
					skip_conv = False
					if user_raw_path.lower().endswith('.wav'):
						try:
							with _wave.open(user_raw_path, 'rb') as wf:
								if wf.getnchannels() == 1 and wf.getframerate() == 16000:
									skip_conv = True
						except Exception:
							skip_conv = False

					if skip_conv:
						user_wav_path = user_raw_path
					else:
						helpers.convert_to_wav16_mono(user_raw_path, user_wav_path)
						if user_raw_path != user_wav_path:
							try:
								os.remove(user_raw_path)
							except Exception:
								pass
				except Exception as e:
					helpers.logger.error(f"Audio conversion failed: {e}")
					try:
						os.remove(user_raw_path)
					except Exception:
						pass
					return _default_compare_response("Audio conversion failed")

		# If sample_id provided -> compare with that sample
		if sample_id:
			helpers.load_persisted_store()
			sample_meta = helpers.PERSISTED_STORE.get("samples", {}).get(sample_id)
			if not sample_meta:
				return _default_compare_response(f"Reference sample '{sample_id}' not found")
			ref_filename = sample_meta.get("filename")
			ref_path = os.path.join(helpers.SAMPLES_DIR, ref_filename)
			if not os.path.exists(ref_path):
				return _default_compare_response(f"Reference audio file not found: {ref_filename}")

			try:
				user_features = helpers.extract_features(user_wav_path)
				ref_features = helpers.extract_features(ref_path)
			except Exception as e:
				helpers.logger.error(f"Feature extraction failed: {e}")
				return _default_compare_response("Feature extraction failed")

			try:
				mfcc_dist, pitch_diff, tempo_diff = helpers.compare_features_dicts(user_features, ref_features)
			except Exception as e:
				helpers.logger.error(f"Comparison failed: {e}")
				return _default_compare_response("Comparison failed")

			mfcc_score = max(0, 100 - mfcc_dist * 2)
			pitch_score = max(0, 100 - pitch_diff * 0.5)
			tempo_score = max(0, 100 - tempo_diff * 0.5)
			overall_score = (mfcc_score * 0.6 + pitch_score * 0.3 + tempo_score * 0.1)

			# Cleanup
			try:
				if created_tmp and user_wav_path and os.path.exists(user_wav_path):
					os.remove(user_wav_path)
			except Exception:
				pass

			return {
				"ok": True,
				"sample_id": sample_id,
				"reference_file": ref_filename,
				"scores": {
					"overall": round(overall_score, 2),
					"mfcc": round(mfcc_score, 2),
					"pitch": round(pitch_score, 2),
					"tempo": round(tempo_score, 2)
				},
				"details": {
					"mfcc_distance": round(mfcc_dist, 4),
					"pitch_difference_hz": round(pitch_diff, 2),
					"tempo_difference_bpm": round(tempo_diff, 2)
				},
				"feedback": _get_feedback(overall_score),
				"comment": _get_review(overall_score, mfcc_dist, pitch_diff, tempo_diff)
			}

		# Otherwise, auto-detect best matching sample across all samples
		try:
			user_features = helpers.extract_features(user_wav_path)
		except Exception as e:
			helpers.logger.error(f"Feature extraction failed: {e}")
			return _default_auto_response("Feature extraction failed")

		helpers.load_persisted_store()
		samples_meta = helpers.PERSISTED_STORE.get("samples", {})
		if not samples_meta:
			try:
				if created_tmp and user_wav_path and os.path.exists(user_wav_path):
					os.remove(user_wav_path)
			except Exception:
				pass
			raise HTTPException(status_code=404, detail="No reference samples found. Please upload sample audios first.")

		# Build candidate list then compare in parallel to reduce latency
		candidates = []
		for s_id, meta in samples_meta.items():
			ref_filename = meta.get("filename")
			ref_path = os.path.join(helpers.SAMPLES_DIR, ref_filename)
			if not os.path.exists(ref_path):
				continue
			candidates.append((s_id, meta, ref_path))

		matches = []
		if candidates:
			for (s_id, meta, ref_path) in candidates:
				try:
					res = _build_match_entry(s_id, meta, ref_path, user_features)
					if res:
						matches.append(res)
				except Exception:
					# ignore individual failures
					continue

		try:
			if created_tmp and user_wav_path and os.path.exists(user_wav_path):
				os.remove(user_wav_path)
		except Exception:
			pass

		if not matches:
			raise HTTPException(status_code=404, detail="No matching samples found")

		matches.sort(key=lambda x: x["overall_score"], reverse=True)
		best_match = matches[0]

		return {
			"ok": True,
			"detected_word": best_match["word"],
			"best_match": best_match,
			"all_matches": matches[:5],
			"total_compared": len(matches),
			"feedback": _get_feedback(best_match["overall_score"]),
			"comment": _get_review(
				best_match["overall_score"],
				best_match.get("details", {}).get("mfcc_distance"),
				best_match.get("details", {}).get("pitch_difference_hz"),
				best_match.get("details", {}).get("tempo_difference_bpm")
			)
		}
	except Exception as e:
		helpers.logger.error(f"Unexpected error in /analyze: {e}")
		return _default_compare_response("Unexpected error preparing user audio")

def _get_feedback(score: float) -> str:
	"""Generate feedback message based on score"""
	if score >= 90:
		return "Xuất sắc! Phát âm rất chuẩn."
	elif score >= 75:
		return "Tốt! Phát âm khá chính xác."
	elif score >= 60:
		return "Khá! Cần luyện tập thêm một chút."
	elif score >= 40:
		return "Cần cải thiện. Hãy nghe kỹ mẫu và thử lại."
	else:
		return "Cần luyện tập nhiều hơn. Nghe lại mẫu và phát âm chậm rãi."


def _get_review(score: float, mfcc_dist: Optional[float] = None, pitch_diff: Optional[float] = None, tempo_diff: Optional[float] = None) -> str:
	"""Return a short Vietnamese comment (concise, actionable).

	Examples: "Xuất sắc — phát âm gần mẫu. Gợi ý: chú ý âm cuối." 
	"""
	# Short overall label
	if score >= 90:
		base = "Xuất sắc — phát âm rất giống mẫu."
	elif score >= 75:
		base = "Tốt — gần đúng, chỉnh nhẹ vài âm."
	elif score >= 60:
		base = "Khá — còn lỗi ở vài âm, luyện lại." 
	elif score >= 40:
		base = "Cần cải thiện — luyện phát âm và ngữ điệu." 
	else:
		base = "Cần luyện nhiều — bắt đầu từ chậm và nhại mẫu."

	parts = [base]

	# Brief component hints
	if mfcc_dist is not None:
		if mfcc_dist > 40:
			parts.append("Âm: chưa giống; luyện âm chính và âm cuối.")
		elif mfcc_dist > 15:
			parts.append("Âm: một số âm cần sửa.")

	if pitch_diff is not None:
		if pitch_diff > 20:
			parts.append("Ngữ điệu: lệch, bắt chước cao độ mẫu.")

	if tempo_diff is not None:
		if tempo_diff > 10:
			parts.append("Tốc độ: điều chỉnh để khớp nhịp mẫu.")

	# Join short pieces
	return " ".join(parts)


def _get_default_sample_path() -> Optional[str]:
	"""Return a default sample path (first available sample) or None."""
	try:
		# Prefer persisted store listing
		helpers.load_persisted_store()
		samples = helpers.PERSISTED_STORE.get("samples", {})
		if samples:
			# get first sample filename
			for meta in samples.values():
				fn = meta.get("filename")
				if fn:
					p = os.path.join(helpers.SAMPLES_DIR, fn)
					if os.path.exists(p):
						return p
		# fallback: first file on disk
		if os.path.isdir(helpers.SAMPLES_DIR):
			for fn in sorted(os.listdir(helpers.SAMPLES_DIR)):
				p = os.path.join(helpers.SAMPLES_DIR, fn)
				if os.path.isfile(p):
					return p
	except Exception:
		pass
	return None


def _default_compare_response(message: str = "No analysis available") -> dict:
	return {
		"ok": False,
		"message": message,
		"scores": {
			"overall": 0.0,
			"mfcc": 0.0,
			"pitch": 0.0,
			"tempo": 0.0
		},
		"details": {}
	}


def _default_auto_response(message: str = "No analysis available") -> dict:
	return {
		"ok": False,
		"message": message,
		"detected_word": None,
		"best_match": None,
		"all_matches": []
	}

@router.post("/analyze/auto", summary="Auto-detect and analyze user pronunciation")
async def api_analyze_auto(
	user_audio: Optional[UploadFile] = File(None, description="User's pronunciation audio"),
	file: Optional[UploadFile] = File(None, description="Alternative field name 'file'"),
	lesson_id: Optional[str] = Form(None, description="Optional: filter by lesson ID")
):
	"""
	Upload audio and automatically detect which word the user is pronouncing.
	Compares with ALL samples and returns best match with similarity score.
	"""
	# Prepare user audio: accept missing/wrong field by falling back to default sample
	user_wav_path = None
	created_tmp = False
	try:
		upload = user_audio or file
		if not upload:
			default_path = _get_default_sample_path()
			if not default_path:
				return _default_auto_response("No user audio provided and no default sample available")
			user_wav_path = default_path
		else:
			user_contents = await upload.read()
			if len(user_contents) == 0:
				default_path = _get_default_sample_path()
				if not default_path:
					return _default_auto_response("Empty user file and no default sample available")
				user_wav_path = default_path
			elif len(user_contents) > helpers.MAX_UPLOAD_BYTES:
				return _default_auto_response("File too large")
			else:
				os.makedirs(helpers.TMP_DIR, exist_ok=True)
				created_tmp = True
				user_id = str(uuid.uuid4())[:8]
				_, ext = os.path.splitext(upload.filename or "audio.wav")
				user_raw_path = os.path.join(helpers.TMP_DIR, f"user_{user_id}{ext or '.wav'}")
				with open(user_raw_path, "wb") as f:
					f.write(user_contents)
				user_wav_path = os.path.join(helpers.TMP_DIR, f"user_{user_id}.wav")
				# If WAV and already 16k mono, skip conversion
				try:
					skip_conv = False
					if user_raw_path.lower().endswith('.wav'):
						try:
							with _wave.open(user_raw_path, 'rb') as wf:
								if wf.getnchannels() == 1 and wf.getframerate() == 16000:
									skip_conv = True
						except Exception:
							skip_conv = False

					if skip_conv:
						user_wav_path = user_raw_path
					else:
						helpers.convert_to_wav16_mono(user_raw_path, user_wav_path)
						if user_raw_path != user_wav_path:
							try:
								os.remove(user_raw_path)
							except Exception:
								pass
				except Exception as e:
					helpers.logger.error(f"Audio conversion failed: {e}")
					try:
						os.remove(user_raw_path)
					except Exception:
						pass
					return _default_auto_response("Audio conversion failed")

		try:
			user_features = helpers.extract_features(user_wav_path)
		except Exception as e:
			helpers.logger.error(f"Feature extraction failed: {e}")
			return _default_auto_response("Feature extraction failed")
	except Exception as e:
		helpers.logger.error(f"Unexpected error preparing user audio: {e}")
		return _default_auto_response("Unexpected error preparing user audio")
	
	# Load all samples and compare
	helpers.load_persisted_store()
	samples_meta = helpers.PERSISTED_STORE.get("samples", {})
	
	if not samples_meta:
		try:
			os.remove(user_wav_path)
		except Exception:
			pass
		raise HTTPException(status_code=404, detail="No reference samples found. Please upload sample audios first.")
	
	# Compare with all samples sequentially
	matches = []
	for sample_id, meta in samples_meta.items():
		# Filter by lesson_id if provided
		if lesson_id and meta.get("lesson_id") != lesson_id:
			continue

		ref_filename = meta.get("filename")
		if not ref_filename:
			continue
		ref_path = os.path.join(helpers.SAMPLES_DIR, ref_filename)

		if not os.path.exists(ref_path):
			continue

		try:
			res = _build_match_entry(sample_id, meta, ref_path, user_features)
			if res:
				matches.append(res)
		except Exception as e:
			helpers.logger.warning(f"Skipped sample {sample_id}: {e}")
			continue
	
	# Cleanup user audio if it was a created temp file
	try:
		if created_tmp and user_wav_path and os.path.exists(user_wav_path):
			os.remove(user_wav_path)
	except Exception:
		pass
	
	if not matches:
		raise HTTPException(status_code=404, detail="No matching samples found")
	
	# Sort by overall_score descending
	matches.sort(key=lambda x: x["overall_score"], reverse=True)
	best_match = matches[0]
	
	return {
		"ok": True,
		"detected_word": best_match["word"],
		"best_match": best_match,
		"all_matches": matches[:5],  # Top 5 matches
		"total_compared": len(matches),
		"feedback": _get_feedback(best_match["overall_score"]),
		"comment": _get_review(
			best_match["overall_score"],
			best_match.get("details", {}).get("mfcc_distance"),
			best_match.get("details", {}).get("pitch_difference_hz"),
			best_match.get("details", {}).get("tempo_difference_bpm")
		)
	}


@router.post("/analyze/vocab", summary="Analyze user audio against a specific vocab's samples")
async def api_analyze_vocab(
	user_audio: Optional[UploadFile] = File(None, description="User's pronunciation audio"),
	file: Optional[UploadFile] = File(None, description="Alternative field name 'file'"),
	vocab_id: str = Form(..., description="Vocab ID to restrict comparison to"),
	lesson_id: Optional[str] = Form(None, description="Optional: lesson ID to further filter samples")
):
	"""
	Compare uploaded user audio only against samples that are linked to `vocab_id`.
	Returns best match and scores for that vocab's sample set.
	"""
	upload = user_audio or file
	user_wav_path = None
	created_tmp = False
	try:
		if not upload:
			return _default_compare_response("No user audio provided")

		user_contents = await upload.read()
		if len(user_contents) == 0:
			return _default_compare_response("Empty user file")
		if hasattr(helpers, "MAX_UPLOAD_BYTES") and len(user_contents) > helpers.MAX_UPLOAD_BYTES:
			return _default_compare_response("File too large")

		os.makedirs(helpers.TMP_DIR, exist_ok=True)
		created_tmp = True
		user_id = str(uuid.uuid4())[:8]
		_, ext = os.path.splitext(upload.filename or "user.wav")
		user_raw_path = os.path.join(helpers.TMP_DIR, f"user_{user_id}{ext or '.wav'}")
		with open(user_raw_path, "wb") as f:
			f.write(user_contents)
		user_wav_path = os.path.join(helpers.TMP_DIR, f"user_{user_id}.wav")

		# If WAV and already 16k mono, skip conversion
		try:
			skip_conv = False
			if user_raw_path.lower().endswith('.wav'):
				try:
					with _wave.open(user_raw_path, 'rb') as wf:
						if wf.getnchannels() == 1 and wf.getframerate() == 16000:
							skip_conv = True
				except Exception:
					skip_conv = False

			if skip_conv:
				user_wav_path = user_raw_path
			else:
				helpers.convert_to_wav16_mono(user_raw_path, user_wav_path)
				if user_raw_path != user_wav_path:
					try:
						os.remove(user_raw_path)
					except Exception:
						pass
		except Exception as e:
			helpers.logger.error(f"Audio conversion failed: {e}")
			try:
				os.remove(user_raw_path)
			except Exception:
				pass
			return _default_compare_response("Audio conversion failed")

		# Load samples and filter by vocab_id (and optional lesson_id)
		helpers.load_persisted_store()
		samples_meta = helpers.PERSISTED_STORE.get("samples", {})
		filtered = []
		for s_id, meta in samples_meta.items():
			if meta.get("vocab_id") != vocab_id:
				continue
			if lesson_id and meta.get("lesson_id") != lesson_id:
				continue
			fn = meta.get("filename")
			if not fn:
				continue
			ref_path = os.path.join(helpers.SAMPLES_DIR, fn)
			if os.path.exists(ref_path):
				filtered.append((s_id, meta, ref_path))

		if not filtered:
			try:
				if created_tmp and user_wav_path and os.path.exists(user_wav_path):
					os.remove(user_wav_path)
			except Exception:
				pass
			raise HTTPException(status_code=404, detail=f"No samples found for vocab_id={vocab_id}")

		# Extract user features
		try:
			user_features = helpers.extract_features(user_wav_path)
		except Exception as e:
			helpers.logger.error(f"Feature extraction failed: {e}")
			try:
				if created_tmp and user_wav_path and os.path.exists(user_wav_path):
					os.remove(user_wav_path)
			except Exception:
				pass
			return _default_compare_response("Feature extraction failed")

		# Compare against filtered set sequentially
		matches = []
		if filtered:
			for (s_id, meta, ref_path) in filtered:
				try:
					res = _build_match_entry(s_id, meta, ref_path, user_features)
					if res:
						matches.append(res)
				except Exception:
					continue

		try:
			if created_tmp and user_wav_path and os.path.exists(user_wav_path):
				os.remove(user_wav_path)
		except Exception:
			pass

		if not matches:
			raise HTTPException(status_code=404, detail="No matching samples found for this vocab")

		matches.sort(key=lambda x: x["overall_score"], reverse=True)
		best = matches[0]

		return {
			"ok": True,
			"vocab_id": vocab_id,
			"detected_word": best["word"],
			"best_match": best,
			"all_matches": matches[:5],
			"total_compared": len(matches),
			"feedback": _get_feedback(best["overall_score"]),
			"comment": _get_review(
				best["overall_score"],
				best.get("details", {}).get("mfcc_distance"),
				best.get("details", {}).get("pitch_difference_hz"),
				best.get("details", {}).get("tempo_difference_bpm")
			) 
		}

	except Exception as e:
		helpers.logger.error(f"Unexpected error in /analyze/vocab: {e}")
		return _default_compare_response("Unexpected error preparing user audio for vocab")


def _get_vocab_info(sample_id: str, meta: dict) -> dict:
	"""Get vocabulary information for a sample"""
	vocab_id = meta.get("vocab_id")
	lesson_id = meta.get("lesson_id")
	
	# Try to find in persisted lessons
	if lesson_id and vocab_id:
		lessons_store = helpers.PERSISTED_STORE.get("lessons", {})
		if lesson_id in lessons_store:
			for vocab in lessons_store[lesson_id]:
				if vocab.get("id") == vocab_id:
					return vocab
	
	# Try to find in built-in VOCAB
	if lesson_id and lesson_id in helpers.VOCAB:
		for vocab in helpers.VOCAB[lesson_id]:
			if vocab.get("id") == vocab_id:
				return vocab
	
	# Fallback: try to extract from filename
	filename = meta.get("filename", "")
	m = re.match(r'^\d+[-_](.+)\.wav$', filename)
	if m:
		word_part = m.group(1)
		word = re.sub(r'[_\-]+', ' ', word_part).strip().capitalize()
		return {"word": word, "meaning": ""}
	
	return {"word": meta.get("original_name", filename), "meaning": ""}

# -------------------------
# API: Secure audio streaming with Range support
# -------------------------
@router.get("/audio/list", summary="List all available audio files")
def api_list_audio():
	"""
	Get list of all audio files in samples directory.
	Returns filename, size, and MIME type for each file.
	"""
	files = []
	samples_path = Path(helpers.SAMPLES_DIR)
	
	if not samples_path.exists():
		return {"files": [], "count": 0}
	
	for p in samples_path.iterdir():
		if p.is_file():
			_, ext = os.path.splitext(p.name)
			if ext.lower() in helpers.ALLOWED_EXTS:
				files.append({
					"name": p.name,
					"size": p.stat().st_size,
					"mime": get_mime_type(p),
					"stream_url": f"/api/audio/stream/{p.name}",
					"download_url": f"/api/audio/download/{p.name}"
				})
	
	return {"files": files, "count": len(files)}

@router.get("/audio/stream/{filename}", summary="Stream audio file with Range support")
async def api_stream_audio(request: Request, filename: str):
	"""
	Stream audio file with full Range request support for seeking.
	Supports ETag caching and partial content (206).
	"""
	file_path = safe_resolve_filename(filename)
	
	if not file_path.exists() or not file_path.is_file():
		raise HTTPException(status_code=404, detail="File not found")
	
	# Validate file type
	_, ext = os.path.splitext(filename)
	if ext.lower() not in helpers.ALLOWED_EXTS:
		raise HTTPException(status_code=400, detail="File type not allowed")
	
	file_size = file_path.stat().st_size
	content_type = get_mime_type(file_path)
	
	# Generate ETag
	etag = f'W/"{file_path.stat().st_mtime_ns:x}-{file_path.stat().st_size:x}"'
	
	# Check If-None-Match for 304
	if_none_match = request.headers.get("if-none-match")
	if if_none_match and if_none_match == etag:
		return JSONResponse(status_code=304, content=None)
	
	range_header = request.headers.get("range")
	
	# No Range header → return full file
	if not range_header:
		headers = {
			"ETag": etag,
			"Accept-Ranges": "bytes",
			"Content-Length": str(file_size),
			"Cache-Control": "public, max-age=3600",
			"Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length, ETag"
		}
		return FileResponse(
			path=str(file_path),
			media_type=content_type,
			filename=file_path.name,
			headers=headers
		)
	
	# Parse Range header
	try:
		range_value = range_header.strip().lower()
		assert range_value.startswith("bytes=")
		r = range_value.split("=", 1)[1]
		start_s, end_s = r.split("-", 1)
		start = int(start_s) if start_s else 0
		end = int(end_s) if end_s else file_size - 1
	except Exception as e:
		helpers.logger.warning(f"Invalid Range header: {range_header}, error: {e}")
		headers = {
			"ETag": etag,
			"Accept-Ranges": "bytes",
			"Content-Length": str(file_size)
		}
		return FileResponse(
			path=str(file_path),
			media_type=content_type,
			filename=file_path.name,
			headers=headers
		)
	
	# Validate range
	if start >= file_size:
		raise HTTPException(
			status_code=416,
			detail="Requested Range Not Satisfiable",
			headers={"Content-Range": f"bytes */{file_size}"}
		)
	
	end = min(end, file_size - 1)
	length = end - start + 1
	
	# Stream generator
	def iter_file(path: Path, start_pos: int, content_length: int):
		with open(path, "rb") as f:
			f.seek(start_pos)
			remaining = content_length
			while remaining > 0:
				chunk_size = min(CHUNK_SIZE, remaining)
				chunk = f.read(chunk_size)
				if not chunk:
					break
				remaining -= len(chunk)
				yield chunk
	
	headers = {
		"Content-Range": f"bytes {start}-{end}/{file_size}",
		"Accept-Ranges": "bytes",
		"Content-Length": str(length),
		"ETag": etag,
		"Cache-Control": "public, max-age=3600",
		"Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length, ETag"
	}
	
	return StreamingResponse(
		iter_file(file_path, start, length),
		status_code=206,
		media_type=content_type,
		headers=headers
	)

@router.head("/audio/stream/{filename}", summary="Get audio metadata (HEAD)")
async def api_stream_audio_head(filename: str):
	"""HEAD request for audio metadata without downloading."""
	file_path = safe_resolve_filename(filename)
	
	if not file_path.exists() or not file_path.is_file():
		raise HTTPException(status_code=404, detail="File not found")
	
	_, ext = os.path.splitext(filename)
	if ext.lower() not in helpers.ALLOWED_EXTS:
		raise HTTPException(status_code=400, detail="File type not allowed")
	
	file_size = file_path.stat().st_size
	content_type = get_mime_type(file_path)
	etag = f'W/"{file_path.stat().st_mtime_ns:x}-{file_path.stat().st_size:x}"'
	
	headers = {
		"Content-Type": content_type,
		"Content-Length": str(file_size),
		"Accept-Ranges": "bytes",
		"ETag": etag,
		"Cache-Control": "public, max-age=3600"
	}
	
	return Response(status_code=200, headers=headers)

@router.get("/audio/download/{filename}", summary="Download audio file (force download)")
async def api_download_audio(filename: str):
	"""Force download (triggers browser download dialog)."""
	file_path = safe_resolve_filename(filename)
	
	if not file_path.exists() or not file_path.is_file():
		raise HTTPException(status_code=404, detail="File not found")
	
	_, ext = os.path.splitext(filename)
	if ext.lower() not in helpers.ALLOWED_EXTS:
		raise HTTPException(status_code=400, detail="File type not allowed")
	
	content_type = get_mime_type(file_path)
	
	return FileResponse(
		path=str(file_path),
		media_type=content_type,
		filename=filename,
		headers={
			"Content-Disposition": f'attachment; filename="{filename}"',
			"Cache-Control": "public, max-age=3600"
		}
	)

# -------------------------
# API: Data sync for mobile app
# -------------------------
@router.get("/sync/lessons", summary="Get all lessons with vocab for offline sync")
def api_sync_lessons():
	"""Get complete lesson data with audio URLs for mobile app sync."""
	helpers.load_persisted_store()
	
	lessons_data = []
	for lesson in helpers.LESSONS:
		lesson_id = lesson["id"]
		vocab_list = helpers.merged_vocab_for_lesson(lesson_id)
		
		# Update URLs to use streaming endpoint
		for vocab in vocab_list:
			if vocab.get("audio_filename"):
				vocab["audio_url"] = f"/api/audio/stream/{vocab['audio_filename']}"
				vocab["download_url"] = f"/api/audio/download/{vocab['audio_filename']}"
		
		progress = helpers.PERSISTED_STORE.get("progress", {}).get(lesson_id, lesson.get("progress", 0))
		
		lessons_data.append({
			"id": lesson["id"],
			"title": lesson["title"],
			"description": lesson["description"],
			"progress": progress,
			"vocab_count": len(vocab_list),
			"vocab": vocab_list
		})
	
	return {
		"ok": True,
		"total_lessons": len(lessons_data),
		"lessons": lessons_data,
		"sync_timestamp": helpers._get_timestamp()
	}

@router.get("/sync/lesson/{lesson_id}", summary="Get single lesson data")
def api_sync_single_lesson(lesson_id: str):
	"""Get complete data for a single lesson."""
	lesson = None
	for l in helpers.LESSONS:
		if l["id"] == lesson_id:
			lesson = l
			break
	
	if not lesson:
		raise HTTPException(status_code=404, detail="Lesson not found")
	
	helpers.load_persisted_store()
	vocab_list = helpers.merged_vocab_for_lesson(lesson_id)
	
	# Update URLs
	for vocab in vocab_list:
		if vocab.get("audio_filename"):
			vocab["audio_url"] = f"/api/audio/stream/{vocab['audio_filename']}"
			vocab["download_url"] = f"/api/audio/download/{vocab['audio_filename']}"
	
	progress = helpers.PERSISTED_STORE.get("progress", {}).get(lesson_id, lesson.get("progress", 0))
	
	return {
		"ok": True,
		"lesson": {
			"id": lesson["id"],
			"title": lesson["title"],
			"description": lesson["description"],
			"progress": progress,
			"vocab_count": len(vocab_list),
			"vocab": vocab_list
		},
		"sync_timestamp": helpers._get_timestamp()
	}

@router.post("/sync/download-batch", summary="Download multiple audio files as ZIP")
async def api_sync_download_batch(filenames: list[str] = Form(...)):
	"""Download multiple audio files in a single ZIP."""
	if not filenames or len(filenames) == 0:
		raise HTTPException(status_code=400, detail="No filenames provided")
	
	if len(filenames) > 100:
		raise HTTPException(status_code=400, detail="Maximum 100 files per batch")
	
	os.makedirs(helpers.TMP_DIR, exist_ok=True)
	zip_name = f"batch_{uuid.uuid4().hex[:8]}.zip"
	zip_path = os.path.join(helpers.TMP_DIR, zip_name)
	
	with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
		for filename in filenames:
			if ".." in filename or "/" in filename or "\\" in filename:
				continue
			
			file_path = os.path.join(helpers.SAMPLES_DIR, filename)
			if os.path.exists(file_path) and os.path.isfile(file_path):
				z.write(file_path, arcname=filename)
	
	def iterfile():
		try:
			with open(zip_path, "rb") as fp:
				while True:
					chunk = fp.read(8192)
					if not chunk:
						break
					yield chunk
		finally:
			try:
				os.remove(zip_path)
			except Exception:
				pass
	
	headers = {"Content-Disposition": f'attachment; filename="{zip_name}"'}
	return StreamingResponse(iterfile(), media_type="application/zip", headers=headers)

@router.get("/sync/check-updates", summary="Check for server updates")
def api_sync_check_updates(last_sync: Optional[str] = None):
	"""Check if server has updates since last sync timestamp."""
	helpers.load_persisted_store()
	
	try:
		vocab_mtime = os.path.getmtime(helpers.VOCAB_STORE_FILE)
		server_timestamp = helpers._timestamp_to_str(vocab_mtime)
	except Exception:
		server_timestamp = helpers._get_timestamp()
	
	has_updates = True
	if last_sync:
		has_updates = server_timestamp > last_sync
	
	return {
		"ok": True,
		"has_updates": has_updates,
		"server_timestamp": server_timestamp,
		"total_lessons": len(helpers.LESSONS),
		"total_samples": len(helpers.PERSISTED_STORE.get("samples", {})),
		"total_vocab": sum(len(helpers.merged_vocab_for_lesson(l["id"])) for l in helpers.LESSONS)
	}

@router.get("/sync/manifest", summary="Get complete data manifest")
def api_sync_manifest():
	"""Get manifest of all available data for app sync planning."""
	helpers.load_persisted_store()
	
	lessons_manifest = []
	for lesson in helpers.LESSONS:
		lesson_id = lesson["id"]
		vocab_list = helpers.merged_vocab_for_lesson(lesson_id)
		
		audio_files = []
		for vocab in vocab_list:
			if vocab.get("audio_filename"):
				audio_files.append({
					"filename": vocab["audio_filename"],
					"stream_url": f"/api/audio/stream/{vocab['audio_filename']}",
					"download_url": f"/api/audio/download/{vocab['audio_filename']}",
					"word": vocab.get("word")
				})
		
		lessons_manifest.append({
			"lesson_id": lesson_id,
			"title": lesson["title"],
			"vocab_count": len(vocab_list),
			"audio_files": audio_files
		})
	
	all_audio = []
	if os.path.exists(helpers.SAMPLES_DIR):
		for fn in os.listdir(helpers.SAMPLES_DIR):
			if os.path.isfile(os.path.join(helpers.SAMPLES_DIR, fn)):
				_, ext = os.path.splitext(fn)
				if ext.lower() in helpers.ALLOWED_EXTS:
					all_audio.append({
						"filename": fn,
						"stream_url": f"/api/audio/stream/{fn}",
						"download_url": f"/api/audio/download/{fn}"
					})
	
	return {
		"ok": True,
		"lessons": lessons_manifest,
		"total_audio_files": len(all_audio),
		"audio_files": all_audio,
		"server_timestamp": helpers._get_timestamp()
	}
