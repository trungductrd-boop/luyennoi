from flask import Flask, request, jsonify
import uuid
import time
import threading
import os
from concurrent.futures import ThreadPoolExecutor
import logging
from . import helpers
from flask import current_app

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Small thread pool to simulate background processing
executor = ThreadPoolExecutor(max_workers=2)


@app.route('/compare', methods=['POST'])
def compare():
    # Accept multipart form upload under 'file'
    f = request.files.get('file')
    orig_name = None
    if f:
        orig_name = helpers.sanitize_filename(f.filename) or f.filename
        # Save uploaded file to server/uploads
        save_path = os.path.join(helpers.UPLOADS_DIR, f"{uuid.uuid4().hex}_{orig_name}")
        try:
            f.save(save_path)
        except Exception:
            # fallback simple write
            with open(save_path, 'wb') as out:
                out.write(f.read())
    else:
        orig_name = request.form.get('filename') or 'no_name'
        orig_name = helpers.sanitize_filename(orig_name) or orig_name

    # Validate/resolve vocab_id early to ensure correct asset mapping
    found, resolved_vid, asset_path, err = helpers.resolve_vocab_id_from_request(request, allow_fallback=False)
    raw_vid = None
    try:
        if request.form:
            raw_vid = request.form.get('vocab_id')
    except Exception:
        raw_vid = None
    # Log incoming fields for debugging
    logger.info("Incoming compare request: vocab_id_raw=%r filename=%r", raw_vid, orig_name)

    if not found:
        # Fail-fast: unknown vocab_id
        return jsonify({"success": False, "error": "unknown vocab_id", "vocab_id": (str(raw_vid) if raw_vid is not None else None)}), 400

    job_id = uuid.uuid4().hex
    # initial job payload (processing) written to server/jobs/<job_id>.json
    payload = {"status": "processing", "created_at": time.time(), "original_filename": orig_name, "vocab_id": resolved_vid, "asset": asset_path}
    helpers.save_job_result(job_id, payload)

    # simulate async work
    def worker(jid, path, orig):
        try:
            # simulate variable processing time
            time.sleep(2 + (os.getpid() % 3))
            # write final result
            result = {"status": "done", "result": {"score": 0.87}, "original_filename": orig}
            helpers.save_job_result(jid, result)
            logger.info(f"Job {jid} done")
        except Exception as e:
            helpers.save_job_result(jid, {"status": "error", "error": str(e)})

    executor.submit(worker, job_id, None, orig_name)

    return jsonify({"status": "accepted", "job_id": job_id}), 202


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    # Load job file if exists; if exists return its status; if missing return not_found
    j = helpers.load_job_result(job_id)
    if j is None:
        # job file not found
        return jsonify({"status": "not_found"})
    # If job exists, always return its payload (may be processing/done/error)
    return jsonify(j)


if __name__ == '__main__':
    # Ensure jobs/uploads dirs
    os.makedirs(helpers.JOBS_DIR, exist_ok=True)
    os.makedirs(helpers.UPLOADS_DIR, exist_ok=True)
    # Run without the reloader/debugger when used in automated runs
    app.run(host='0.0.0.0', port=5000, debug=False)
