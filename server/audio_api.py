from flask import Flask, request, jsonify
import uuid
import time
import threading
import os
from concurrent.futures import ThreadPoolExecutor
import logging
from . import helpers

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Small thread pool to simulate background processing
executor = ThreadPoolExecutor(max_workers=4)


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

    job_id = uuid.uuid4().hex
    # initial job payload (processing) written to server/jobs/<job_id>.json
    payload = {"status": "processing", "created_at": time.time(), "original_filename": orig_name}
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
