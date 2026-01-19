#!/usr/bin/env python3
"""Upload a file to /api/upload and poll /status/{job_id} with exponential backoff.

Usage: python tools/upload_and_poll.py <file_path> [--url http://127.0.0.1:8000] [--api-key KEY]
"""
import os
import sys
import time
import argparse
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import helpers


def upload_file(path: str, url: str, api_key: str):
    files = {"file": (helpers.sanitize_filename(os.path.basename(path)) or os.path.basename(path), open(path, "rb"))}
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # `url` is expected to be the full upload endpoint (e.g. http://host:port/compare or http://host:port/api/upload)
    r = requests.post(url, files=files, headers=headers)
    try:
        data = r.json()
    except Exception:
        r.raise_for_status()
    return r.status_code, data


def poll_status(job_id: str, url: str, api_key: str, max_attempts: int = 30, initial_interval: float = 1.0):
    attempt = 0
    interval = initial_interval
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    while attempt < max_attempts:
        attempt += 1
        try:
            r = requests.get(f"{url.rstrip('/')}/api/status/{job_id}", headers=headers, timeout=10)
            data = r.json()
        except Exception as e:
            print(f"Poll attempt {attempt} failed: {e}")
            data = None
        print(f"Poll {attempt}: {data}")
        if data and isinstance(data, dict):
            status = data.get("status")
            if status and status != "processing" and status != "queued":
                return data
            # Some handlers return full job payload under 'result' when done
            if status == "done" or data.get("result"):
                return data
        time.sleep(interval)
        # exponential backoff with cap
        interval = min(interval * 2, 10.0)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to audio file to upload")
    parser.add_argument("--url", default=os.environ.get("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("UPLOAD_API_KEY", ""))
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--use-local-on-fail", action="store_true", help="Offer to run local conversion/analysis on failure")
    parser.add_argument("--flask-demo", action="store_true", help="Target Flask demo server endpoints (/compare and /status/<job_id>)")
    parser.add_argument("--upload-endpoint", default=None, help="Override upload endpoint path (e.g. /compare or /api/upload)")
    parser.add_argument("--status-endpoint", default=None, help="Override status endpoint template (use {job_id}) (e.g. /status/{job_id} or /api/status/{job_id})")
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        print("File not found:", path)
        sys.exit(2)

    print("Uploading", path)
    # determine endpoints
    if args.upload_endpoint:
        upload_path = args.upload_endpoint
    else:
        upload_path = "/compare" if args.flask_demo else "/api/upload"

    if args.status_endpoint:
        status_template = args.status_endpoint
    else:
        status_template = "/status/{job_id}" if args.flask_demo else "/api/status/{job_id}"

    # upload
    status, data = upload_file(path, args.url.rstrip('/') + upload_path, args.api_key)
    print("Upload response:", status, data)
    if not data or "job_id" not in data:
        print("Upload did not return job_id. Response:", data)
        sys.exit(1)
    job_id = data["job_id"]
    print("Job ID:", job_id)

    print("Start polling status... (attempts=", args.attempts, ")")
    # poll using the chosen status endpoint template
    def poll_using_template(jid):
        url_base = args.url.rstrip('/')
        status_url = url_base + status_template.format(job_id=jid)
        attempt = 0
        interval = args.interval
        headers = {}
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"
        while attempt < args.attempts:
            attempt += 1
            try:
                r = requests.get(status_url, headers=headers, timeout=10)
                data = r.json()
            except Exception as e:
                print(f"Poll attempt {attempt} failed: {e}")
                data = None
            print(f"Poll {attempt}: {data}")
            if data and isinstance(data, dict):
                status = data.get("status")
                if status and status != "processing" and status != "queued":
                    return data
                if status == "done" or data.get("result"):
                    return data
            time.sleep(interval)
            interval = min(interval * 2, 10.0)
        return None

    res = poll_using_template(job_id)
    if res:
        print("Final status:", res)
        return

    print("Polling exhausted without a definitive result.")
    if args.use_local_on_fail:
        ans = input("Server seems unresponsive. Run local conversion test (y/N)? ")
        if ans.strip().lower().startswith('y'):
            print("Running local conversion test via run_convert_test.py")
            os.system(f"python run_convert_test.py")
        else:
            print("Skipped local action.")


if __name__ == '__main__':
    main()
