#!/usr/bin/env python3
"""Simple Redis-backed queue worker.

Listens on Redis list `job_queue` (BRPOP) and executes supported tasks.
Currently supports task 'compare' which calls `main.process_compare_job`.

Run as: `python -u worker/redis_worker.py` or set up as a separate Render background service.
"""
import os
import time
import json
import traceback

try:
    import audio_api
    redis_client = getattr(audio_api, 'redis_client', None)
except Exception:
    redis_client = None

if not redis_client:
    print('No redis client available. Set REDIS_URL or REDIS_HOST/PORT in env.')
    raise SystemExit(1)

# Import main lazily to avoid app startup side-effects if any
try:
    import main
except Exception:
    print('Failed to import main:', traceback.format_exc())
    raise

print('Redis worker started, listening on job_queue')

while True:
    try:
        # BRPOP returns (queue_name, payload) or None on timeout
        item = redis_client.brpop('job_queue', timeout=5)
        if not item:
            continue
        # item is a tuple (queue, data)
        try:
            payload = item[1]
            if isinstance(payload, bytes):
                payload = payload.decode('utf-8')
            job = json.loads(payload)
        except Exception:
            print('Invalid job payload, skipping:', payload)
            continue

        task = job.get('task')
        if task == 'compare':
            sample_path = job.get('sample_path')
            user_temp_path = job.get('user_temp_path')
            sample_uploaded_temp = bool(job.get('sample_uploaded_temp'))
            sample_id = job.get('sample_id')
            job_id = job.get('job_id')
            try:
                print(f'Processing compare job {job_id}...')
                # Call the same function used by background task
                main.process_compare_job(sample_path, user_temp_path, sample_uploaded_temp, sample_id, job_id)
                print(f'Job {job_id} processed')
            except Exception:
                print('Error processing job', job_id, traceback.format_exc())
        else:
            print('Unknown task type:', task)
    except Exception:
        print('Worker loop exception:', traceback.format_exc())
        time.sleep(1)
