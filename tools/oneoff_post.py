import requests, time, os
from pathlib import Path

url_base = 'http://127.0.0.1:5000'
file_path = 'data/samples/xin_chao.wav'
if not os.path.exists(file_path):
    print('file missing', file_path)
    raise SystemExit(1)

files = {'file': (os.path.basename(file_path), open(file_path,'rb'))}
print('uploading...')
r = requests.post(f'{url_base}/compare', files=files, timeout=10)
print('upload status', r.status_code, r.text)
if r.status_code != 202:
    raise SystemExit(1)
job_id = r.json().get('job_id')
print('job_id', job_id)
# poll
for i in range(20):
    time.sleep(1)
    try:
        rr = requests.get(f'{url_base}/status/{job_id}', timeout=5)
        print('poll', i+1, rr.status_code, rr.text)
        data = rr.json()
        if data.get('status') and data.get('status') not in ('processing','queued'):
            print('final', data)
            break
        if data.get('status') == 'done' or data.get('result'):
            print('done result', data)
            break
    except Exception as e:
        print('poll error', e)
print('done')
