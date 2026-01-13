import requests, sys
fpath = r"C:/fastapi_server/data/samples/xin_chao.mp3"
with open(fpath,'rb') as f:
    r = requests.post('http://127.0.0.1:8000/api/analyze', files={'file':('xin_chao.mp3',f,'audio/mpeg')}, data={'mode':'vocab'})
    print('STATUS', r.status_code)
    print(r.text)
