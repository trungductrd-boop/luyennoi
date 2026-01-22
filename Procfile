web: gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT -w 1 --threads 1 --timeout 30 --max-requests 150 --max-requests-jitter 50
