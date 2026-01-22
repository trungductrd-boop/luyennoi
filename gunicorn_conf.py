workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
threads = 1
timeout = 30
max_requests = 150
max_requests_jitter = 50
preload_app = False
# Keep logging minimal to reduce overhead
loglevel = "info"
