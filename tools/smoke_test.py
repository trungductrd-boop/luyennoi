import importlib, json, sys
from pathlib import Path

# Ensure project root is on sys.path so `import main` works when running
# this script from the `tools/` directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
importlib.invalidate_caches()
try:
    m = importlib.import_module('main')
    from fastapi.testclient import TestClient
    client = TestClient(m.app)
    print('ROOT', client.get('/').status_code)
    print('HEALTH', json.dumps(client.get('/health').json(), ensure_ascii=False))
except Exception as e:
    import traceback
    traceback.print_exc()
    raise
