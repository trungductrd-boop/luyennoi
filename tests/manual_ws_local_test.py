from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

def run():
    job_id = "localtestjob"
    with client.websocket_connect(f"/ws/status/{job_id}") as ws:
        # send a small ping to ensure connection stays open
        ws.send_text("ping")
        # trigger notify via test endpoint
        resp = client.post(f"/__test_notify/{job_id}", json={"hello": "world"})
        assert resp.status_code == 200, resp.text
        data = ws.receive_json()
        print("RECV:", data)

if __name__ == '__main__':
    run()
