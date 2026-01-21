import asyncio
import main

class DummyWS:
    def __init__(self):
        self.received = []

    async def send_json(self, data):
        self.received.append(data)


async def run_test():
    # ensure MAIN_LOOP is set (not required for direct notify)
    main.MAIN_LOOP = asyncio.get_event_loop()
    job_id = "testjob123"
    dummy = DummyWS()
    main.job_ws_clients.setdefault(job_id, []).append(dummy)
    payload = {"status": "done", "result": {"ok": True}}

    # call the coroutine directly
    await main.notify_ws(job_id, payload)

    assert dummy.received and dummy.received[0] == payload, f"expected payload, got {dummy.received}"
    print("TEST-PASSED")


if __name__ == "__main__":
    asyncio.run(run_test())
