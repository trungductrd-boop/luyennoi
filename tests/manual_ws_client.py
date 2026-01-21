import asyncio
import websockets

async def run():
    uri = "ws://127.0.0.1:8000/ws/status/testjob123"
    print("Connecting to", uri)
    try:
        async with websockets.connect(uri) as ws:
            print("Connected, waiting for message (10s timeout)")
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
                print("RECV:", msg)
            except asyncio.TimeoutError:
                print("TIMEOUT waiting for message")
    except Exception as e:
        print("WS connection failed:", e)

if __name__ == '__main__':
    asyncio.run(run())
