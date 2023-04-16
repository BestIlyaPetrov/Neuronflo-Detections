import asyncio
import websockets
import time

async def hello(websocket, path):
    print("Client connected. Sending messages now.")
    while True:
        try:
            await websocket.send("Hello world!")
            await asyncio.sleep(1)
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected.")
            break

start_server = websockets.serve(hello, "0.0.0.0", 8765)

try:
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
except KeyboardInterrupt:
    pass
finally:
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*asyncio.all_tasks()))
    asyncio.get_event_loop().stop()
