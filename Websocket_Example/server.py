# import asyncio
# import websockets
# import time

# async def hello(websocket, path):
#     print("Client connected. Sending messages now.")
#     while True:
#         try:
#             await websocket.send("Hello world!")
#             time.sleep(1)
#         except websockets.exceptions.ConnectionClosed:
#             print("Client disconnected.")
#             break

# start_server = websockets.serve(hello, "0.0.0.0", 8765)

# try:
#     asyncio.get_event_loop().run_until_complete(start_server)
#     asyncio.get_event_loop().run_forever()
# except KeyboardInterrupt:
#     pass
# finally:
#     asyncio.get_event_loop().run_until_complete(asyncio.gather(*asyncio.all_tasks()))
#     asyncio.get_event_loop().stop()

import threading
import time
import asyncio
import websockets
from queue import Queue

# shared variable and queue
counter = 0
counter_queue = Queue()

async def hello(websocket, path):
    global counter
    print("Client connected. Sending messages now.")
    while True:
        try:
            # wait for a new item to be added to the queue
            counter = counter_queue.get()
            # send the updated counter value
            await websocket.send(str(counter))
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected.")
            break

# function that updates the variable
def increment():
    global counter
    while True:
        counter += 1
        # add the current counter value to the queue
        counter_queue.put(counter)
        # sleep for 1 second before incrementing again
        time.sleep(1)

# create and start the threads
thread1 = threading.Thread(target=increment)
thread1.start()

# start the websocket server in the main thread
start_server = websockets.serve(hello, "0.0.0.0", 8765)

try:
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
except KeyboardInterrupt:
    pass
finally:
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*asyncio.all_tasks()))
    asyncio.get_event_loop().stop()
