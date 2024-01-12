import asyncio
import time
from threading import Thread
import websockets
from data_get import *

data_list = [0,0,0,0,0]

class Upload(Thread):
    def run(self):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.values())
        loop.close()
    
    async def values(self):
        global data_list
        uri = "ws://localhost:24050/ws"
        websocket = await connect_websocket(uri)
        while True:
            try:
                data = await get_data(websocket)
                data_list[0] = game_state(data)
                data_list[1] = hit_score(data, 300)
                data_list[2] = hit_score(data, 100)
                data_list[3] = hit_score(data, 50)
                data_list[4] = hit_score(data, 0)
                await asyncio.sleep(0.1)

            except websockets.exceptions.ConnectionClosedError:
                print('Reconnecting...')
                await asyncio.sleep(0.1)
                websocket = await connect_websocket(uri)

def thread_start():
    t = Upload()
    t.daemon = True
    t.start()  
    
    
# def test():
#     while True:
#         time.sleep(0.1)
#         print(data_bridge)

# if __name__ == "__main__":
#     thread_start()
#     test()