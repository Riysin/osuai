import asyncio
from threading import Thread
import websockets
from data_get import *

status = {
    'state' : 0,
    '300' : 0,
    '100' : 0,
    '50' : 0,
    'miss' : 0
}

class Upload(Thread):
    def run(self):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.values())
    
    async def values(self):
        uri = "ws://localhost:24050/ws"
        websocket = await connect_websocket(uri)
        while True:
            try:
                data = await get_data(websocket)
                status['state'] = game_state(data)
                status['300'] = hit_score(data, 300)
                status['100'] = hit_score(data, 100)
                status['50'] = hit_score(data, 50)
                status['miss'] = hit_score(data, 0)

            except websockets.exceptions.ConnectionClosedError:
                print('Reconnecting...')
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