import asyncio
import time
import websockets
from get_api import *

async def main():
    uri = "ws://localhost:24050/ws"
    websocket = await connect_websocket(uri)
    while True:
        try:
            data = await get_data(websocket)
            if game_state(data) == 2:
                hit300 = hit_score(data, 300)
                hit100 = hit_score(data, 100)
                hit50 = hit_score(data, 50)
                hit0 = hit_score(data, 0)
                print('wowwi')
                print(f"300:{hit300}, 100:{hit100}, 50:{hit50}, miss:{hit0}")
            else:
                print('test')
            time.sleep(0.1)
        except websockets.exceptions.ConnectionClosedError:
            print('Reconnecting in 1...')
            time.sleep(1)
            websocket = await connect_websocket(uri)       

if __name__ == "__main__":
    asyncio.run(main())
