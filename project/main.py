import asyncio
import time
import websockets
from get_api import connect_websocket, game_state, hit_score

async def main():
    uri = "ws://localhost:24050/ws"
    websocket = await connect_websocket(uri)
    while True:
        try:
            if await game_state(websocket) == 2:
                hit300 = await hit_score(websocket, 300)
                hit100 = await hit_score(websocket, 100)
                hit50 = await hit_score(websocket, 50)
                hit0 = await hit_score(websocket, 0)
                print('wowwi')
                print(f"300:{hit300}, 100:{hit100}, 50:{hit50}, miss:{hit0}")
            else:
                print('test')
            time.sleep(0.1)
        except websockets.exceptions.ConnectionClosedError:
            print('ctimeout')
            await time.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())
