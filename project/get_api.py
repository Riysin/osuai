import websockets
import json
import asyncio

async def connect_websocket(uri):
    try:
        return await websockets.connect(uri)
    except websockets.exceptions.ConnectionClosedError:
        return await asyncio.sleep(5)
    except Exception as e:
        print(f"Error: {e}")

async def get_data(websocket):
    data = await websocket.recv()
    jdata = json.loads(data)
    return jdata

async def game_state(websocket):
    data = await get_data(websocket)
    return data['menu']['state']

async def hit_score(websocket, score):
    data = await get_data(websocket)
    return data['gameplay']['hits'][str(score)]