import websockets
import json

async def connect_websocket(uri):
    return await websockets.connect(uri, timeout = 0.1)

async def get_data(websocket):
    data = await websocket.recv()
    jdata = json.loads(data)
    return jdata

def game_state(data):
    return data['menu']['state']

def hit_score(data, score):
    return data['gameplay']['hits'][str(score)]