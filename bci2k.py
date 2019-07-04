import asyncio
import websockets


async def dataConnector():
    async with websockets.connect('ws://localhost:20100') as websocket:
        # await websocket.send(f"")
        data = await websocket.recv()
        print(data)
        data1 = await websocket.recv()
        print(data1)
        data2 = await websocket.recv()
        print(data2)

while True:
    asyncio.get_event_loop().run_until_complete(dataConnector())
