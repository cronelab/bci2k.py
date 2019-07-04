import asyncio
import websockets


async def dataConnector():
    async with websockets.connect('ws://localhost:20100') as websocket:
        stateFormat = await websocket.recv()
        # print(stateFormat.decode("utf-8"))
        print("")
        signalProperties = await websocket.recv()
        # print(signalProperties.decode("utf-8"))
        print("")
        data = await websocket.recv()
        if data[0] == 4:  # Visualization and Brain Signal Data Format
            if data[1] == 1:  # Signal
                if(data[2] == 2):  # float32
                    # number of channels
                    print("number of channels: ")
                    print(data[3])
                    # print(data[4])  # + 256 channels
                    # number of elements
                    print("number of elements: ")
                    print(data[5])
                    # print(data[6])  # + 256 elements


while True:
    asyncio.get_event_loop().run_until_complete(dataConnector())
