import asyncio
import websockets


from functools import reduce
import struct

import numpy as np

class DataView:
    def __init__(self, array, bytes_per_element=1):
        """
        bytes_per_element is the size of each element in bytes.
        By default we are assume the array is one byte per element.
        """
        self.array = array
        self.bytes_per_element = 1

    def __get_binary(self, start_index, byte_count, signed=False):
        integers = [self.array[start_index + x] for x in range(byte_count)]
        bytes = [integer.to_bytes(self.bytes_per_element, byteorder='little', signed=signed) for integer in integers]
        return reduce(lambda a, b: a + b, bytes)

    def get_uint_16(self, start_index):
        bytes_to_read = 2
        return int.from_bytes(self.__get_binary(start_index, bytes_to_read), byteorder='little')

    def get_uint_32(self, start_index):
        bytes_to_read = 4
        return int.from_bytes(self.__get_binary(start_index, bytes_to_read), byteorder='little')
    
    def get_uint_8(self, start_index):
        bytes_to_read = 1
        return int.from_bytes(self.__get_binary(start_index, bytes_to_read), byteorder='little')

    def get_float_32(self, start_index):
        bytes_to_read = 4
        binary = self.__get_binary(start_index, bytes_to_read)
        return struct.unpack('<f', binary)[0] # <f for little endian



async def dataConnector():
    async with websockets.connect('ws://localhost:20100') as websocket:
        stateFormat = await websocket.recv()
        # print(stateFormat.decode("utf-8"))
        print("")
        signalProperties = await websocket.recv()
        # print(signalProperties.decode("utf-8"))
        print("")
        data = await websocket.recv()
        dv = DataView(data)
        if data[0] == 4:  # Visualization and Brain Signal Data Format
            if data[1] == 1:  # Signal
                if(data[2] == 2):  # float32
                    # number of channels
                    print("number of channels: ")
                    print(data[3])
                    channel_count = data[3]
                    # print(data[4])  # + 256 channels
                    # number of elements
              
                    print("number of elements: ")
                    print(data[5])
                    element_count = data[5]
                    # print(data[6])  # + 256 elements
                    
                    datastream = []
                    for ii in range(channel_count*element_count):
                        datastream.append(dv.get_float_32(7+ii))    ## TODO: write this function directly in get_float_32?
                    
                    datastream = np.array(datastream).reshape(channel_count, element_count)  
                    ### TODO: integrate datastream into preprocessing and feed it to model
                    print('datastream')
                    print(datastream)
              
                    
                    
while True:
    asyncio.get_event_loop().run_until_complete(dataConnector())
