import asyncio
import websockets


from functools import reduce
import struct

import numpy as np

import preprocess


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
        bytes = [integer.to_bytes(
            self.bytes_per_element, byteorder='little', signed=signed) for integer in integers]
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
        return struct.unpack('<f', binary)[0]  # <f for little endian


async def BCI2K_OperatorConnection():
    async with websockets.connect('ws://localhost:80') as websocket:
        websocket.send('Get Parameter SystemState')
        systemState = await websocket.recv()
        print(systemState)


async def BCI2K_DataConnection(endpoint):
    if endpoint == 'Source':
        address = '20100'
    elif endpoint == 'Processing':
        address = '20203'
    async with websockets.connect('ws://localhost:{}'.format(address)) as websocket:
        stateFormat = await websocket.recv()
        # print(stateFormat.decode("utf-8"))
        print("")
        signalProperties = await websocket.recv()
        # print(signalProperties.decode("utf-8"))
        print("")
        data = await websocket.recv()
        dv = DataView(data)
        run_count = 1
        if data[0] == 4:  # Visualization and Brain Signal Data Format
            if data[1] == 1:  # Signal
                if(data[2] == 2):  # float32
                    # number of channels
                    channel_count = data[3]
                    print("Number of channels: {}".format(channel_count))
                    # print(data[4])  # + 256 channels
                    # number of elements
                    element_count = data[5]
                    print("Number of elements: {}".format(element_count))
                    # print(data[6])  # + 256 elements
                    
                    datastream_buffer = []
                    if not run_count:
                        datastream_buffer = datastream # most recent batch
                        run_count = 0
                    datastream = []
                    for ii in range(channel_count*element_count):
                        datastream.append(dv.get_float_32(7+ii))

                    datastream = np.array(datastream).reshape(
                        channel_count, element_count)                    
                    
                    # TODO: feed it to model
                    print('datastream')
                    print(datastream)
                    np.save('datastream.npy', datastream)
                    
                    if datastream_buffer == []:
                        processed = preprocess.preprocess(datastream.T, channel_count)
                    else:
                        processed = preprocess.preprocess(
                            np.concatenate((datastream_buffer,datastream),axis=1).T, channel_count)
                    print('processed')
                    print(processed)


while True:
    asyncio.get_event_loop().run_until_complete(BCI2K_OperatorConnection())
    asyncio.get_event_loop().run_until_complete(BCI2K_DataConnection('Source'))
    # asyncio.get_event_loop().run_until_complete(BCI2K_DataConnection('Processing'))
