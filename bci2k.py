###################################################################################
# A BCI2k connector
# Dummy consumer, recommend multiprocessing.queue for 
# shared memory b/w producer and consumer (websocket _app.py modification needed)
###################################################################################

import multiprocessing
from functools import reduce
import struct
import websocket
import numpy as np

class DataView:
    def __init__(self, array, bytes_per_element=1):
        """
        bytes_per_element is the size of each element in bytes.
        By default we are assuming the array is one byte per element.
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

def on_message(ws, data):
    dv = DataView(data)
    freq_count = 13
    if data[0] == 4:  # Visualization and Brain Signal Data Format
        if data[1] == 1:  # Signal
            if(data[2] == 2):  # float32
                # number of channels
                channel_count = data[3] + data[4]*256
                print("Number of channels: {}".format(channel_count))
                # number of elements
                element_count = data[5] + data[6]*256
                print("Number of elements: {}".format(element_count))

                datastream = []
                for ii in range(channel_count*element_count):
                    datastream.append(dv.get_float_32(7+ii)) 

                datastream = np.swapaxes(np.swapaxes(np.array(datastream).reshape(
                    freq_count, int(channel_count / freq_count), element_count), 0, 1), 0, 2) # time by freq by element    
                
                #print(datastream)
               
def on_error(ws, error):
    print('#####ERROR#####')
    print(error)

def on_close(ws):
    print("#####CLOSED#####")

def recv_BCI2k(): # producer
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp('ws://localhost:20203', #20100
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    ws.run_forever()

def audi_BCI2k():  # delay*16000   
    pass

if __name__ == "__main__":
    p_audi = multiprocessing.Process(target = audi_BCI2k) # consumer
    p_audi.start()
    p_recv = multiprocessing.Process(target = recv_BCI2k) # producer
    p_recv.start()

