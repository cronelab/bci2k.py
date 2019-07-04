
# %%
import numpy as np
from functools import reduce
import struct


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


# def parse(byte_array):
#     d = DataView(byte_array)
#     return {
#         "headerSize": d.get_uint_8(0),
#         "numverOfPlanes": d.get_uint_16(1),
#         "width": d.get_uint_16(3),
#         "hieght": d.get_uint_16(5),
#         "offset": d.get_uint_16(7),
#     }

# result = parse([8, 96, 0, 0, 2, 0, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# import json
# print(json.dumps(result, indent=2))

# d = DataView([111, 62, 163, 36],8)
# d.get_uint_8(2)


# %%
#from dataview import DataView


# do i need to put these functions into a class
def _decodeMessage(data):
    dv = DataView(data)  # (data, 0, len(data))
    descriptor = dv.get_uint_8(0)  # start index is 0

    if descriptor == 3:
        _decodeStateFormat(dv)
    elif descriptor == 4:
        supplement = dv.get_uint_8(0)
        if supplement == 1:
            _decodeGenericSignal(dv)
        elif supplement == 3:
            _decodeSignalProperties(dv)
        else:
            raise ValueError('Unsupported supplement' + str(supplement))
        # onReceiveBlock()
    elif descriptor == 5:
        _decodeStateVector(dv)
    else:
        raise ValueError('Unsupported descriptor' + str(descriptor))


# def _decodeStateFormat(data):

def _decodeGenericSignal(dv):
    signalType = dv.get_uint_8(0)  # signaltype is descriptor
    nChannels = getLengthField(dv, 2)
    nElements = getLengthField(dv, 2)
    signal = []
    for ch in range(nChannels):
        for el in range(nElements):
            if signalType == 0:
                signal.append(dv.get_uint_16(0))
            elif signalType == 2:
                signal.append(dv.get_float_32(0))
            elif signalType == 3:
                signal.append(dv.get_unit_32(0))
            else:
                raise ValueError('Unsupported signal type')
     # what kind of output should be here


# class signalProperties:
#     def _init_(self,name,channels):
#         self.name = name
#         self.channels = channels


def _decodeSignalProperties(dv):
    propstr = getNullTermString(dv)

    # replace???

    signalProperties = []
    prop_tokens = propstr.split()  # white space seperated
    props = []
    for i in range(len(prop_token)):
        if prop_tokensp[i] == '':
            continue
        props.append(prop_tokens[i])

    pidx = 0
    signalProperties_name = props  # ???

    signalProperties_channels = []
    if props[pidx] == '{':
        pidx_new = pidx
        while props[pidx_new] != '}':
            signalProperties_channels.append(props[pidx_new])
            pidx_new += 1
    else:
        numChannels = int(props)  # ???
        for i in range(numChannels):
            signalProperties_channels.append(str(i+1))

    signalProperties.elements = []
    if props[pidx] == '{':
        pidx_new = pidx
        while props[pidx_new] != '}':
            signalProperties.elements.append(props[pidx_new])
            pidx_new += 1
    else:
        numElements = int(props)
        for i in range(numElements):
            signalProperties.element.append(str(i+1))


def getNullTermString(dv, val=''):
    while dv._offset < dv.byteLength:  # ????
        v = dv.get_uint_8()
        if (v == 0):
            break
        val = str(v)
    return val


def getLengthField(dv, n, length=0, extended=False):
    if n == 1:
        length = dv.get_uint_8(0)
        extended = length == 0xff
    elif n == 2:
        length = dv.get_uint_16(0)
        extended = length = 0xffff
    elif n == 4:
        length = dv.get_uint_32(0)  # ???
        extended = length = 0xffffffff
    else:
        raise ValueError('Unsupported getlengthfield')
    return


def _decodeSignalProperties(data):
    x = 5
# def _decodeStateVector(data):

# def onReceiveBlock():
