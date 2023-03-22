import time
from array import array as Array
import numpy as np
import roslibpy


def occupancygrid_to_numpy(msg):
    data = \
        np.asarray(msg.data,
                   dtype=np.int8).reshape(msg.info.height, msg.info.width)

    return np.ma.array(data, mask=data == -1, fill_value=-1)


def numpy_to_occupancy_grid(arr, info=None, frame_id='map'):
    if not len(arr.shape) == 2:
        raise TypeError('Array must be 2D')
    if not arr.dtype == np.int8:
        raise TypeError('Array must be of int8s')

    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.data

    data = Array('b', arr.ravel().astype(np.int8))
    if info is None:
        info = roslibpy.Message({
            'width': arr.shape[1],
            'height': arr.shape[0],
            'resolution': 0.05,
            'origin': {
                'position': {
                    'x': 0,
                    'y': 0,
                    'z': 0
                },
                'orientation': {
                    'x': 0,
                    'y': 0,
                    'z': 0,
                    'w': 1
                }
            }
        })
    return roslibpy.Message({
        'header': {
            'frame_id': frame_id,
            'stamp': time.time()
        },
        'info': info,
        'data': data
    })
