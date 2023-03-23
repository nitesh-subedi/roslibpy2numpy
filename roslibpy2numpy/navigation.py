import time
from array import array as Array
import numpy as np
import roslibpy


def odometry_to_numpy(msg):
    return dict(position=(np.array([
        msg['pose']['pose']['position']['x'],
        msg['pose']['pose']['position']['y'],
        msg['pose']['pose']['position']['z'],
        msg['pose']['pose']['orientation']['x'],
        msg['pose']['pose']['orientation']['y'],
        msg['pose']['pose']['orientation']['z'],
        msg['pose']['pose']['orientation']['w'],
        msg['twist']['twist']['linear']['x'],
        msg['twist']['twist']['linear']['y'],
        msg['twist']['twist']['linear']['z'],
        msg['twist']['twist']['angular']['x'],
        msg['twist']['twist']['angular']['y'],
        msg['twist']['twist']['angular']['z']
    ])), velocity=(np.array([
        msg['twist']['twist']['linear']['x'],
        msg['twist']['twist']['linear']['y'],
        msg['twist']['twist']['linear']['z'],
        msg['twist']['twist']['angular']['x'],
        msg['twist']['twist']['angular']['y'],
        msg['twist']['twist']['angular']['z']
    ])))


def numpy_to_odometry(msg, frame_id="odom", child_frame_id="base_footprint"):
    """
    Convert a numpy array to a ROS Odometry message. The array must be of shape (13,) and must be in the following order:
    x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz
    :param msg:
    :param frame_id:
    :param child_frame_id:
    :return:
    """
    return roslibpy.Message({
        'header': {
            'frame_id': frame_id,
            'stamp': time.time()
        },
        'child_frame_id': child_frame_id,
        'pose': {
            'pose': {
                'position': {
                    'x': msg['position'][0],
                    'y': msg['position'][1],
                    'z': msg['position'][2]
                },
                'orientation': {
                    'x': msg['position'][3],
                    'y': msg['position'][4],
                    'z': msg['position'][5],
                    'w': msg['position'][6]
                }
            }
        },
        'twist': {
            'twist': {
                'linear': {
                    'x': msg['velocity'][0],
                    'y': msg['velocity'][1],
                    'z': msg['velocity'][2]
                },
                'angular': {
                    'x': msg['velocity'][3],
                    'y': msg['velocity'][4],
                    'z': msg['velocity'][5]
                }
            }
        }
    })


def path_to_numpy(msg):
    """
    Convert a ROS Path message to a numpy array. The array will be of shape (n, 3) where n is the number of poses in the
    path. The first column will be the x position, the second column will be the y position, and the third column will be
    the orientation.

    The poses are assumed to be in the same frame as the path. Map origin must be subtracted from the x and y positions
    and then should be divided by the resolution to get the map coordinates.
    :param msg:
    :return: numpy array of shape (n, 3)
    """
    path = [[pose['pose']['position']['x'], pose['pose']['position']['y']] for pose in msg['poses']]
    return np.array(path)


def occupancygrid_to_numpy(msg):
    """
    Convert a ROS OccupancyGrid message to a numpy array. The array will be of shape (height, width) and will be of type
    np.int8. The values will be in the range [-1, 100] where -1 is unknown, 0 is free, and 100 is occupied. The array
    will be masked where the value is -1.
    :param msg:
    :return:
    """
    data = np.asarray(msg['data'], dtype=np.int8).reshape(msg['info']['height'], msg['info']['width'])
    return np.ma.array(data, mask=data == -1, fill_value=-1)


def numpy_to_occupancy_grid(arr, info=None, frame_id='map'):
    """
    Convert a numpy array to a ROS OccupancyGrid message.
    :param arr:
    :param info:
    :param frame_id:
    :return:
    """
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
