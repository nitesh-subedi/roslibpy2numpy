import transformations as transformations
import numpy as np
import roslibpy


def vector3_to_numpy(msg, hom=False):
    if hom:
        return np.array([msg['x'], msg['y'], msg['z'], 0])
    else:
        return np.array([msg['x'], msg['y'], msg['z']])


def numpy_to_vector3(arr):
    if arr.dtype != np.float64:
        raise ValueError("Expected a floating point array")

    if arr.shape != (3, 1):
        raise ValueError("Expected a 3-element array")
    msg = roslibpy.Message({
        'x': arr[0],
        'y': arr[1],
        'z': arr[2]
    })
    return msg


def point_to_numpy(msg, hom=False):
    if hom:
        return np.array([msg['x'], msg['y'], msg['z'], 1])
    else:
        return np.array([msg['x'], msg['y'], msg['z']]).reshape(3, 1)


def numpy_to_point(arr):
    if arr.dtype != np.float64:
        raise ValueError("Expected a floating point array")

    if arr.shape != (3, 1):
        raise ValueError("Expected a 3x1 dimensional array")
    msg = roslibpy.Message({
        'x': arr[0],
        'y': arr[1],
        'z': arr[2]
    })
    return msg


def quat_to_numpy(msg):
    return np.array([msg['x'], msg['y'], msg['z'], msg['w']])


def numpy_to_quat(arr):
    if arr.dtype != np.float64:
        raise ValueError("Expected a floating point array")

    if arr.shape != (4, 1):
        raise ValueError("Expected a 4-element array")
    msg = roslibpy.Message({
        'x': arr[0],
        'y': arr[1],
        'z': arr[2],
        'w': arr[3]
    })
    return msg


def transform_to_numpy(msg):
    trans = np.array(transformations.translation_matrix(
        [msg['translation']['x'], msg['translation']['y'], msg['translation']['z']]))
    rot = np.array(transformations.quaternion_matrix(
        [msg['rotation']['x'], msg['rotation']['y'], msg['rotation']['z'], msg['rotation']['w']]))
    return np.dot(trans, rot)


def numpy_to_transform(arr):
    if arr.dtype != np.float64:
        raise ValueError("Expected a floating point array")
    if arr.shape != (4, 4):
        raise ValueError("Expected a 4x4 array")
    # Convert the 4x4 matrix to a translation and rotation
    x = transformations.translation_from_matrix(arr)
    q = transformations.quaternion_from_matrix(arr)
    msg = roslibpy.Message({
        'translation': {
            'x': x[0],
            'y': x[1],
            'z': x[2]
        },
        'rotation': {
            'x': q[0],
            'y': q[1],
            'z': q[2],
            'w': q[3]
        }
    })
    return msg


def pose_to_numpy(msg):
    return np.dot(
        transformations.translation_matrix([msg['position']['x'], msg['position']['y'], msg['position']['z']]),
        transformations.quaternion_matrix(
            [msg['orientation']['x'], msg['orientation']['y'], msg['orientation']['z'], msg['orientation']['w']])
    )


def numpy_to_pose(arr):
    if arr.dtype != np.float64:
        raise ValueError("Expected a floating point array")
    if arr.shape != (4, 4):
        raise ValueError("Expected a 4x4 array")
    # Convert the 4x4 matrix to a translation and rotation
    x = transformations.translation_from_matrix(arr)
    q = transformations.quaternion_from_matrix(arr)
    msg = roslibpy.Message({
        'position': {
            'x': x[0],
            'y': x[1],
            'z': x[2]
        },
        'orientation': {
            'x': q[0],
            'y': q[1],
            'z': q[2],
            'w': q[3]
        }
    })
    return msg
