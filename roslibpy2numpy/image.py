import base64
import sys
import numpy as np
import roslibpy
import cv2

name_to_dtypes = {
    "rgb8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "rgb16": (np.uint16, 3),
    "rgba16": (np.uint16, 4),
    "bgr8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "bgr16": (np.uint16, 3),
    "bgra16": (np.uint16, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),

    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8": (np.uint8, 1),
    "bayer_bggr8": (np.uint8, 1),
    "bayer_gbrg8": (np.uint8, 1),
    "bayer_grbg8": (np.uint8, 1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),

    # OpenCV CvMat types
    "8UC1": (np.uint8, 1),
    "8UC2": (np.uint8, 2),
    "8UC3": (np.uint8, 3),
    "8UC4": (np.uint8, 4),
    "8SC1": (np.int8, 1),
    "8SC2": (np.int8, 2),
    "8SC3": (np.int8, 3),
    "8SC4": (np.int8, 4),
    "16UC1": (np.uint16, 1),
    "16UC2": (np.uint16, 2),
    "16UC3": (np.uint16, 3),
    "16UC4": (np.uint16, 4),
    "16SC1": (np.int16, 1),
    "16SC2": (np.int16, 2),
    "16SC3": (np.int16, 3),
    "16SC4": (np.int16, 4),
    "32SC1": (np.int32, 1),
    "32SC2": (np.int32, 2),
    "32SC3": (np.int32, 3),
    "32SC4": (np.int32, 4),
    "32FC1": (np.float32, 1),
    "32FC2": (np.float32, 2),
    "32FC3": (np.float32, 3),
    "32FC4": (np.float32, 4),
    "64FC1": (np.float64, 1),
    "64FC2": (np.float64, 2),
    "64FC3": (np.float64, 3),
    "64FC4": (np.float64, 4)
}


# noinspection PyArgumentList
def raw_image_to_numpy(msg):
    if not msg['encoding'] in name_to_dtypes:
        raise TypeError('Unrecognized encoding {}'.format(msg.encoding))

    dtype_class, channels = name_to_dtypes[msg['encoding']]
    dtype = np.dtype(dtype_class)
    dtype = dtype.newbyteorder('>' if msg['is_bigendian'] else '<')
    shape = (msg['height'], msg['width'], channels)

    base64_bytes = msg['data'].encode('ascii')
    image_bytes = base64.b64decode(base64_bytes)
    # # Convert to a NumPy array
    data = np.frombuffer(image_bytes, dtype=dtype).reshape(shape)
    if msg['encoding'] == 'rgb8':
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    if channels == 1:
        data = data[..., 0]
    return data


def numpy_to_image_raw(arr, encoding="bgr8", frame_id='camera_frame'):
    if encoding not in name_to_dtypes:
        raise TypeError('Unrecognized encoding {}'.format(encoding))

    # extract width, height, and channels
    dtype_class, exp_channels = name_to_dtypes[encoding]
    if len(arr.shape) == 2:
        height, width, channels = arr.shape + (1,)
    elif len(arr.shape) == 3:
        height, width, channels = arr.shape
    else:
        raise TypeError("Array must be two or three dimensional")

    # check type and channels
    if exp_channels != channels:
        raise TypeError("Array has {} channels, {} requires {}".format(
            channels, encoding, exp_channels
        ))
    if dtype_class != arr.dtype.type:
        raise TypeError("Array is {}, {} requires {}".format(
            arr.dtype.type, encoding, dtype_class
        ))

    # make the array contiguous in memory, as mostly required by the format
    contig = np.ascontiguousarray(arr)
    data = contig.tostring()
    step = contig.strides[0]

    im = roslibpy.Message({
        'header': {
            'frame_id': frame_id
        },
        'height': height,
        'width': width,
        'encoding': encoding,
        # 'is_bigendian': is_bigendian,
        'step': step,
        'data': base64.b64encode(data).decode('ascii')
    })

    return im


def compressed_image_to_numpy(img):
    base64_bytes = img['data'].encode('ascii')
    image_bytes = base64.b64decode(base64_bytes)
    # Convert the image to a numpy array
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image
    img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_np


def numpy_to_compressed_image(arr, frame_id='camera_frame', encoding='jpeg'):
    if encoding not in ['jpeg', 'png']:
        raise TypeError('Unrecognized encoding {}'.format(encoding))
    encoded = base64.b64encode(arr).decode('ascii')
    return dict(header=dict(frame_id=frame_id), format=encoding, data=encoded)
