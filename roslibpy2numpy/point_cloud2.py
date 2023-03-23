import sys
import array
import numpy as np
import roslibpy
import time

# prefix to the names of dummy fields we add to get byte alignment
# correct. this needs to not clash with any actual field names
DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
# noinspection PyArgumentList
type_mappings = [(1, np.dtype('int8')),
                 (2, np.dtype('uint8')),
                 (3, np.dtype('int16')),
                 (4, np.dtype('uint16')),
                 (5, np.dtype('int32')),
                 (6, np.dtype('uint32')),
                 (7, np.dtype('float32')),
                 (8, np.dtype('float64'))]

pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)


class PointField:
    def __init__(self):
        self.name = None
        self.offset = None
        self.datatype = None
        self.count = None

    def __repr__(self):
        return 'PointField(%s, %s, %s, %s)' % (self.name, self.offset, self.datatype, self.count)


# noinspection PyArgumentList
def fields_to_dtype(fields, point_step):
    """Convert a list of PointFields to a numpy record datatype.
    """
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(
                ('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_to_nptype[f.datatype].itemsize * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list


def dtype_to_fields(dtype):
    """Convert a numpy record datatype into a list of PointFields.
    """
    fields = []
    for field_name in dtype.names:
        np_field_type, field_offset = dtype.fields[field_name]
        # Create a PointField for this field

        pf = PointField()
        pf.name = field_name
        if np_field_type.subdtype:
            item_dtype, shape = np_field_type.subdtype
            pf.count = int(np.prod(shape))
            np_field_type = item_dtype
        else:
            pf.count = 1

        pf.datatype = nptype_to_pftype[np_field_type]
        pf.offset = field_offset
        fields.append(pf)
    return fields


def pointcloud2_to_array(cloud_msg, squeeze=True):
    """ Converts a roslib PointCloud2 message to a numpy recordarray

    Reshapes the returned array to have shape (height, width), even if the
    height is 1.

    The reason for using np.frombuffer rather than struct.unpack is
    speed... especially for large point clouds, this will be <much> faster.
    """
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (
                fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


def array_to_pointcloud2(cloud_arr, frame_id='base_link'):
    """Converts a numpy record array to a sensor_msgs.msg.PointCloud2.
    """
    # make it 2d (even if height will be 1)
    cloud_arr = np.atleast_2d(cloud_arr)
    height = cloud_arr.shape[0]
    width = cloud_arr.shape[1]
    fields = dtype_to_fields(cloud_arr.dtype)
    is_bigendian = sys.byteorder != 'little'
    point_step = cloud_arr.dtype.itemsize
    row_step = point_step * cloud_arr.shape[1]
    is_dense = \
        all([np.isfinite(
            cloud_arr[fname]).all() for fname in cloud_arr.dtype.names])

    # The PointCloud2.data setter will create an array.array object for you if you don't
    # provide it one directly. This causes very slow performance because it iterates
    # over each byte in python.
    # Here we create an array.array object using a memoryview, limiting copying and
    # increasing performance.
    memory_view = memoryview(cloud_arr)
    if memory_view.nbytes > 0:
        array_bytes = memory_view.cast("B")
    else:
        # Casting raises a TypeError if the array has no elements
        array_bytes = b""
    as_array = array.array("B")
    as_array.frombytes(array_bytes)
    data = as_array
    cloud_msg = roslibpy.Message({
        'header': {
            'stamp': time.time(),
            'frame_id': frame_id
        },
        'height': height,
        'width': width,
        'fields': fields,
        'is_bigendian': is_bigendian,
        'point_step': point_step,
        'row_step': row_step,
        'data': data,
        'is_dense': is_dense
    })

    return cloud_msg


def merge_rgb_fields(cloud_arr):
    """Takes an array with named np.uint8 fields 'r', 'g', and 'b', and returns
       an array in which they have been merged into a single np.float32 'rgb'
       field. The first byte of this field is the 'r' uint8, the second is the
       'g', uint8, and the third is the 'b' uint8.

       This is the way that pcl likes to handle RGB colors for some reason.
    """
    r = np.asarray(cloud_arr['r'], dtype=np.uint32)
    g = np.asarray(cloud_arr['g'], dtype=np.uint32)
    b = np.asarray(cloud_arr['b'], dtype=np.uint32)
    rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)

    # not sure if there is a better way to do this. i'm changing the type of
    # the array from uint32 to float32, but i don't want any conversion to take
    # place -jdb
    rgb_arr.dtype = np.float32

    # create a new array, without r, g, and b, but with rgb float32 field
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if field_name not in ('r', 'g', 'b'):
            new_dtype.append((field_name, field_type))
    new_dtype.append(('rgb', np.float32))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == 'rgb':
            new_cloud_arr[field_name] = rgb_arr
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]

    return new_cloud_arr


def split_rgb_field(cloud_arr):
    """Takes an array with a named 'rgb' float32 field, and returns an array in
    which this has been split into 3 uint 8 fields: 'r', 'g', and 'b'.

    (pcl stores rgb in packed 32 bit floats)
    """
    rgb_arr = cloud_arr['rgb'].copy()
    rgb_arr.dtype = np.uint32
    r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb_arr & 255, dtype=np.uint8)

    # create a new array, without rgb, but with r, g, and b fields
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if not field_name == 'rgb':
            new_dtype.append((field_name, field_type))
    new_dtype.append(('r', np.uint8))
    new_dtype.append(('g', np.uint8))
    new_dtype.append(('b', np.uint8))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == 'r':
            new_cloud_arr[field_name] = r
        elif field_name == 'g':
            new_cloud_arr[field_name] = g
        elif field_name == 'b':
            new_cloud_arr[field_name] = b
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]
    return new_cloud_arr


def get_xyz_points(cloud_array, remove_nans=True, dtype=float):
    """Pulls out x, y, and z columns from the cloud recordarray, and returns
    a 3xN matrix.
    """
    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & \
               np.isfinite(cloud_array['y']) & \
               np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']

    return points


def pointcloud2_to_xyz_array(cloud_msg, remove_nans=True):
    return get_xyz_points(
        pointcloud2_to_array(cloud_msg), remove_nans=remove_nans)
