import transformations as transformations
import numpy as np


class Transform:
    def __init__(self, translation, rotation):
        self.translation = translation
        self.rotation = rotation

    def __repr__(self):
        return "Transform(translation={}, rotation={})".format(self.translation, self.rotation)


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return "Vector3(x={}, y={}, z={})".format(self.x, self.y, self.z)


class Quaternion:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __repr__(self):
        return "Quaternion(x={}, y={}, z={}, w={})".format(self.x, self.y, self.z, self.w)


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return "Point(x={}, y={}, z={})".format(self.x, self.y, self.z)


class Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation

    def __repr__(self):
        return "Pose(position={}, orientation={})".format(self.position, self.orientation)


def vector3_to_numpy(msg, hom=False):
    if hom:
        return np.array([msg.x, msg.y, msg.z, 0])
    else:
        return np.array([msg.x, msg.y, msg.z])


def numpy_to_vector3(arr):
    if arr.shape[-1] == 4:
        assert np.all(arr[..., -1] == 0)
        arr = arr[..., :-1]

    if len(arr.shape) == 1:
        return Vector3(**dict(zip(['x', 'y', 'z'], arr)))
    else:
        return np.apply_along_axis(
            lambda v: Vector3(**dict(zip(['x', 'y', 'z'], v))), axis=-1,
            arr=arr)


def point_to_numpy(msg, hom=False):
    if hom:
        return np.array([msg.x, msg.y, msg.z, 1])
    else:
        return np.array([msg.x, msg.y, msg.z])


def numpy_to_point(arr):
    if arr.shape[-1] == 4:
        arr = arr[..., :-1] / arr[..., -1]

    if len(arr.shape) == 1:
        return Point(**dict(zip(['x', 'y', 'z'], arr)))
    else:
        return np.apply_along_axis(
            lambda v: Point(**dict(zip(['x', 'y', 'z'], v))), axis=-1, arr=arr)


def quat_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z, msg.w])


def numpy_to_quat(arr):
    assert arr.shape[-1] == 4

    if len(arr.shape) == 1:
        return Quaternion(**dict(zip(['x', 'y', 'z', 'w'], arr)))
    else:
        return np.apply_along_axis(
            lambda v: Quaternion(**dict(zip(['x', 'y', 'z', 'w'], v))),
            axis=-1, arr=arr)


def transform_to_numpy(msg):
    trans = np.array(transformations.translation_matrix(
        [msg.translation.x, msg.translation.y, msg.translation.z]))
    rot = np.array(transformations.quaternion_matrix(
        [msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w]))
    return np.dot(trans, rot)


def numpy_to_transform(arr):
    shape, rest = arr.shape[:-2], arr.shape[-2:]
    assert rest == (4, 4)

    if len(shape) == 0:
        trans = transformations.translation_from_matrix(arr)
        quat = transformations.quaternion_from_matrix(arr)

        return Transform(
            translation=Vector3(**dict(zip(['x', 'y', 'z'], trans))),
            rotation=Quaternion(**dict(zip(['x', 'y', 'z', 'w'], quat)))
        )
    else:
        res = np.empty(shape, dtype=np.object_)
        for idx in np.ndindex(shape):
            res[idx] = Transform(
                translation=Vector3(
                    **dict(
                        zip(['x', 'y', 'z'],
                            transformations.translation_from_matrix(arr[idx])))),
                rotation=Quaternion(
                    **dict(
                        zip(['x', 'y', 'z', 'w'],
                            transformations.quaternion_from_matrix(arr[idx]))))
            )


def pose_to_numpy(msg):
    return np.dot(
        transformations.translation_matrix([msg.position.x, msg.position.y, msg.position.z]),
        transformations.quaternion_matrix([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
    )


def numpy_to_pose(arr):
    shape, rest = arr.shape[:-2], arr.shape[-2:]
    assert rest == (4, 4)

    if len(shape) == 0:
        trans = transformations.translation_from_matrix(arr)
        quat = transformations.quaternion_from_matrix(arr)

        return Pose(
            position=Point(**dict(zip(['x', 'y', 'z'], trans))),
            orientation=Quaternion(**dict(zip(['x', 'y', 'z', 'w'], quat)))
        )
    else:
        res = np.empty(shape, dtype=np.object_)
        for idx in np.ndindex(shape):
            res[idx] = Pose(
                position=Point(
                    **dict(
                        zip(['x', 'y', 'z'],
                            transformations.translation_from_matrix(arr[idx])))),
                orientation=Quaternion(
                    **dict(
                        zip(['x', 'y', 'z', 'w'],
                            transformations.quaternion_from_matrix(arr[idx]))))
            )
