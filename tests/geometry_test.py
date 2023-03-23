import roslibpy2numpy
import numpy as np
import unittest


class TestRoslibpy2numpy(unittest.TestCase):
    def test_numpy_to_quat(self):  # Passed
        msg = roslibpy2numpy.geometry.numpy_to_quat(np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1))
        self.assertEqual(msg, {'x': 1.0, 'y': 2.0, 'z': 3.0, 'w': 4.0})

    def test_numpy_to_vector3(self):  # Passed
        msg = roslibpy2numpy.geometry.numpy_to_vector3(np.array([1.0, 2.0, 3.0]).reshape(3, 1))
        self.assertEqual(msg, {'x': 1.0, 'y': 2.0, 'z': 3.0})

    def test_point_to_numpy(self):  # Passed
        msg = roslibpy2numpy.geometry.point_to_numpy({'x': 1.0, 'y': 2.0, 'z': 3.0})
        self.assertEqual(msg.all(), np.array([1.0, 2.0, 3.0]).reshape(3, 1).all())

    def test_quat_to_numpy(self):  # Passed
        msg = roslibpy2numpy.geometry.quat_to_numpy({'x': 1.0, 'y': 2.0, 'z': 3.0, 'w': 4.0})
        self.assertEqual(msg.all(), np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1).all())

    #
    def test_vector3_to_numpy(self):
        msg = roslibpy2numpy.geometry.vector3_to_numpy({'x': 1.0, 'y': 2.0, 'z': 3.0})
        self.assertEqual(msg.all(), np.array([1.0, 2.0, 3.0]).reshape(3, 1).all())

    #
    def test_vector3_to_numpy_hom(self):
        msg = roslibpy2numpy.geometry.vector3_to_numpy({'x': 1.0, 'y': 2.0, 'z': 3.0}, hom=True)
        self.assertEqual(msg.all(), np.array([1.0, 2.0, 3.0, 0.0]).reshape(4, 1).all())

    def test_point_to_numpy_hom(self):
        msg = roslibpy2numpy.geometry.point_to_numpy({'x': 1.0, 'y': 2.0, 'z': 3.0}, hom=True)
        self.assertEqual(msg.all(), np.array([1.0, 2.0, 3.0, 1.0]).reshape(4, 1).all())

    def test_numpy_to_point(self):
        msg = roslibpy2numpy.geometry.numpy_to_point(np.array([1.0, 2.0, 3.0]).reshape(3, 1))
        self.assertEqual(msg, {'x': 1.0, 'y': 2.0, 'z': 3.0})
