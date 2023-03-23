import unittest
import numpy as np
import roslibpy
import roslibpy2numpy


class TestPointCloud(unittest.TestCase):
    def test_numpy_to_pointcloud2(self):
        arr = np.random.rand(3, 3)
        msg = roslibpy2numpy.point_cloud2.array_to_pointcloud2(arr)
        self.assertEqual(msg['width'], 3)
        self.assertEqual(msg['height'], 3)
        self.assertEqual(msg['fields'][0]['name'], 'x')
        self.assertEqual(msg['fields'][1]['name'], 'y')
        self.assertEqual(msg['fields'][2]['name'], 'z')
        self.assertEqual(msg['fields'][0]['offset'], 0)
        self.assertEqual(msg['fields'][1]['offset'], 4)
        self.assertEqual(msg['fields'][2]['offset'], 8)
        self.assertEqual(msg['fields'][0]['datatype'], 7)
        self.assertEqual(msg['fields'][1]['datatype'], 7)
        self.assertEqual(msg['fields'][2]['datatype'], 7)
        self.assertEqual(msg['fields'][0]['count'], 1)
        self.assertEqual(msg['fields'][1]['count'], 1)
        self.assertEqual(msg['fields'][2]['count'], 1)
        self.assertEqual(msg['is_bigendian'], False)
        self.assertEqual(msg['point_step'], 12)
        self.assertEqual(msg['row_step'],
                         msg['point_step'] * msg['width'])
        self.assertEqual(msg['is_dense'], True)
