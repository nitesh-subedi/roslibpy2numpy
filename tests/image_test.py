import roslibpy
import roslibpy2numpy as r2n
import cv2
import sys


def receive_image(message):
    image = r2n.raw_image_to_numpy(message)
    image_raw = r2n.numpy_to_image_raw(image)
    image_publisher.publish(image_raw)


if __name__ == '__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    image_publisher = roslibpy.Topic(client, '/color/image_raw/published', 'sensor_msgs/Image')
    image_subscriber = roslibpy.Topic(client, '/color/image_raw', 'sensor_msgs/Image')
    image_subscriber.subscribe(receive_image)

    client.run_forever()
