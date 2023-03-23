import roslibpy
import roslibpy2numpy as r2n
import cv2
import sys


def receive_map(message):
    map = r2n.occupancygrid_to_numpy(message)
    cv2.imshow('map', map)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit()


if __name__ == '__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    map_subscriber = roslibpy.Topic(client, '/map', 'nav_msgs/OccupancyGrid')
    map_subscriber.subscribe(receive_map)

    client.run_forever()
