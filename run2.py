import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
from DroneTerminal import Drone
from time import sleep
import threading
from datetime import datetime
import os

drone = Drone(connection_string='127.0.0.1:14550')
drone.speed(3)
altitude = 17

model = YOLO('best.pt')
model.fuse()
circle_count=0

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Image, '/camera', self.captured_frame_callback, 10)
        self.subscription
        self.bridge = CvBridge()
        print("Initialized")

    def captured_frame_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        global circle_count
        results = model(frame)
        annotated_frame = results[0].plot()
        height, width, _ = annotated_frame.shape

        cv2.line(annotated_frame, (0, height // 2), (width, height // 2), (255, 255, 255), 2)

        center_frame_x = width // 2
        center_frame_y = height // 2

        for result in results[0].boxes:
            # Get the center coordinates of the bounding box
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Get box coordinates
            cls = result.cls[0]  # Class ID
            if(cls!=29):
                continue
            # # Draw bounding box on the image
            label = f'Class: {int(cls)}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # # Mark the center of the detected object
            if center_y >= center_frame_y - 10 and center_y <= center_frame_y + 10:
                circle_count+=1
                coords = drone.get_gps_coords()
                self.take_snapshot(annotated_frame,coords)
                sleep(1.5)

        text = f"Counter: {circle_count}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Image", annotated_frame)
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)

    def take_snapshot(self, frame, coords):
        snapshot_dir = 'snapshots'
        x, y = coords
        text_coords = f"Coords: {x},{y}"
        cv2.putText(frame, text_coords, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Get the current time in hh:mm:ss:ms format
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S:%f")[:-3]  # Get milliseconds as well

        text_time = f"Time: {timestamp}"
        cv2.putText(frame, text_time, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        os.makedirs(snapshot_dir, exist_ok=True)  # Create directory if it doesn't exist
        snapshot_path = os.path.join(snapshot_dir, f'snapshot_{timestamp}.png')
        cv2.imwrite(snapshot_path, frame)  # Save the frame as an image
        print(f"Snapshot taken and saved at: {snapshot_path}")


    def center_drone(self, target_position, frame_width, frame_height):
        if target_position:
            center_x, center_y = target_position
            offset_x = (center_x - frame_width // 2) / frame_width
            offset_y = (center_y - frame_height // 2) / frame_height
            print('Centering drone...')
            drone.vehicle.simple_goto(drone.vehicle.location.global_relative_frame.lat + offset_y * 0.0001,
                                       drone.vehicle.location.global_relative_frame.lon + offset_x * 0.0001)

def camera(args=None):
    rclpy.init(args=args)
    cam_feedback = MinimalSubscriber()
    rclpy.spin(cam_feedback)
    cam_feedback.destroy_node()
    rclpy.shutdown()

def traversal():
    drone.arm_and_takeoff(altitude)
    sleep(1)
    
    print("Starting journey...")
    with open('coordinate.txt', 'r') as file:
        coordinates = [tuple(map(float, line.split())) for line in file]

        for coord in coordinates:
            drone.goto_gps(coord)
            
            while True:
                x, y = drone.get_gps_coords()
                errorx = abs(round((coord[0] - x) * 10**6, 3))
                errory = abs(round((coord[1] - y) * 10**6, 3))
                sleep(0.5)

                if errorx + errory < 12:
                    print("Lock acquired! Sleeping for 5 seconds...")
                    sleep(5)
                    break
        drone.rtl()

t1 = threading.Thread(target=camera)
t2 = threading.Thread(target=traversal)

t1.start()
t2.start()
t1.join()
t2.join()
