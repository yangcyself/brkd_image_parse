#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


polygon_points = [
    [250.0, 100.0], # left top
    [570.0, 100.0], # right top
    [570.0, 410.0], # right bottom
    [273.0, 395.0], # left bottom
]

class MegaBlockStateEstimator(Node):
    def __init__(self):
        super().__init__('mega_block_state_estimator')
        # Define the 4 points of the convex polygon (modify via ROS2 parameters if desired)
        # self.declare_parameter('polygon_points', [100.0, 100.0, 400.0, 100.0, 400.0, 400.0, 100.0, 400.0])
        self.declare_parameter('grid_rows', 9)
        self.declare_parameter('grid_cols', 5)

        # pts_list = self.get_parameter('polygon_points').value
        # self.pts = np.array(pts_list, dtype=np.int32).reshape((-1, 2))
        self.pts = np.array(polygon_points, dtype=np.int32)
        self.grid_rows = self.get_parameter('grid_rows').value
        self.grid_cols = self.get_parameter('grid_cols').value

        self.bridge = CvBridge()
        # Subscribers and Publishers
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.polygon_pub = self.create_publisher(Image, '/polygon_overlay', 10)
        self.grid_pub = self.create_publisher(Image, '/grid_overlay', 10)

    def image_callback(self, msg: Image):
        # Convert ROS Image to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 1) Publish polygon overlay on original image
        self.publish_polygon_overlay(cv_img)

        # 2) Warp convex polygon to a 400x400 square
        mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, self.pts, 255)
        masked = cv2.bitwise_and(cv_img, cv_img, mask=mask)
        src_pts = self.pts.astype(np.float32)
        dst_pts = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        crop = cv2.warpPerspective(masked, M, (400, 400))


        # 3) Divide into grid and compute mean color
        h_crop, w_crop = crop.shape[:2]
        cell_h = h_crop // self.grid_rows
        cell_w = w_crop // self.grid_cols
        gray_std_table = np.zeros((self.grid_rows, self.grid_cols))
        lap_mean_table = np.zeros((self.grid_rows, self.grid_cols))
        mean_bgr_table = np.zeros((self.grid_rows, self.grid_cols, 3))
        has_color_table = np.zeros((self.grid_rows, self.grid_cols))
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                y1, y2 = i * cell_h, (i + 1) * cell_h if i < self.grid_rows - 1 else h_crop
                x1, x2 = j * cell_w, (j + 1) * cell_w if j < self.grid_cols - 1 else w_crop
                cell = crop[y1+2:y2-2, x1+2:x2-2]
                mean_bgr = cv2.mean(cell)[:3]
                mean_bgr_table[i, j] = mean_bgr
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                gray_std = np.std(gray)
                gray_std_table[i, j] = gray_std
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                lap_mean = np.mean(np.abs(lap))
                lap_mean_table[i, j] = lap_mean
                has_color_table[i,j] = ((cell.max(-1)+0.1)/(cell.min(-1)+0.1)).mean()
                # table.append(((i, j), mean_bgr))

        # 4) Print results as a table
        # self.print_table(table)
        # print("gray_std_table", gray_std_table)
        # print("lap_mean_table", lap_mean_table)      
        valid_table = gray_std_table < 10
        has_color_table = has_color_table > 120
        # has_color_table = ((mean_bgr_table/255.0)**2).sum(-1) < 0.6
        has_block_table = valid_table & has_color_table
        

        # 5) Publish grid overlay on cropped image
        self.publish_grid_overlay(crop, cell_w, cell_h, has_block_table)

    def publish_polygon_overlay(self, img: np.ndarray):
        overlay = img.copy()
        cv2.polylines(overlay, [self.pts], isClosed=True, color=(0, 255, 0), thickness=2)
        msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        self.polygon_pub.publish(msg)

    def publish_grid_overlay(self, crop: np.ndarray, cell_w: int, cell_h: int, valid_table=None):
        overlay = crop.copy()
        h, w = overlay.shape[:2]
        # Draw horizontal lines
        for i in range(1, self.grid_rows):
            y = i * cell_h
            cv2.line(overlay, (0, y), (w, y), (255, 0, 0), 1)
        # Draw vertical lines
        for j in range(1, self.grid_cols):
            x = j * cell_w
            cv2.line(overlay, (x, 0), (x, h), (255, 0, 0), 1)
        if valid_table is not None:
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    if valid_table[i, j]:
                        cv2.rectangle(overlay, (j * cell_w, i * cell_h), ((j + 1) * cell_w, (i + 1) * cell_h), (0, 255, 0), 2)
        msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        self.grid_pub.publish(msg)

    def print_table(self, table):
        # Simple ASCII table output
        print(f"{'Cell':>6} | {'B':>6} | {'G':>6} | {'R':>6}")
        print('-' * 29)
        for (i, j), (b, g, r) in table:
            print(f"({i},{j}) | {b:6.2f} | {g:6.2f} | {r:6.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = MegaBlockStateEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
