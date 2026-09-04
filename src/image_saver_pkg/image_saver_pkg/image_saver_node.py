#!/usr/bin/env python3
"""
图像保存节点 - 保存检测图像
功能:
1. 检测到缺陷时立即拍照保存
2. 每0.5秒定时保存一张图像
3. 保存带标注的图像到指定文件夹
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime


class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')

        # 参数
        self.declare_parameter('save_dir', '/home/sunrise/saved_images')
        self.declare_parameter('save_interval', 0.5)  # 定时保存间隔（秒）
        self.declare_parameter('save_on_defect', True)  # 检测到缺陷时立即保存

        self.save_dir = self.get_parameter('save_dir').value
        save_interval = self.get_parameter('save_interval').value
        self.save_on_defect = self.get_parameter('save_on_defect').value

        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f'图像保存目录: {self.save_dir}')

        # CV Bridge
        self.bridge = CvBridge()

        # 当前图像
        self.current_image = None
        self.defect_detected = False

        # 订阅器
        self.image_sub = self.create_subscription(
            Image, '/detection/annotated_image', self.image_callback, 10
        )
        self.defect_sub = self.create_subscription(
            Bool, '/detection/defect_detected', self.defect_callback, 10
        )

        # 定时器 - 每0.5秒保存一次
        self.timer = self.create_timer(save_interval, self.timer_callback)

        self.save_count = 0
        self.defect_save_count = 0

    def image_callback(self, msg):
        """接收图像"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')

    def defect_callback(self, msg):
        """接收缺陷检测结果"""
        # 检测到缺陷且之前没有缺陷时，立即保存
        if msg.data and not self.defect_detected and self.save_on_defect:
            if self.current_image is not None:
                self.save_image(prefix='defect')
                self.defect_save_count += 1
                self.get_logger().info(f'⚠️ 检测到缺陷，立即保存图像 (#{self.defect_save_count})')

        self.defect_detected = msg.data

    def timer_callback(self):
        """定时保存图像"""
        if self.current_image is not None:
            self.save_image(prefix='periodic')
            self.save_count += 1

            if self.save_count % 10 == 0:
                self.get_logger().info(f'已定时保存 {self.save_count} 张图像')

    def save_image(self, prefix='image'):
        """保存图像到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filename = f'{prefix}_{timestamp}.jpg'
        filepath = os.path.join(self.save_dir, filename)

        try:
            cv2.imwrite(filepath, self.current_image)
        except Exception as e:
            self.get_logger().error(f'保存图像失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
