#!/usr/bin/env python3
"""
摄像头节点 - 采集并发布图像流
功能:
1. 从 /dev/video0 采集图像
2. 发布原始图像到 /camera/image_raw
3. 发布压缩图像到 /camera/image_raw/compressed (用于网络传输)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import os
import subprocess
import threading
import time
from rclpy.qos import qos_profile_sensor_data


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # 参数
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('camera_device', '')  # 可选：直接指定设备路径
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 15)
        self.declare_parameter('pixel_format', 'YUYV')
        self.declare_parameter('reconnect_after_failures', 3)
        self.declare_parameter('watchdog_timeout', 8.0)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('compressed_topic', '/camera/image_raw/compressed')
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('sharpness', 100)
        self.declare_parameter('contrast', 55)
        self.declare_parameter('saturation', 70)
        self.declare_parameter('crop_top', 0)

        camera_id = self.get_parameter('camera_id').value
        camera_device = self.get_parameter('camera_device').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
        self.pixel_format = self.get_parameter('pixel_format').value
        self.reconnect_after_failures = int(
            self.get_parameter('reconnect_after_failures').value
        )
        self.watchdog_timeout = float(
            self.get_parameter('watchdog_timeout').value
        )
        self.image_topic = self.get_parameter('image_topic').value
        self.compressed_topic = self.get_parameter('compressed_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        sharpness = self.get_parameter('sharpness').value
        contrast = self.get_parameter('contrast').value
        saturation = self.get_parameter('saturation').value
        crop_top = self.get_parameter('crop_top').value

        self.camera_id = camera_id
        self.camera_device = camera_device
        self.width = width
        self.height = height
        self.fps = fps
        self.sharpness = sharpness
        self.contrast = contrast
        self.saturation = saturation
        self.crop_top = max(0, int(crop_top))
        self.failed_reads = 0
        self.last_frame_monotonic = time.monotonic()
        self.watchdog_stop = threading.Event()

        self.cap = None
        self._open_camera()

        # 发布器
        self.image_pub = self.create_publisher(
            Image, self.image_topic, qos_profile_sensor_data
        )
        self.compressed_pub = self.create_publisher(
            CompressedImage, self.compressed_topic, qos_profile_sensor_data
        )

        # CV Bridge
        self.bridge = CvBridge()

        # 定时器 - 根据fps计算周期
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.frame_count = 0
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
        )
        self.watchdog_thread.start()

    def _watchdog_loop(self):
        while not self.watchdog_stop.wait(1.0):
            if self.watchdog_timeout <= 0:
                continue
            age = time.monotonic() - self.last_frame_monotonic
            if age <= self.watchdog_timeout:
                continue
            self.get_logger().fatal(
                f'摄像头超过 {age:.1f} 秒无新帧，退出并由启动脚本自动重启'
            )
            os._exit(70)

    def _set_v4l2_controls(self, device_path, sharpness, contrast, saturation):
        controls = f'sharpness={sharpness},contrast={contrast},saturation={saturation}'
        try:
            subprocess.run(
                ['v4l2-ctl', '-d', device_path, f'--set-ctrl={controls}'],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            self.get_logger().info(f'已应用摄像头画质参数: {controls}')
        except Exception as e:
            self.get_logger().warn(f'应用摄像头画质参数失败: {e}')

    def _open_camera(self):
        if self.cap is not None:
            self.cap.release()

        camera_str = self.camera_device or str(self.camera_id)
        source = self.camera_device if self.camera_device else self.camera_id
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)

        if self.pixel_format:
            fourcc = cv2.VideoWriter_fourcc(*self.pixel_format[:4])
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        device_path = self.camera_device or f'/dev/video{self.camera_id}'
        if os.path.exists(device_path):
            self._set_v4l2_controls(
                device_path,
                self.sharpness,
                self.contrast,
                self.saturation,
            )

        if not self.cap.isOpened():
            self.get_logger().error(f'无法打开摄像头 {camera_str}')
            raise RuntimeError(f'Camera {camera_str} not available')

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        actual_format = ''.join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))
        self.failed_reads = 0
        self.get_logger().info(
            f'摄像头已打开: {camera_str} '
            f'(请求 {self.width}x{self.height} @ {self.fps}fps {self.pixel_format}, '
            f'实际 {actual_width}x{actual_height} @ {actual_fps:.1f}fps {actual_format}, '
            f'顶部裁切 {self.crop_top}px)'
        )

    def _recover_camera_if_needed(self):
        if self.failed_reads < self.reconnect_after_failures:
            return

        self.get_logger().warn(
            f'连续 {self.failed_reads} 次读取帧失败，正在重连摄像头'
        )
        try:
            self._open_camera()
        except Exception as e:
            self.get_logger().error(f'重连摄像头失败: {e}')
            self.failed_reads = 0

    def timer_callback(self):
        try:
            # 清空缓存，读取最新帧

            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.failed_reads += 1
                self.get_logger().warn(f'读取帧失败 ({self.failed_reads})')
                self._recover_camera_if_needed()
                return

            self.failed_reads = 0
            self.last_frame_monotonic = time.monotonic()

            if self.crop_top > 0:
                frame_height, frame_width = frame.shape[:2]
                crop_top = min(self.crop_top, frame_height - 1)
                frame = cv2.resize(
                    frame[crop_top:, :],
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            # 发布原始图像
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = self.frame_id
            self.image_pub.publish(img_msg)

            # 发布压缩图像（用于远程查看）
            #ret_encode, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            #if ret_encode:
                #compressed_msg = CompressedImage()
                #compressed_msg.header = img_msg.header
                #compressed_msg.format = 'jpeg'
                #compressed_msg.data = jpeg.tobytes()
                #self.compressed_pub.publish(compressed_msg)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'已发布 {self.frame_count} 帧')

        except Exception as e:
            self.get_logger().error(f'处理图像失败: {e}')

    def destroy_node(self):
        self.watchdog_stop.set()
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
