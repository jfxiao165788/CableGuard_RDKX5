#!/usr/bin/env python3
"""
缺陷检测节点 - YOLO 粗检测 + OpenCV 精定位增强版

功能:
1. 订阅摄像头图像
2. 使用 hobot_dnn 加载 BPU 模型进行推理 (粗定位)
3. 使用 OpenCV 传统视觉算法进行精定位:
   - ROI 裁剪
   - 电缆主体分割
   - 黑帽 + 梯度融合增强
   - Otsu 阈值分割
   - 形态学连接碎块
   - 多轮廓合并
   - 旋转矩形拟合
4. 发布检测结果（缺陷类型、位置、置信度）
5. 发布带标注的图像
6. 判断缺陷是否在图像中央（只看横向，更适合电机找正）
"""

import json
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
from hobot_dnn import pyeasy_dnn as dnn


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        # ==================== 参数 ====================
        self.declare_parameter('model_path', '/home/sunrise/yolo26cable_bayese_640x640_nv12.bin')
        self.declare_parameter('conf_thresh', 0.10)
        self.declare_parameter('nms_thresh', 0.45)
        self.declare_parameter('center_threshold', 80)     # 横向中心阈值（像素）
        self.declare_parameter('center_confirm_frames', 3) # 连续几帧满足才算真正到中央
        self.declare_parameter('roi_pad', 30)              # YOLO 框外扩像素
        self.declare_parameter('min_defect_area', 10)      # 最小缺陷面积，过滤噪点

        model_path = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('conf_thresh').value
        self.nms_thresh = self.get_parameter('nms_thresh').value
        self.center_threshold = self.get_parameter('center_threshold').value
        self.center_confirm_frames = self.get_parameter('center_confirm_frames').value
        self.roi_pad = self.get_parameter('roi_pad').value
        self.min_defect_area = self.get_parameter('min_defect_area').value

        # ==================== 模型配置 ====================
        self.CLASSES = ['burn', 'pr']
        self.COLORS = [(0, 255, 0), (255, 0, 0)]
        self.STRIDES = [8, 16, 32]
        self.INPUT_SIZE = 640

        # ==================== 加载模型 ====================
        self.get_logger().info(f'加载模型: {model_path}')
        self.models = dnn.load(model_path)[0]
        self.get_logger().info('模型加载成功')

        # 预计算网格
        self.grids = {}
        for s in self.STRIDES:
            gh, gw = self.INPUT_SIZE // s, self.INPUT_SIZE // s
            grid = np.stack(np.indices((gh, gw))[::-1], axis=-1)
            self.grids[s] = grid.reshape(-1, 2).astype(np.float32) + 0.5

        # 置信度阈值对应 raw logit
        self.conf_raw = -np.log(1 / self.conf_thresh - 1)

        # ==================== ROS 组件 ====================
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.detection_pub = self.create_publisher(String, '/detection/results', 10)
        self.annotated_pub = self.create_publisher(Image, '/detection/annotated_image', 10)
        self.defect_center_pub = self.create_publisher(Bool, '/detection/defect_in_center', 10)
        self.defect_detected_pub = self.create_publisher(Bool, '/detection/defect_detected', 10)

        # ==================== 运行状态 ====================
        self.frame_count = 0
        self.W = None
        self.H = None

        # 中心判定稳定器
        self.center_hit_count = 0

    # ==========================================================
    # 图像预处理
    # ==========================================================
    def preprocess_image(self, frame):
        """预处理图像为 NV12 格式"""
        if self.W is None:
            self.H, self.W = frame.shape[:2]

        scale = min(self.INPUT_SIZE / self.H, self.INPUT_SIZE / self.W)
        nw, nh = int(self.W * scale), int(self.H * scale)
        x_shift = (self.INPUT_SIZE - nw) // 2
        y_shift = (self.INPUT_SIZE - nh) // 2

        resized = cv2.resize(frame, (nw, nh))
        padded = cv2.copyMakeBorder(
            resized,
            y_shift, self.INPUT_SIZE - nh - y_shift,
            x_shift, self.INPUT_SIZE - nw - x_shift,
            cv2.BORDER_CONSTANT,
            value=127,
        )

        # BGR -> NV12
        yuv = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420).flatten()
        nv12 = np.empty((self.INPUT_SIZE * self.INPUT_SIZE * 3 // 2,), dtype=np.uint8)
        y_sz = self.INPUT_SIZE * self.INPUT_SIZE
        nv12[:y_sz] = yuv[:y_sz]
        nv12[y_sz::2] = yuv[y_sz:y_sz + y_sz // 4]
        nv12[y_sz + 1::2] = yuv[y_sz + y_sz // 4:]

        return nv12, scale, x_shift, y_shift

    # ==========================================================
    # YOLO 后处理
    # ==========================================================
    def postprocess_detections(self, outputs, scale, x_shift, y_shift):
        """YOLO 输出后处理"""
        num_cls = len(self.CLASSES)
        dets = []

        for i, stride in enumerate(self.STRIDES):
            h_grid, w_grid = self.INPUT_SIZE // stride, self.INPUT_SIZE // stride
            box_data = outputs[i * 2].buffer.reshape(h_grid, w_grid, 4)
            cls_data = outputs[i * 2 + 1].buffer.reshape(h_grid, w_grid, num_cls)

            max_scores = np.max(cls_data, axis=2)
            mask = max_scores >= self.conf_raw
            if not np.any(mask):
                continue

            grid = self.grids[stride][mask.flatten()]
            v_box = box_data.reshape(-1, 4)[mask.flatten()]
            v_score = 1 / (1 + np.exp(-max_scores[mask]))
            v_id = np.argmax(cls_data.reshape(-1, num_cls)[mask.flatten()], axis=1)
            xyxy = np.hstack([(grid - v_box[:, :2]), (grid + v_box[:, 2:])]) * stride
            dets.extend(np.hstack([xyxy, v_score[:, None], v_id[:, None]]))

        results = []
        if len(dets) == 0:
            return results

        dets = np.array(dets)
        xywh = dets[:, :4].copy()
        xywh[:, 2:] -= xywh[:, :2]

        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            dets[:, 4].tolist(),
            self.conf_thresh,
            self.nms_thresh
        )

        if len(indices) == 0:
            return results

        for idx in indices.flatten():
            d = dets[idx]
            x1, y1, x2, y2 = (d[:4] - [x_shift, y_shift, x_shift, y_shift]) / scale

            x1 = int(np.clip(x1, 0, self.W))
            y1 = int(np.clip(y1, 0, self.H))
            x2 = int(np.clip(x2, 0, self.W))
            y2 = int(np.clip(y2, 0, self.H))

            cls_id = int(d[5])
            conf = float(d[4])

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            results.append({
                'class': self.CLASSES[cls_id],
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
                'center': [cx, cy]
            })

        return results

    # ==========================================================
    # 电缆主体提取
    # ==========================================================
    def extract_cable_mask(self, gray):
        """
        提取电缆主体区域：
        电缆通常较暗，先用 Otsu 二值化，再保留最大连通区域。
        """
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 电缆一般偏暗，反相后更容易提取
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=2
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cable_contour = max(contours, key=cv2.contourArea)

        cable_mask = np.zeros_like(mask)
        cv2.drawContours(cable_mask, [cable_contour], -1, 255, -1)

        return cable_mask

    # ==========================================================
    # 缺陷精定位
    # ==========================================================
    def refine_defect_v2(self, frame, det, pad=30):
        """
        使用 OpenCV 对 YOLO 粗检测框做精定位：
        1. 扩展 ROI
        2. 提取电缆 mask
        3. 黑帽 + 梯度融合
        4. Otsu 阈值
        5. 形态学连接碎块
        6. 合并多个缺陷轮廓
        7. 使用旋转矩形拟合更完整的缺陷范围
        """
        x1, y1, x2, y2 = det['bbox']
        H, W = frame.shape[:2]

        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(W, x2 + pad)
        ry2 = min(H, y2 + pad)

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return det

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1) 提取电缆主体
        cable_mask = self.extract_cable_mask(gray)
        if cable_mask is None:
            return det

        # 2) 黑帽增强：提取暗缺陷
        blackhat = cv2.morphologyEx(
            blur,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )

        # 3) 梯度增强：提取边缘/破损边界
        grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        # 4) 局部对比度增强
        blur2 = cv2.GaussianBlur(blur, (15, 15), 0)
        contrast = cv2.absdiff(blur, blur2)

        # 5) 归一化后融合
        blackhat_n = cv2.normalize(blackhat.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        grad_n = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)
        contrast_n = cv2.normalize(contrast.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

        score = 0.50 * blackhat_n + 0.30 * grad_n + 0.20 * contrast_n
        score = (score * 255).astype(np.uint8)

        # 只保留电缆区域内的响应
        score = cv2.bitwise_and(score, score, mask=cable_mask)

        # 6) 自适应阈值（Otsu）
        _, defect_mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 7) 形态学连接碎块，避免大缺陷只剩一小块
        defect_mask = cv2.morphologyEx(
            defect_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=2
        )
        defect_mask = cv2.morphologyEx(
            defect_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1
        )

        # 8) 提取轮廓
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return det

        # 过滤极小噪点
        valid = [c for c in contours if cv2.contourArea(c) >= self.min_defect_area]
        if not valid:
            return det

        # 9) 合并多个轮廓，而不是只取最大一个
        all_pts = np.vstack(valid)

        # 10) 旋转矩形拟合
        rect = cv2.minAreaRect(all_pts)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # 11) 高精度中心
        (cx_local, cy_local) = rect[0]
        refined_cx = rx1 + cx_local
        refined_cy = ry1 + cy_local

        # 12) 转回原图坐标
        box_global = box + np.array([rx1, ry1])

        x_coords = box_global[:, 0]
        y_coords = box_global[:, 1]
        refined_x1 = int(np.clip(np.min(x_coords), 0, W))
        refined_y1 = int(np.clip(np.min(y_coords), 0, H))
        refined_x2 = int(np.clip(np.max(x_coords), 0, W))
        refined_y2 = int(np.clip(np.max(y_coords), 0, H))

        # 更新检测结果
        det['bbox'] = [refined_x1, refined_y1, refined_x2, refined_y2]
        det['center'] = [float(refined_cx), float(refined_cy)]
        det['rotated_box'] = box_global.tolist()

        # 也保留一个整体轮廓用于可视化
        hull = cv2.convexHull(all_pts)
        det['contour'] = (hull + np.array([[rx1, ry1]])).reshape(-1, 1, 2)

        return det

    # ==========================================================
    # 中央判定：只看横向，更适合电机找正
    # ==========================================================
    def is_defect_in_center(self, detections):
        """
        只判断横向是否居中：
        - 只取置信度最高的目标
        - 只看 x 方向与图像中心的距离
        """
        if not detections or self.W is None:
            return False

        img_cx = self.W // 2

        best_det = max(detections, key=lambda d: d['confidence'])
        cx, _ = best_det['center']

        return abs(cx - img_cx) < self.center_threshold

    # ==========================================================
    # 主回调
    # ==========================================================
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 预处理
            nv12, scale, x_shift, y_shift = self.preprocess_image(frame)

            # 模型推理
            outputs = self.models.forward(nv12)

            # YOLO 粗检测
            raw_detections = self.postprocess_detections(outputs, scale, x_shift, y_shift)

            # OpenCV 精定位
            detections = []
            for det in raw_detections:
                refined_det = self.refine_defect_v2(frame, det, pad=self.roi_pad)
                detections.append(refined_det)

            # 发布 JSON（剥离 numpy 结构）
            publish_detections = []
            for d in detections:
                pub_d = d.copy()
                pub_d.pop('contour', None)
                pub_d.pop('rotated_box', None)
                publish_detections.append(pub_d)

            result_msg = String()
            result_msg.data = json.dumps(publish_detections)
            self.detection_pub.publish(result_msg)

            # 发布“是否检测到缺陷”
            defect_detected = Bool()
            defect_detected.data = len(detections) > 0
            self.defect_detected_pub.publish(defect_detected)

            # 发布“是否到中央”（连续多帧确认）
            in_center_now = self.is_defect_in_center(detections)
            if in_center_now:
                self.center_hit_count += 1
            else:
                self.center_hit_count = 0

            in_center = self.center_hit_count >= self.center_confirm_frames

            center_msg = Bool()
            center_msg.data = in_center
            self.defect_center_pub.publish(center_msg)

            # ==================== 可视化 ====================
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cls_name = det['class']
                conf = det['confidence']
                cls_id = self.CLASSES.index(cls_name)
                color = self.COLORS[cls_id % len(self.COLORS)]

                # 1. 粗/精框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                # 2. 旋转矩形
                if 'rotated_box' in det:
                    box = np.array(det['rotated_box'], dtype=np.int32)
                    cv2.polylines(frame, [box], True, (0, 255, 255), 2)

                # 3. 轮廓
                if 'contour' in det:
                    cv2.drawContours(frame, [det['contour']], -1, (255, 255, 0), 1)

                # 4. 标签
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(
                    frame, label,
                    (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

                # 5. 亚像素中心（绘制时取整）
                cx, cy = det['center']
                cv2.drawMarker(frame, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 12, 1)

            # 图像中心
            if self.W is not None:
                img_cx = self.W // 2
                img_cy = self.H // 2
                cv2.drawMarker(
                    frame, (img_cx, img_cy),
                    (0, 255, 255),
                    cv2.MARKER_CROSS, 30, 2
                )

            annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'已处理 {self.frame_count} 帧, 检测到 {len(detections)} 个缺陷, 中央: {in_center}'
                )

        except Exception as e:
            self.get_logger().error(f'检测失败: {e}')

    # ==========================================================
    # 销毁
    # ==========================================================
    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()