#!/usr/bin/env python3
"""
Defect detection node.

Pipeline:
1. Extract a long dark cable band with OpenCV.
2. Suppress background before YOLO inference.
3. Run BPU YOLO coarse detection.
4. Filter detections outside the cable band.
5. Refine each YOLO box in ROI with OpenCV features.
"""

import json
import os
import re
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from hobot_dnn import pyeasy_dnn as dnn
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        self.declare_parameter('model_path', '/home/sunrise/cable1/models/deployed/yolo26dino3_bpu_bayese_640x640_nv12.bin')
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('result_topic', '/detection/results')
        self.declare_parameter('annotated_topic', '/detection/annotated_image')
        self.declare_parameter('mask_topic', '/detection/cable_mask')

        self.declare_parameter('conf_thresh', 0.50)
        self.declare_parameter('nms_thresh', 0.45)
        self.declare_parameter('class_names', 'burn,pr')

        self.declare_parameter('use_cable_mask', True)
        self.declare_parameter('pre_mask_for_yolo', True)
        self.declare_parameter('show_cable_mask', True)
        self.declare_parameter('publish_cable_mask', False)
        self.declare_parameter('background_gray', 127)
        self.declare_parameter('inference_interval', 3)

        self.declare_parameter('cable_dark_l_thresh', 90)
        self.declare_parameter('cable_dark_v_thresh', 90)
        self.declare_parameter('cable_extra_pad', 10)
        self.declare_parameter('bbox_mask_min_ratio', 0.12)
        self.declare_parameter('min_cable_span_ratio', 0.35)

        self.declare_parameter('roi_pad', 24)
        self.declare_parameter('min_defect_area', 12)
        self.declare_parameter('refine_blackhat_weight', 0.50)
        self.declare_parameter('refine_gradient_weight', 0.30)
        self.declare_parameter('refine_contrast_weight', 0.20)

        model_path = self.get_parameter('model_path').value
        input_topic = self.get_parameter('input_topic').value
        result_topic = self.get_parameter('result_topic').value
        annotated_topic = self.get_parameter('annotated_topic').value
        mask_topic = self.get_parameter('mask_topic').value

        self.conf_thresh = float(self.get_parameter('conf_thresh').value)
        self.nms_thresh = float(self.get_parameter('nms_thresh').value)

        class_names = str(self.get_parameter('class_names').value)
        self.CLASSES = [name.strip() for name in class_names.split(',') if name.strip()]
        if not self.CLASSES:
            self.CLASSES = ['burn', 'pr']

        self.use_cable_mask = bool(self.get_parameter('use_cable_mask').value)
        self.pre_mask_for_yolo = bool(self.get_parameter('pre_mask_for_yolo').value)
        self.show_cable_mask = bool(self.get_parameter('show_cable_mask').value)
        self.publish_cable_mask = bool(self.get_parameter('publish_cable_mask').value)
        self.background_gray = int(self.get_parameter('background_gray').value)
        self.inference_interval = max(1, int(self.get_parameter('inference_interval').value))

        self.cable_dark_l_thresh = int(self.get_parameter('cable_dark_l_thresh').value)
        self.cable_dark_v_thresh = int(self.get_parameter('cable_dark_v_thresh').value)
        self.cable_extra_pad = int(self.get_parameter('cable_extra_pad').value)
        self.bbox_mask_min_ratio = float(self.get_parameter('bbox_mask_min_ratio').value)
        self.min_cable_span_ratio = float(self.get_parameter('min_cable_span_ratio').value)

        self.roi_pad = int(self.get_parameter('roi_pad').value)
        self.min_defect_area = int(self.get_parameter('min_defect_area').value)
        self.refine_blackhat_weight = float(self.get_parameter('refine_blackhat_weight').value)
        self.refine_gradient_weight = float(self.get_parameter('refine_gradient_weight').value)
        self.refine_contrast_weight = float(self.get_parameter('refine_contrast_weight').value)

        self.COLORS = [(0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 255, 0)]
        self.STRIDES = [8, 16, 32]
        self.INPUT_SIZE = 640

        self.get_logger().info(f'加载模型: {model_path}')
        self.models = dnn.load(model_path)[0]
        self.get_logger().info('模型加载成功')

        self.grids = {}
        for stride in self.STRIDES:
            gh, gw = self.INPUT_SIZE // stride, self.INPUT_SIZE // stride
            grid = np.stack(np.indices((gh, gw))[::-1], axis=-1)
            self.grids[stride] = grid.reshape(-1, 2).astype(np.float32) + 0.5

        self.conf_raw = -np.log(1 / self.conf_thresh - 1)

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, input_topic, self.image_callback, qos_profile_sensor_data
        )
        self.detection_pub = self.create_publisher(String, result_topic, 10)
        self.annotated_pub = self.create_publisher(
            Image, annotated_topic, qos_profile_sensor_data
        )
        self.mask_pub = self.create_publisher(
            Image, mask_topic, qos_profile_sensor_data
        )

        self.frame_count = 0
        self.W = None
        self.H = None
        self.last_detections = []
        self.last_cable_mask = None
        self.last_frame_shape = None
        self.last_inference_frame = -1
        self.reused_detection_frames = 0

        self.get_logger().info(
            '检测增强已启用: '
            f'classes={self.CLASSES}, conf={self.conf_thresh:.2f}, '
            f'cable_mask={self.use_cable_mask}, pre_mask={self.pre_mask_for_yolo}, '
            f'L<{self.cable_dark_l_thresh}, V<{self.cable_dark_v_thresh}, '
            f'roi_pad={self.roi_pad}, inference_interval={self.inference_interval}'
        )

    def fit_cable_band_from_dark_mask(self, dark_mask):
        """Fit a bounded band around the best long dark component."""
        h, w = dark_mask.shape[:2]
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.002 * h * w:
                continue

            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if rw < 2 or rh < 2:
                continue

            aspect = max(rw, rh) / max(min(rw, rh), 1.0)
            x, y, bw, bh = cv2.boundingRect(contour)
            span = max(bw / max(w, 1), bh / max(h, 1))
            score = area * (1.0 + min(aspect, 14.0)) * (0.6 + span)
            candidates.append((score, contour))

        if not candidates:
            return None

        contour = max(candidates, key=lambda item: item[0])[1]
        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) < 8:
            return None

        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        direction = np.array([vx, vy], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        direction /= norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        origin = np.array([x0, y0], dtype=np.float32)

        rel = points - origin
        t = rel @ direction
        dist = np.abs(rel @ normal)
        t_min = float(np.percentile(t, 2))
        t_max = float(np.percentile(t, 98))
        if t_max - t_min < min(w, h) * 0.25:
            return None

        half_width = int(np.percentile(dist, 82) + self.cable_extra_pad)
        half_width = int(np.clip(half_width, max(8, h * 0.05), max(12, h * 0.12)))
        extend = max(12, int(min(w, h) * 0.06))
        p1 = origin + direction * (t_min - extend)
        p2 = origin + direction * (t_max + extend)

        polygon = np.array([
            p1 + normal * half_width,
            p2 + normal * half_width,
            p2 - normal * half_width,
            p1 - normal * half_width,
        ], dtype=np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        close_k = self.odd_kernel(min(h, w) * 0.04, 3, 13)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
            iterations=1,
        )
        return mask

    def odd_kernel(self, value, min_value=3, max_value=99):
        value = int(np.clip(value, min_value, max_value))
        return value if value % 2 == 1 else value + 1

    def preprocess_image(self, frame):
        """Convert BGR frame to model NV12 input with letterbox padding."""
        self.H, self.W = frame.shape[:2]

        scale = min(self.INPUT_SIZE / self.H, self.INPUT_SIZE / self.W)
        nw, nh = int(self.W * scale), int(self.H * scale)
        x_shift = (self.INPUT_SIZE - nw) // 2
        y_shift = (self.INPUT_SIZE - nh) // 2

        resized = cv2.resize(frame, (nw, nh))
        padded = cv2.copyMakeBorder(
            resized,
            y_shift,
            self.INPUT_SIZE - nh - y_shift,
            x_shift,
            self.INPUT_SIZE - nw - x_shift,
            cv2.BORDER_CONSTANT,
            value=self.background_gray,
        )

        yuv = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420).flatten()
        nv12 = np.empty((self.INPUT_SIZE * self.INPUT_SIZE * 3 // 2,), dtype=np.uint8)
        y_sz = self.INPUT_SIZE * self.INPUT_SIZE
        nv12[:y_sz] = yuv[:y_sz]
        nv12[y_sz::2] = yuv[y_sz:y_sz + y_sz // 4]
        nv12[y_sz + 1::2] = yuv[y_sz + y_sz // 4:]

        return nv12, scale, x_shift, y_shift

    def postprocess_detections(self, outputs, scale, x_shift, y_shift):
        """Decode YOLO outputs and map boxes back to original image coordinates."""
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
            xyxy = np.hstack([grid - v_box[:, :2], grid + v_box[:, 2:]]) * stride
            dets.extend(np.hstack([xyxy, v_score[:, None], v_id[:, None]]))

        if len(dets) == 0:
            return []

        dets = np.array(dets)
        xywh = dets[:, :4].copy()
        xywh[:, 2:] -= xywh[:, :2]

        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            dets[:, 4].tolist(),
            self.conf_thresh,
            self.nms_thresh,
        )
        if len(indices) == 0:
            return []

        results = []
        for idx in np.array(indices).flatten():
            d = dets[int(idx)]
            x1, y1, x2, y2 = (d[:4] - [x_shift, y_shift, x_shift, y_shift]) / scale
            x1 = int(np.clip(x1, 0, self.W))
            y1 = int(np.clip(y1, 0, self.H))
            x2 = int(np.clip(x2, 0, self.W))
            y2 = int(np.clip(y2, 0, self.H))
            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(d[5])
            if cls_id < 0 or cls_id >= len(self.CLASSES):
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            results.append({
                'class': self.CLASSES[cls_id],
                'confidence': float(d[4]),
                'bbox': [x1, y1, x2, y2],
                'center': [float(cx), float(cy)],
                'refined': False,
            })

        return results

    def extract_cable_band_mask(self, frame):
        """
        Extract the full long cable band, not only dark pixels.

        The cable is expected to be a black, long, straight or slightly curved object.
        Defects can be brighter than the cable, so the final mask is a fitted band
        around the dark cable centerline.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        dark = (
            (lab[:, :, 0] < self.cable_dark_l_thresh) |
            (hsv[:, :, 2] < self.cable_dark_v_thresh)
        ).astype(np.uint8) * 255
        dark = cv2.medianBlur(dark, 5)

        close_w = self.odd_kernel(w * 0.18, 9, 61)
        close_h = self.odd_kernel(h * 0.06, 3, 17)
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
            iterations=2,
        )
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        fitted_mask = self.fit_cable_band_from_dark_mask(dark)
        if fitted_mask is not None:
            area_ratio = float(np.count_nonzero(fitted_mask)) / float(fitted_mask.size)
            if 0.02 <= area_ratio <= 0.70:
                return fitted_mask

        xs = []
        ys = []
        widths = []
        step = max(1, w // 80)
        min_run_len = max(4, int(h * 0.015))

        for x in range(0, w, step):
            y_idx = np.where(dark[:, x] > 0)[0]
            if len(y_idx) < min_run_len:
                continue

            splits = np.where(np.diff(y_idx) > 1)[0] + 1
            runs = np.split(y_idx, splits)
            run = max(runs, key=len)
            if len(run) < min_run_len:
                continue

            xs.append(x)
            ys.append(float(np.median(run)))
            widths.append(float(run[-1] - run[0] + 1))

        min_points = max(8, int(w * self.min_cable_span_ratio / max(step, 1)))
        if len(xs) < min_points:
            return self.extract_cable_mask_fallback(frame, dark)

        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        widths = np.array(widths, dtype=np.float32)

        med_y = np.median(ys)
        mad_y = np.median(np.abs(ys - med_y)) + 1e-6
        keep = np.abs(ys - med_y) < (3 * mad_y + max(20, h * 0.18))
        xs2 = xs[keep]
        ys2 = ys[keep]
        widths2 = widths[keep]
        if len(xs2) < max(8, int(len(xs) * 0.35)):
            xs2, ys2, widths2 = xs, ys, widths

        degree = 2 if len(xs2) > 45 else 1
        try:
            poly = np.polyfit(xs2, ys2, degree)
        except Exception:
            return self.extract_cable_mask_fallback(frame, dark)

        span = (float(np.max(xs2)) - float(np.min(xs2))) / max(w, 1)
        if span < self.min_cable_span_ratio:
            return self.extract_cable_mask_fallback(frame, dark)

        half_width = int(np.percentile(widths2, 80) / 2 + self.cable_extra_pad)
        half_width = int(np.clip(half_width, max(8, h * 0.05), max(12, h * 0.12)))

        x_min = max(0, int(np.min(xs2) - 20))
        x_max = min(w - 1, int(np.max(xs2) + 20))
        mask = np.zeros((h, w), dtype=np.uint8)
        for x in range(x_min, x_max + 1):
            yc = float(np.polyval(poly, x))
            y1 = max(0, int(yc - half_width))
            y2 = min(h, int(yc + half_width))
            mask[y1:y2, x] = 255

        close_k = self.odd_kernel(min(h, w) * 0.08, 5, 25)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
            iterations=1,
        )
        if np.count_nonzero(mask) / float(mask.size) > 0.70 and fitted_mask is not None:
            return fitted_mask
        return mask

    def extract_cable_mask_fallback(self, frame, dark_mask=None):
        """Fallback: keep the largest long dark connected component, then dilate it."""
        h, w = frame.shape[:2]
        if dark_mask is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            dark_mask = (
                (lab[:, :, 0] < self.cable_dark_l_thresh) |
                (hsv[:, :, 2] < self.cable_dark_v_thresh)
            ).astype(np.uint8) * 255

        close_k = self.odd_kernel(min(h, w) * 0.12, 7, 31)
        dark = cv2.morphologyEx(
            dark_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
            iterations=2,
        )

        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones((h, w), dtype=np.uint8) * 255

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.002 * h * w:
                continue

            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if rw < 1 or rh < 1:
                continue

            aspect = max(rw, rh) / max(min(rw, rh), 1.0)
            x, y, bw, bh = cv2.boundingRect(contour)
            span = max(bw / max(w, 1), bh / max(h, 1))
            score = area * (1.0 + 0.7 * min(aspect, 12.0) + 1.0 * span)
            candidates.append((score, contour))

        mask = np.zeros((h, w), dtype=np.uint8)
        best = max(candidates, key=lambda item: item[0])[1] if candidates else max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [best], -1, 255, -1)

        fitted_mask = self.fit_cable_band_from_dark_mask(dark)
        if fitted_mask is not None:
            return fitted_mask

        dilate_k = self.odd_kernel(min(h, w) * 0.08, 5, 19)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)),
            iterations=2,
        )
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)),
            iterations=1,
        )
        return mask

    def apply_cable_mask_for_yolo(self, frame, cable_mask):
        masked = frame.copy()
        masked[cable_mask == 0] = (self.background_gray, self.background_gray, self.background_gray)
        return masked

    def bbox_mask_ratio(self, bbox, mask):
        x1, y1, x2, y2 = bbox
        h, w = mask.shape[:2]
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h))
        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(np.count_nonzero(roi)) / float(roi.size)

    def filter_detections_by_cable_mask(self, detections, cable_mask):
        if cable_mask is None:
            return detections

        filtered = []
        h, w = cable_mask.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']
            cx_i = int(np.clip(cx, 0, w - 1))
            cy_i = int(np.clip(cy, 0, h - 1))
            center_on_cable = cable_mask[cy_i, cx_i] > 0
            ratio = self.bbox_mask_ratio([x1, y1, x2, y2], cable_mask)
            if center_on_cable or ratio >= self.bbox_mask_min_ratio:
                det['mask_ratio'] = float(ratio)
                filtered.append(det)
        return filtered

    def extract_roi_cable_mask(self, gray):
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        close_k = self.odd_kernel(min(gray.shape[:2]) * 0.12, 5, 17)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
            iterations=2,
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cable_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cable_contour) < max(20, 0.01 * gray.shape[0] * gray.shape[1]):
            return None

        cable_mask = np.zeros_like(mask)
        cv2.drawContours(cable_mask, [cable_contour], -1, 255, -1)
        return cable_mask

    def normalize_score(self, data):
        if data.size == 0:
            return data.astype(np.float32)
        data = data.astype(np.float32)
        if float(np.max(data) - np.min(data)) < 1e-6:
            return np.zeros_like(data, dtype=np.float32)
        return cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX)

    def refine_defect(self, frame, det, cable_band_mask=None):
        x1, y1, x2, y2 = det['bbox']
        h, w = frame.shape[:2]
        pad = self.roi_pad

        rx1 = max(0, int(x1 - pad))
        ry1 = max(0, int(y1 - pad))
        rx2 = min(w, int(x2 + pad))
        ry2 = min(h, int(y2 + pad))
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return det

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        roi_cable_mask = self.extract_roi_cable_mask(gray)
        if roi_cable_mask is None and cable_band_mask is not None:
            roi_cable_mask = cable_band_mask[ry1:ry2, rx1:rx2]
        if roi_cable_mask is None or np.count_nonzero(roi_cable_mask) == 0:
            return det

        blackhat_k = self.odd_kernel(min(gray.shape[:2]) * 0.16, 7, 21)
        blackhat = cv2.morphologyEx(
            blur,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blackhat_k, blackhat_k)),
        )

        grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        local_blur = cv2.GaussianBlur(blur, (15, 15), 0)
        contrast = cv2.absdiff(blur, local_blur)

        score = (
            self.refine_blackhat_weight * self.normalize_score(blackhat) +
            self.refine_gradient_weight * self.normalize_score(grad_mag) +
            self.refine_contrast_weight * self.normalize_score(contrast)
        )
        score = np.clip(score * 255, 0, 255).astype(np.uint8)
        score = cv2.bitwise_and(score, score, mask=roi_cable_mask)

        if np.count_nonzero(score) < self.min_defect_area:
            return det

        _, defect_mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=roi_cable_mask)
        defect_mask = cv2.morphologyEx(
            defect_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=2,
        )
        defect_mask = cv2.morphologyEx(
            defect_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [contour for contour in contours if cv2.contourArea(contour) >= self.min_defect_area]
        if not valid:
            return det

        all_pts = np.vstack(valid)
        rect = cv2.minAreaRect(all_pts)
        box = cv2.boxPoints(rect).astype(np.int32)
        cx_local, cy_local = rect[0]
        box_global = box + np.array([rx1, ry1])

        x_coords = box_global[:, 0]
        y_coords = box_global[:, 1]
        refined_x1 = int(np.clip(np.min(x_coords), 0, w))
        refined_y1 = int(np.clip(np.min(y_coords), 0, h))
        refined_x2 = int(np.clip(np.max(x_coords), 0, w))
        refined_y2 = int(np.clip(np.max(y_coords), 0, h))
        if refined_x2 <= refined_x1 or refined_y2 <= refined_y1:
            return det

        hull = cv2.convexHull(all_pts) + np.array([[rx1, ry1]])
        det['bbox'] = [refined_x1, refined_y1, refined_x2, refined_y2]
        det['center'] = [float(rx1 + cx_local), float(ry1 + cy_local)]
        det['rotated_box'] = box_global.tolist()
        det['contour'] = hull.reshape(-1, 1, 2)
        det['refined'] = True
        det['refine_area'] = float(sum(cv2.contourArea(contour) for contour in valid))
        return det

    def json_ready_detections(self, detections):
        publish_detections = []
        for det in detections:
            item = det.copy()
            item.pop('contour', None)
            item.pop('rotated_box', None)
            item['bbox'] = [int(v) for v in item['bbox']]
            item['center'] = [float(v) for v in item['center']]
            item['confidence'] = float(item['confidence'])
            item['refined'] = bool(item.get('refined', False))
            if 'mask_ratio' in item:
                item['mask_ratio'] = float(item['mask_ratio'])
            if 'refine_area' in item:
                item['refine_area'] = float(item['refine_area'])
            publish_detections.append(item)
        return publish_detections

    def draw_annotations(self, frame, detections, cable_mask):
        vis = frame.copy()

        if self.show_cable_mask and cable_mask is not None:
            contours, _ = cv2.findContours(cable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class']
            cls_id = self.CLASSES.index(cls_name) if cls_name in self.CLASSES else 0
            color = self.COLORS[cls_id % len(self.COLORS)]

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            if 'rotated_box' in det:
                box = np.array(det['rotated_box'], dtype=np.int32)
                cv2.polylines(vis, [box], True, (0, 255, 255), 2)
            if 'contour' in det:
                cv2.drawContours(vis, [det['contour']], -1, (255, 255, 0), 1)

            label = f"{cls_name}:{det['confidence']:.2f}"
            if det.get('refined'):
                label += " R"
            if 'mask_ratio' in det:
                label += f" m:{det['mask_ratio']:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cx, cy = det['center']
            cv2.drawMarker(vis, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 12, 1)

        if self.W is not None and self.H is not None:
            cv2.drawMarker(
                vis,
                (self.W // 2, self.H // 2),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                30,
                2,
            )
        return vis

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.H, self.W = frame.shape[:2]

            cable_mask = None
            current_shape = frame.shape[:2]
            shape_changed = self.last_frame_shape is not None and self.last_frame_shape != current_shape
            should_infer = (
                self.last_inference_frame < 0
                or shape_changed
                or self.frame_count % self.inference_interval == 0
            )

            if should_infer:
                frame_for_yolo = frame
                if self.use_cable_mask:
                    cable_mask = self.extract_cable_band_mask(frame)
                    if self.pre_mask_for_yolo and cable_mask is not None:
                        frame_for_yolo = self.apply_cable_mask_for_yolo(frame, cable_mask)

                nv12, scale, x_shift, y_shift = self.preprocess_image(frame_for_yolo)
                outputs = self.models.forward(nv12)
                detections = self.postprocess_detections(outputs, scale, x_shift, y_shift)

                if self.use_cable_mask and cable_mask is not None:
                    detections = self.filter_detections_by_cable_mask(detections, cable_mask)

                detections = [self.refine_defect(frame, det, cable_mask) for det in detections]
                self.last_detections = detections
                self.last_cable_mask = cable_mask
                self.last_frame_shape = current_shape
                self.last_inference_frame = self.frame_count
            else:
                detections = self.last_detections
                cable_mask = self.last_cable_mask
                self.reused_detection_frames += 1

            result_msg = String()
            result_msg.data = json.dumps(self.json_ready_detections(detections), ensure_ascii=False)
            self.detection_pub.publish(result_msg)


            vis = self.draw_annotations(frame, detections, cable_mask)
            annotated_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

            if self.publish_cable_mask and cable_mask is not None:
                mask_msg = self.bridge.cv2_to_imgmsg(cable_mask, encoding='mono8')
                mask_msg.header = msg.header
                self.mask_pub.publish(mask_msg)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                mask_area = 0.0
                if cable_mask is not None:
                    mask_area = float(np.count_nonzero(cable_mask)) / float(cable_mask.size)
                refined_count = sum(1 for det in detections if det.get('refined'))
                self.get_logger().info(
                    f'已处理 {self.frame_count} 帧, 检测 {len(detections)} 个, '
                    f'精修 {refined_count} 个, cable_mask={mask_area:.2f}, '
                    f'复用 {self.reused_detection_frames} 帧'
                )

        except Exception as exc:
            self.get_logger().error(f'检测失败: {exc}')

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
