#!/usr/bin/env python3
"""
Infrared detection node.

Subscribes to the infrared camera image, detects relative heat hotspots in
the UVC pseudo-color video, and publishes an annotated false-color image plus
JSON detection results.
"""

import json

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class InfraredDetectionNode(Node):
    def __init__(self):
        super().__init__("infrared_detection_node")

        self.declare_parameter("image_topic", "/infrared/image_raw")
        self.declare_parameter("annotated_topic", "/infrared/annotated_image")
        self.declare_parameter("results_topic", "/infrared/results")
        self.declare_parameter("alarm_topic", "/infrared/alarm")
        self.declare_parameter("intensity_threshold", 225)
        self.declare_parameter("temperature_threshold_c", 70.0)
        self.declare_parameter("min_area", 200)
        self.declare_parameter("max_regions", 12)
        self.declare_parameter("fallback_to_intensity_alarm", False)
        self.declare_parameter("relative_heat_alarm", False)
        self.declare_parameter("relative_heat_confirmations", 2)
        self.declare_parameter("relative_heat_roi_top_fraction", 0.16)
        self.declare_parameter("relative_heat_roi_bottom_fraction", 0.90)
        self.declare_parameter("relative_heat_percentile", 98.0)
        self.declare_parameter("relative_heat_min_score", 205.0)
        self.declare_parameter("relative_heat_min_contrast", 30.0)
        self.declare_parameter("relative_heat_min_area", 80)
        self.declare_parameter("relative_heat_max_area_ratio", 0.18)
        self.declare_parameter("relative_heat_local_sigma", 19.0)
        self.declare_parameter("display_width", 640)
        self.declare_parameter("display_height", 360)
        self.declare_parameter("display_low_percentile", 3.0)
        self.declare_parameter("display_high_percentile", 97.0)
        self.declare_parameter("display_smoothing", 0.35)
        self.declare_parameter("sharpen_amount", 0.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.annotated_topic = self.get_parameter("annotated_topic").value
        self.results_topic = self.get_parameter("results_topic").value
        self.alarm_topic = self.get_parameter("alarm_topic").value
        self.intensity_threshold = int(self.get_parameter("intensity_threshold").value)
        self.temperature_threshold_c = float(
            self.get_parameter("temperature_threshold_c").value
        )
        self.min_area = int(self.get_parameter("min_area").value)
        self.max_regions = int(self.get_parameter("max_regions").value)
        self.fallback_to_intensity_alarm = bool(
            self.get_parameter("fallback_to_intensity_alarm").value
        )
        self.relative_heat_alarm = bool(
            self.get_parameter("relative_heat_alarm").value
        )
        self.relative_heat_confirmations = max(
            1,
            int(self.get_parameter("relative_heat_confirmations").value),
        )
        self.relative_heat_roi_top_fraction = max(
            0.0,
            min(
                0.95,
                float(self.get_parameter("relative_heat_roi_top_fraction").value),
            ),
        )
        self.relative_heat_roi_bottom_fraction = max(
            self.relative_heat_roi_top_fraction + 0.01,
            min(
                1.0,
                float(self.get_parameter("relative_heat_roi_bottom_fraction").value),
            ),
        )
        self.relative_heat_percentile = max(
            50.0,
            min(99.9, float(self.get_parameter("relative_heat_percentile").value)),
        )
        self.relative_heat_min_score = max(
            0.0,
            min(255.0, float(self.get_parameter("relative_heat_min_score").value)),
        )
        self.relative_heat_min_contrast = max(
            0.0,
            min(255.0, float(self.get_parameter("relative_heat_min_contrast").value)),
        )
        self.relative_heat_min_area = max(
            1,
            int(self.get_parameter("relative_heat_min_area").value),
        )
        self.relative_heat_max_area_ratio = max(
            0.001,
            min(1.0, float(self.get_parameter("relative_heat_max_area_ratio").value)),
        )
        self.relative_heat_local_sigma = max(
            1.0,
            float(self.get_parameter("relative_heat_local_sigma").value),
        )
        self.display_width = int(self.get_parameter("display_width").value)
        self.display_height = int(self.get_parameter("display_height").value)
        self.display_low_percentile = float(
            self.get_parameter("display_low_percentile").value
        )
        self.display_high_percentile = float(
            self.get_parameter("display_high_percentile").value
        )
        self.display_smoothing = float(self.get_parameter("display_smoothing").value)
        self.sharpen_amount = float(self.get_parameter("sharpen_amount").value)

        self.bridge = CvBridge()
        self.frame_count = 0
        self._display_gray = None
        self._relative_heat_streak = 0

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.annotated_pub = self.create_publisher(Image, self.annotated_topic, 10)
        self.results_pub = self.create_publisher(String, self.results_topic, 10)
        self.alarm_pub = self.create_publisher(Bool, self.alarm_topic, 10)

        self.get_logger().info(
            "Infrared detection started: "
            f"{self.image_topic} -> {self.annotated_topic}, "
            f"temperature threshold={self.temperature_threshold_c:.1f} C"
        )
        self.get_logger().info(
            "Relative heat hotspot detection: "
            f"enabled={self.relative_heat_alarm}, "
            f"percentile={self.relative_heat_percentile:.1f}, "
            f"min_score={self.relative_heat_min_score:.1f}, "
            f"min_contrast={self.relative_heat_min_contrast:.1f}, "
            f"confirmations={self.relative_heat_confirmations}"
        )
        self.get_logger().info(
            "Infrared overlay text reading is disabled; relative heat is "
            "diagnostic only."
        )
        if self.fallback_to_intensity_alarm:
            self.get_logger().warning(
                "fallback_to_intensity_alarm is ignored; realtime infrared "
                "alarms are handled by the inspection Qwen3 temperature worker."
            )

    def _find_relative_heat_regions(self, frame):
        height, width = frame.shape[:2]
        y1 = int(height * self.relative_heat_roi_top_fraction)
        y2 = int(height * self.relative_heat_roi_bottom_fraction)
        y1 = max(0, min(height - 1, y1))
        y2 = max(y1 + 1, min(height, y2))

        b, g, r = cv2.split(frame)
        heat = np.clip(
            0.60 * r.astype(np.float32)
            + 0.45 * g.astype(np.float32)
            - 0.35 * b.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
        heat = cv2.GaussianBlur(heat, (5, 5), 0)
        local = cv2.GaussianBlur(
            heat,
            (0, 0),
            self.relative_heat_local_sigma,
        )
        contrast = cv2.subtract(heat, local)

        roi_heat = heat[y1:y2, :]
        if roi_heat.size == 0:
            return [], {
                "roi": [0, 0, 0, 0],
                "threshold": None,
                "max_score": None,
                "max_contrast": None,
            }

        threshold = max(
            self.relative_heat_min_score,
            float(np.percentile(roi_heat, self.relative_heat_percentile)),
        )
        mask = (
            (heat.astype(np.float32) >= threshold)
            & (contrast.astype(np.float32) >= self.relative_heat_min_contrast)
        ).astype(np.uint8) * 255
        mask[:y1, :] = 0
        mask[y2:, :] = 0

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = float((y2 - y1) * width) * self.relative_heat_max_area_ratio
        regions = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.relative_heat_min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            heat_roi = heat[y:y + h, x:x + w]
            contrast_roi = contrast[y:y + h, x:x + w]
            if heat_roi.size == 0:
                continue
            regions.append({
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": round(area, 2),
                "max_intensity": int(heat_roi.max()),
                "mean_intensity": round(float(heat_roi.mean()), 2),
                "max_heat_score": int(heat_roi.max()),
                "mean_heat_score": round(float(heat_roi.mean()), 2),
                "max_contrast": int(contrast_roi.max()),
                "mean_contrast": round(float(contrast_roi.mean()), 2),
                "center": [int(x + w / 2), int(y + h / 2)],
            })

        regions.sort(
            key=lambda item: (
                item["max_heat_score"],
                item["max_contrast"],
                item["area"],
            ),
            reverse=True,
        )
        diagnostics = {
            "roi": [0, y1, int(width), y2],
            "threshold": round(float(threshold), 2),
            "max_score": int(roi_heat.max()),
            "mean_score": round(float(roi_heat.mean()), 2),
            "max_contrast": int(contrast[y1:y2, :].max()),
            "percentile": self.relative_heat_percentile,
            "min_score": self.relative_heat_min_score,
            "min_contrast": self.relative_heat_min_contrast,
            "min_area": self.relative_heat_min_area,
            "max_area_ratio": self.relative_heat_max_area_ratio,
        }
        return regions[:self.max_regions], diagnostics

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if self.display_smoothing > 0 and self._display_gray is not None:
                alpha = max(0.0, min(0.95, self.display_smoothing))
                display_gray = cv2.addWeighted(
                    display_gray,
                    1.0 - alpha,
                    self._display_gray,
                    alpha,
                    0,
                )
            self._display_gray = display_gray.copy()

            low_pct = max(0.0, min(99.0, self.display_low_percentile))
            high_pct = max(low_pct + 0.1, min(100.0, self.display_high_percentile))
            lo, hi = np.percentile(display_gray, [low_pct, high_pct])
            if hi <= lo:
                hi = lo + 1.0
            enhanced = np.clip((display_gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(
                blurred,
                self.intensity_threshold,
                255,
                cv2.THRESH_BINARY,
            )
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions = []
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y:y + h, x:x + w]
                max_intensity = int(roi.max()) if roi.size else 0
                mean_intensity = float(round(float(roi.mean()), 2)) if roi.size else 0.0
                regions.append({
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "area": round(area, 2),
                    "max_intensity": max_intensity,
                    "mean_intensity": mean_intensity,
                    "center": [int(x + w / 2), int(y + h / 2)],
                })

            regions.sort(key=lambda item: (item["max_intensity"], item["area"]), reverse=True)
            regions = regions[:self.max_regions]
            relative_regions, relative_heat = self._find_relative_heat_regions(frame)
            relative_heat_detected = bool(relative_regions)
            if self.relative_heat_alarm and relative_heat_detected:
                self._relative_heat_streak += 1
            else:
                self._relative_heat_streak = 0
            relative_heat_confirmed = (
                self.relative_heat_alarm
                and relative_heat_detected
                and self._relative_heat_streak >= self.relative_heat_confirmations
            )

            max_temperature_c = None
            temperature_available = False
            alarm = bool(relative_heat_confirmed)
            threshold_source = "relative_heat_video"
            alarm_basis = "relative_heat_video" if alarm else None
            result_regions = relative_regions if alarm else []
            result_count = len(result_regions) if alarm else 0
            if alarm:
                message = (
                    "检测到红外热斑诊断信号: "
                    f"{len(relative_regions)} 个热斑连续 "
                    f"{self._relative_heat_streak} 帧命中"
                )
            elif self.relative_heat_alarm and relative_heat_detected:
                message = (
                    "检测到红外热斑，正在确认: "
                    f"{self._relative_heat_streak}/"
                    f"{self.relative_heat_confirmations}"
                )
            else:
                message = "红外画面诊断正常"

            annotated = cv2.applyColorMap(enhanced, cv2.COLORMAP_TURBO)

            src_h, src_w = annotated.shape[:2]
            if self.display_width > 0 and self.display_height > 0:
                target_size = (self.display_width, self.display_height)
                interpolation = (
                    cv2.INTER_LANCZOS4
                    if self.display_width > src_w or self.display_height > src_h
                    else cv2.INTER_AREA
                )
                annotated = cv2.resize(annotated, target_size, interpolation=interpolation)

            if self.sharpen_amount > 0:
                blurred_display = cv2.GaussianBlur(annotated, (0, 0), 1.0)
                annotated = cv2.addWeighted(
                    annotated,
                    1.0 + self.sharpen_amount,
                    blurred_display,
                    -self.sharpen_amount,
                    0,
                )

            if result_regions:
                scale_x = annotated.shape[1] / float(src_w)
                scale_y = annotated.shape[0] / float(src_h)
                box_color = (0, 0, 255)
                label = "IR DIAGNOSTIC"
                for region in result_regions:
                    x1, y1, x2, y2 = region["bbox"]
                    pt1 = (int(x1 * scale_x), int(y1 * scale_y))
                    pt2 = (int(x2 * scale_x), int(y2 * scale_y))
                    cv2.rectangle(annotated, pt1, pt2, box_color, 2)
                    cv2.putText(
                        annotated,
                        label,
                        (pt1[0], max(16, pt1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        1,
                        cv2.LINE_AA,
                    )

            summary = {
                "status": "ok",
                "frame": self.frame_count,
                "count": result_count,
                "alarm": alarm,
                "temperature_available": temperature_available,
                "temperature_source": "uvc_relative_heat_video",
                "temperature_threshold_c": self.temperature_threshold_c,
                "max_temperature_c": None,
                "message": message,
                "threshold": self.intensity_threshold,
                "max_intensity": max([r["max_intensity"] for r in regions], default=int(gray.max())),
                "mean_intensity": round(float(gray.mean()), 2),
                "relative_heat_alarm": self.relative_heat_alarm,
                "relative_heat_detected": relative_heat_detected,
                "relative_heat_streak": self._relative_heat_streak,
                "relative_heat_confirmations": self.relative_heat_confirmations,
                "relative_hotspot_count": len(relative_regions),
                "relative_heat": relative_heat,
                "display": "percentile_turbo_smooth",
                "display_size": [int(annotated.shape[1]), int(annotated.shape[0])],
                "threshold_source": threshold_source,
                "alarm_basis": alarm_basis,
                "fallback_to_intensity_alarm": False,
                "regions": result_regions,
                "intensity_hotspot_count": len(regions),
            }

            result_msg = String()
            result_msg.data = json.dumps(summary, ensure_ascii=False)
            self.results_pub.publish(result_msg)

            alarm_msg = Bool()
            alarm_msg.data = summary["alarm"]
            self.alarm_pub.publish(alarm_msg)

            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f"Processed {self.frame_count} infrared frames, "
                    f"alarm={alarm}, "
                    f"relative_hotspots={len(relative_regions)}, "
                    f"intensity_hotspots={len(regions)}"
                )

        except Exception as exc:
            self.get_logger().error(f"Infrared detection failed: {exc}")

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = InfraredDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
