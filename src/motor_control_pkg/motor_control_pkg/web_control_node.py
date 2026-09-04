#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web控制节点 - 稳定推流 + 直接大模型助手版

功能：
1. 提供 8010 网页服务
2. 订阅 ROS2 图像话题，默认 /detection/annotated_image
3. 将 ROS 图像提前编码为 JPEG，稳定输出 /video_feed
4. 提供电机控制接口：/forward /reverse /stop /status
5. 提供智能助手接口：/api/claw/command
   - 不再走 OpenClaw Gateway
   - 直接读取 ~/.openclaw/openclaw.json 里的模型配置
   - 直接调用 https://cursor.scihub.edu.kg/api/v1/chat/completions
6. 保留前端兼容接口名 /api/claw/command，前端不用大改
"""

import os
import json
import time
import threading
import subprocess
import shutil
import shlex
import signal
import uuid
import re
import hashlib
import importlib.util
import urllib.request
import urllib.error
import base64
import concurrent.futures
from urllib.parse import quote
import zipfile
import io
from datetime import datetime, timedelta

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from flask import Flask, Response, render_template_string, jsonify, request, send_from_directory, send_file
from werkzeug.utils import secure_filename


app = Flask(__name__)


BASE_DIR = os.getenv("CABLE_WORKSPACE_DIR", "/home/sunrise/cable1")
HTML_PATH = os.path.join(BASE_DIR, "agentic_aiops_clean_sidebar.html")
FP_SAMPLE_DIR = os.path.join(BASE_DIR, "false_positive_samples")
REVIEW_POOL_DIR = os.path.join(BASE_DIR, "review_sample_pool")
REVIEW_POOL_IMAGES_DIR = os.path.join(REVIEW_POOL_DIR, "images")
REVIEW_POOL_LABELS_DIR = os.path.join(REVIEW_POOL_DIR, "labels")
REVIEW_POOL_META = os.path.join(REVIEW_POOL_DIR, "sample_pool.json")
LOOP_AUDIT_JSONL = os.path.join(REVIEW_POOL_DIR, "loop_audit.jsonl")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
AD_ANALYZER_DIR = os.path.join(BASE_DIR, "ad5933_cable_analyzer")
AD_RUNNER_SCRIPT = os.path.join(AD_ANALYZER_DIR, "web_runner.py")
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "scripts", "train_to_bin_pipeline.py")
DEPLOY_DIR = os.path.join(MODELS_DIR, "deployed")
CURRENT_MODEL_JSON = os.path.join(BASE_DIR, "config", "current_model.json")
MODEL_HISTORY_JSONL = os.path.join(BASE_DIR, "config", "model_history.jsonl")
APPLIED_MODEL_BIN = os.path.join(DEPLOY_DIR, "current.bin")
DETECTION_LOG = "/tmp/cable1_detection_node.log"
INSPECTION_DIR = os.path.join(BASE_DIR, "inspection_reports")
INSPECTION_FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
INSPECTION_CAPTURE_MIN_INTERVAL_SECONDS = 3.0
VISUAL_REPORT_MERGE_DISTANCE_MM = 120.0
INFRARED_REPORT_MERGE_DISTANCE_MM = 60.0
INFRARED_AI_SAMPLE_INTERVAL_SECONDS = max(
    0.1,
    float(os.getenv("INFRARED_AI_SAMPLE_INTERVAL_SECONDS", "0.5")),
)
INFRARED_AI_MAX_INFLIGHT_REQUESTS = max(
    1,
    int(os.getenv("INFRARED_AI_MAX_INFLIGHT_REQUESTS", "4")),
)
INFRARED_AI_TEMPERATURE_THRESHOLD_C = 70.0
DASHSCOPE_API_KEY_PATH = os.path.join(BASE_DIR, ".dashscope_key")
QWEN_VISION_BASE_URL = os.getenv(
    "QWEN_VISION_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_VISION_MODEL = os.getenv("QWEN_VISION_MODEL", "qwen3-vl-plus")

pipeline_state = {
    "running": False,
    "stage": "idle",
    "status": "idle",
    "message": "",
    "dataset": "",
    "model": "",
    "log_path": "",
    "started_at": None,
    "finished_at": None,
}
pipeline_lock = threading.Lock()
ad_state_lock = threading.Lock()
latest_ad_status = {}
latest_ad_result = {}
latest_ad_result_time = 0.0
fusion_lock = threading.RLock()
fusion_tracker = None
review_pool_lock = threading.Lock()
inspection_lock = threading.RLock()
inspection_stop_event = threading.Event()
inspection_impedance_ready = threading.Event()
inspection_state = {
    "status": "idle",
    "session_id": "",
    "message": "尚未开始检测",
    "started_at": None,
    "finished_at": None,
    "start_position_mm": None,
    "end_position_mm": None,
    "target_position_mm": None,
    "direction": "",
    "defects": [],
    "infrared_ai_samples": [],
    "impedance_samples": [],
    "report_path": "",
    "report_url": "",
    "error": "",
}

AD_ACTIONS = {
    "raw",
    "calibrate_low",
    "calibrate_cap",
    "calibrate_stray",
    "calibrate_normal",
    "set_profile",
    "analyze",
    "check_low",
    "locate_open",
    "cap_diagnose",
    "calibrate_moisture_baseline",
    "detect_moisture",
    "moisture_diagnose",
    "status",
}


# =========================
# 大模型配置读取
# =========================

OPENCLAW_CONFIG_PATH = os.getenv(
    "OPENCLAW_CONFIG_PATH",
    "/home/sunrise/.openclaw/openclaw.json"
)


def _read_cpu_times():
    with open("/proc/stat", "r", encoding="utf-8") as f:
        parts = f.readline().split()[1:]
    nums = [int(x) for x in parts]
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
    total = sum(nums)
    return total, idle


def get_cpu_percent():
    try:
        total1, idle1 = _read_cpu_times()
        time.sleep(0.08)
        total2, idle2 = _read_cpu_times()
        total_delta = total2 - total1
        idle_delta = idle2 - idle1
        if total_delta <= 0:
            return None
        return round((1.0 - idle_delta / total_delta) * 100, 1)
    except Exception:
        return None


def get_memory_info():
    try:
        data = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, val = line.split(":", 1)
                data[key] = int(val.strip().split()[0]) * 1024
        total = data.get("MemTotal", 0)
        available = data.get("MemAvailable", 0)
        used = max(0, total - available)
        percent = round((used / total) * 100, 1) if total else None
        return {
            "total": total,
            "available": available,
            "used": used,
            "percent": percent
        }
    except Exception:
        return {
            "total": 0,
            "available": 0,
            "used": 0,
            "percent": None
        }


def get_disk_info(path=BASE_DIR):
    try:
        st = os.statvfs(path)
        total = st.f_blocks * st.f_frsize
        available = st.f_bavail * st.f_frsize
        used = total - available
        percent = round((used / total) * 100, 1) if total else None
        return {
            "path": path,
            "total": total,
            "available": available,
            "used": used,
            "percent": percent
        }
    except Exception:
        return {
            "path": path,
            "total": 0,
            "available": 0,
            "used": 0,
            "percent": None
        }


def get_temperature_c():
    temps = []
    base = "/sys/class/thermal"
    try:
        for name in os.listdir(base):
            if not name.startswith("thermal_zone"):
                continue
            path = os.path.join(base, name, "temp")
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if raw:
                temps.append(int(raw) / 1000.0)
    except Exception:
        pass

    if not temps:
        return None

    return round(max(temps), 1)


def get_uptime_info():
    try:
        with open("/proc/uptime", "r", encoding="utf-8") as f:
            seconds = float(f.read().split()[0])
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        if days:
            text = f"{days}d {hours}h {minutes}m"
        elif hours:
            text = f"{hours}h {minutes}m"
        else:
            text = f"{minutes}m"
        return {
            "seconds": round(seconds, 1),
            "text": text,
        }
    except Exception:
        return {
            "seconds": None,
            "text": "unreported",
        }


def run_ad5933_action(action, payload=None, timeout=90):
    """Run AD5933 analyzer actions in an isolated process."""
    payload = payload or {}
    if action not in AD_ACTIONS:
        return {
            "ok": False,
            "error": f"unsupported action: {action}",
        }, 400

    if not os.path.exists(AD_RUNNER_SCRIPT):
        return {
            "ok": False,
            "error": "AD5933 analyzer is not installed",
            "path": AD_RUNNER_SCRIPT,
        }, 404

    cmd = ["python3", AD_RUNNER_SCRIPT, action]
    if payload.get("freq") not in (None, ""):
        cmd += ["--freq", str(int(payload["freq"]))]
    if payload.get("resistance") not in (None, ""):
        cmd += ["--resistance", str(float(payload["resistance"]))]
    if payload.get("pf_per_m") not in (None, ""):
        cmd += ["--pf-per-m", str(float(payload["pf_per_m"]))]
    if payload.get("ref_temp") not in (None, ""):
        cmd += ["--ref-temp", str(float(payload["ref_temp"]))]
    if payload.get("temp_coeff") not in (None, ""):
        cmd += ["--temp-coeff", str(float(payload["temp_coeff"]))]
    if payload.get("count") not in (None, ""):
        cmd += ["--count", str(int(payload["count"]))]
    if payload.get("cable_length_m") not in (None, ""):
        cmd += ["--cable-length-m", str(float(payload["cable_length_m"]))]
    if payload.get("length_tolerance_m") not in (None, ""):
        cmd += ["--length-tolerance-m", str(float(payload["length_tolerance_m"]))]

    env = os.environ.copy()
    env["PYTHONPATH"] = AD_ANALYZER_DIR + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            cmd,
            cwd=AD_ANALYZER_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"AD5933 action timed out after {timeout}s",
            "action": action,
        }, 504
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "action": action,
        }, 500

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    try:
        data = json.loads(stdout.splitlines()[-1]) if stdout else {}
    except Exception:
        data = {
            "ok": False,
            "error": "AD5933 runner returned non-JSON output",
            "stdout": stdout,
        }

    if stderr:
        data["stderr"] = stderr
    data.setdefault("action", action)
    data.setdefault("returncode", result.returncode)

    if result.returncode != 0 or data.get("ok") is False:
        return data, 500
    return data, 200


def get_bpu_info():
    info = {
        "percent": None,
        "temperature_c": None,
        "cur_freq_mhz": None,
        "min_freq_mhz": None,
        "max_freq_mhz": None,
        "source": "unreported",
    }

    try:
        result = subprocess.run(
            ["hrut_somstatus"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        output = result.stdout + "\n" + result.stderr
        if result.returncode == 0 and output.strip():
            info["source"] = "hrut_somstatus"

            temp_match = re.search(r"\bBPU\s*:\s*([0-9.]+)\s*\(C\)", output)
            if temp_match:
                info["temperature_c"] = round(float(temp_match.group(1)), 1)

            row_match = re.search(
                r"\bbpu0:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
                output,
                flags=re.IGNORECASE,
            )
            if row_match:
                info["min_freq_mhz"] = round(float(row_match.group(1)), 1)
                info["cur_freq_mhz"] = round(float(row_match.group(2)), 1)
                info["max_freq_mhz"] = round(float(row_match.group(3)), 1)
                info["percent"] = round(float(row_match.group(4)), 1)
    except Exception:
        pass

    if info["cur_freq_mhz"] is None:
        for path in (
            "/sys/class/devfreq/3a000000.bpu/cur_freq",
            "/sys/devices/platform/soc/3a000000.bpu/devfreq/3a000000.bpu/cur_freq",
        ):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                if raw:
                    info["cur_freq_mhz"] = round(float(raw) / 1000000.0, 1)
                    if info["source"] == "unreported":
                        info["source"] = path
                    break
            except Exception:
                continue

    return info


ROS_GRAPH_CACHE = {
    "nodes": [],
    "updated_at": 0.0,
    "error": "",
}
ROS_NODE_LAST_SEEN = {}
ROS_OFFLINE_GRACE_SEC = 15.0


def get_ros_graph_snapshot():
    try:
        result = subprocess.run(
            [
                "bash",
                "-lc",
                "source /opt/ros/humble/setup.bash && "
                f"source {shlex.quote(BASE_DIR)}/install/setup.bash 2>/dev/null || true; "
                "ros2 node list"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            nodes = [x.strip() for x in result.stdout.splitlines() if x.strip()]
            if nodes:
                ROS_GRAPH_CACHE.update({
                    "nodes": nodes,
                    "updated_at": time.time(),
                    "error": "",
                })
                return {
                    "nodes": nodes,
                    "source": "ros2",
                    "stale": False,
                    "age_sec": 0.0,
                    "error": "",
                }
        error = (result.stderr or "").strip() or "ros2 node list returned no nodes"
    except Exception as exc:
        error = str(exc)

    cached_nodes = ROS_GRAPH_CACHE.get("nodes") or []
    cached_at = float(ROS_GRAPH_CACHE.get("updated_at") or 0.0)
    if cached_nodes and cached_at:
        return {
            "nodes": list(cached_nodes),
            "source": "cached",
            "stale": True,
            "age_sec": round(time.time() - cached_at, 1),
            "error": error,
        }

    return {
        "nodes": [],
        "source": "unavailable",
        "stale": True,
        "age_sec": None,
        "error": error,
    }


EXPECTED_ROS_NODES = [
    {
        "name": "camera_node",
        "topic": "/camera/image_raw",
        "role": "采集摄像头画面",
    },
    {
        "name": "detection_node",
        "topic": "/detection/results",
        "role": "BPU 模型推理",
    },
    {
        "name": "infrared_camera_node",
        "topic": "/infrared/image_raw",
        "role": "红外摄像头采集",
    },
    {
        "name": "infrared_detection_node",
        "topic": "/infrared/results",
        "role": "红外异常检测",
    },
    {
        "name": "motor_control_node",
        "topic": "/motor/control",
        "role": "电机控制",
    },
    {
        "name": "web_control_node",
        "topic": "/video_feed",
        "role": "网页控制台",
    },
]


def get_node_process_counts():
    counts = {}
    expected = {item["name"] for item in EXPECTED_ROS_NODES}
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            path = os.path.join("/proc", name, "cmdline")
            try:
                with open(path, "rb") as f:
                    raw = f.read()
            except OSError:
                continue
            if not raw:
                continue
            cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
            for node_name in expected:
                if node_name in cmd:
                    counts[node_name] = counts.get(node_name, 0) + 1
    except Exception:
        pass
    return counts


def get_ros_node_status(ros_nodes, graph_source="ros2"):
    normalized = []
    for item in ros_nodes:
        name = item.strip().lstrip("/")
        if name:
            normalized.append(name)

    counts = {}
    for name in normalized:
        counts[name] = counts.get(name, 0) + 1

    now = time.time()
    proc_counts = get_node_process_counts()
    rows = []
    expected_names = set()
    for item in EXPECTED_ROS_NODES:
        name = item["name"]
        expected_names.add(name)
        count = counts.get(name, 0)
        source = graph_source
        stale = graph_source != "ros2"
        if count >= 1:
            state = "online"
            ROS_NODE_LAST_SEEN[name] = now
        elif proc_counts.get(name, 0) > 0:
            state = "online"
            count = 1
            source = "process"
            stale = True
        elif now - ROS_NODE_LAST_SEEN.get(name, 0.0) <= ROS_OFFLINE_GRACE_SEC:
            state = "online"
            count = 1
            source = "grace"
            stale = True
        else:
            state = "offline"
        rows.append({
            "name": name,
            "topic": item["topic"],
            "role": item["role"],
            "state": state,
            "count": count,
            "graph_count": counts.get(name, 0),
            "process_count": proc_counts.get(name, 0),
            "source": source,
            "stale": stale,
        })

    for name in sorted(set(normalized) - expected_names):
        ROS_NODE_LAST_SEEN[name] = now
        rows.append({
            "name": name,
            "topic": "",
            "role": "ROS2 graph node",
            "state": "online",
            "count": counts.get(name, 0),
            "graph_count": counts.get(name, 0),
            "process_count": proc_counts.get(name, 0),
            "source": graph_source,
            "stale": graph_source != "ros2",
        })

    return rows


def count_files(root, exts):
    total = 0
    if not os.path.exists(root):
        return 0
    for _, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(exts):
                total += 1
    return total


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
LABEL_EXTS = (".txt",)
DATASET_ROOT_MARKERS = ("data.yaml", "data.yml")
ARCHIVE_EXTS = (".zip", ".rar", ".7z")


def find_dataset_yaml(path):
    for name in DATASET_ROOT_MARKERS:
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            return candidate
    return ""


def split_image_count(path, split):
    return count_files(os.path.join(path, split, "images"), IMAGE_EXTS)


def looks_like_yolo_dataset(path):
    if not os.path.isdir(path):
        return False
    if find_dataset_yaml(path):
        return True
    if count_files(os.path.join(path, "images"), IMAGE_EXTS) > 0:
        return True
    return any(split_image_count(path, split) > 0 for split in ("train", "valid", "test"))


def dataset_summary(path):
    name = os.path.basename(path.rstrip("/"))
    data_yaml = find_dataset_yaml(path)
    train_images = split_image_count(path, "train")
    val_images = split_image_count(path, "valid")
    test_images = split_image_count(path, "test")
    if not (train_images or val_images or test_images):
        train_images = count_files(os.path.join(path, "images"), IMAGE_EXTS)
    labels = count_files(path, LABEL_EXTS)
    best_pt = os.path.join(path, "runs", "detect", "train", "weights", "best.pt")
    yolo26n = os.path.join(path, "yolo26n.pt")
    images = train_images + val_images + test_images

    mtime = os.path.getmtime(path)
    return {
        "name": name,
        "path": path,
        "data_yaml": data_yaml,
        "train_images": train_images,
        "val_images": val_images,
        "test_images": test_images,
        "images": images,
        "labels": labels,
        "valid_yolo_dataset": bool(images or os.path.exists(data_yaml)),
        "can_receive_false_positive": bool(os.path.isdir(path)),
        "has_best_pt": os.path.exists(best_pt),
        "best_pt": best_pt if os.path.exists(best_pt) else "",
        "has_yolo26n": os.path.exists(yolo26n),
        "yolo26n": yolo26n if os.path.exists(yolo26n) else "",
        "updated_at": datetime.fromtimestamp(mtime).isoformat(),
    }


def list_datasets():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    items = []
    for name in sorted(os.listdir(DATASETS_DIR)):
        path = os.path.join(DATASETS_DIR, name)
        if os.path.isdir(path):
            items.append(dataset_summary(path))
    return items


def dataset_totals():
    datasets = list_datasets()
    totals = {
        "count": len(datasets),
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "images": 0,
        "labels": 0,
    }
    for item in datasets:
        for key in ("train_images", "val_images", "test_images", "images", "labels"):
            totals[key] += int(item.get(key) or 0)
    return totals


def read_yolo_annotations(label_path, class_names):
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception:
        return annotations

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = [
                max(0.0, min(1.0, float(value)))
                for value in parts[1:]
            ]
        except Exception:
            continue
        class_name = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        annotations.append({
            "class_id": class_id,
            "class_name": class_name,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        })
    return annotations


def list_dataset_samples(dataset_name, split_filter="all", annotation_filter="all",
                         query="", page=1, page_size=24):
    dataset_path, error = validate_dataset_path(dataset_name)
    if error:
        return None, error

    split_filter = str(split_filter or "all").strip().lower()
    if split_filter in ("val", "validation"):
        split_filter = "valid"
    splits = ("train", "valid", "test")
    selected_splits = splits if split_filter == "all" else (split_filter,)
    if any(split not in splits for split in selected_splits):
        return None, "Invalid dataset split."

    annotation_filter = str(annotation_filter or "all").strip().lower()
    if annotation_filter not in ("all", "annotated", "unannotated", "negative"):
        annotation_filter = "all"
    query = str(query or "").strip().lower()
    class_names = read_dataset_classes(dataset_path)
    samples = []

    for split in selected_splits:
        images_dir = os.path.join(dataset_path, split, "images")
        labels_dir = os.path.join(dataset_path, split, "labels")
        if not os.path.isdir(images_dir):
            continue
        for filename in sorted(os.listdir(images_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            stem = os.path.splitext(filename)[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            has_label_file = os.path.isfile(label_path)
            annotations = read_yolo_annotations(label_path, class_names)
            is_negative = has_label_file and not annotations
            is_annotated = has_label_file
            class_labels = sorted({
                item["class_name"] for item in annotations
            })

            if annotation_filter == "annotated" and not is_annotated:
                continue
            if annotation_filter == "unannotated" and is_annotated:
                continue
            if annotation_filter == "negative" and not is_negative:
                continue
            searchable = " ".join([filename, split, *class_labels]).lower()
            if query and query not in searchable:
                continue

            encoded_dataset = quote(safe_dataset_name(dataset_name), safe="")
            encoded_split = quote(split, safe="")
            encoded_filename = quote(filename, safe="")
            samples.append({
                "id": f"{split}:{filename}",
                "filename": filename,
                "split": split,
                "label_filename": stem + ".txt" if has_label_file else "",
                "has_label_file": has_label_file,
                "is_negative": is_negative,
                "annotation_count": len(annotations),
                "classes": class_labels,
                "annotations": annotations,
                "preview_url": (
                    f"/api/datasets/{encoded_dataset}/preview/"
                    f"{encoded_split}/{encoded_filename}"
                ),
            })

    page_size = max(1, min(60, int(page_size or 24)))
    page = max(1, int(page or 1))
    total = len(samples)
    pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, pages)
    start = (page - 1) * page_size
    summary = dataset_summary(dataset_path)
    return {
        "dataset": summary,
        "classes": class_names,
        "samples": samples[start:start + page_size],
        "filters": {
            "split": split_filter,
            "annotation": annotation_filter,
            "query": query,
        },
        "pagination": {
            "page": page,
            "page_size": page_size,
            "pages": pages,
            "total": total,
        },
    }, ""


def false_positive_summary(limit=8):
    meta_path = os.path.join(FP_SAMPLE_DIR, "metadata.jsonl")
    today = datetime.now().date().isoformat()
    entries = []
    today_count = 0
    reason_counts = {}
    dataset_counts = {}

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                entries.append(entry)
                ts = str(entry.get("time", ""))
                if ts.startswith(today):
                    today_count += 1

                request_data = entry.get("request") or {}
                reason = str(request_data.get("reason") or "未填写").strip() or "未填写"
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                dataset_name = entry.get("dataset") or request_data.get("dataset") or ""
                split = entry.get("split") or request_data.get("split") or ""
                if dataset_name:
                    key = f"{dataset_name}:{split or 'all'}"
                    dataset_counts[key] = dataset_counts.get(key, 0) + 1

    recent = []
    for item in reversed(entries[-limit:]):
        request_data = item.get("request") or {}
        recent.append({
            "sample_id": item.get("sample_id", ""),
            "time": item.get("time", ""),
            "image_path": item.get("image_path", ""),
            "dataset": item.get("dataset", request_data.get("dataset", "")),
            "dataset_path": item.get("dataset_path", ""),
            "dataset_image_path": item.get("dataset_image_path", ""),
            "dataset_label_path": item.get("dataset_label_path", ""),
            "reason": request_data.get("reason", ""),
            "split": request_data.get("split", ""),
            "mode": request_data.get("mode", ""),
            "prediction": request_data.get("pred", ""),
        })

    return {
        "count": len(entries),
        "today": today_count,
        "metadata_path": meta_path if os.path.exists(meta_path) else "",
        "recent": recent,
        "dataset_counts": dataset_counts,
        "reason_counts": sorted(
            [{"reason": k, "count": v} for k, v in reason_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:8],
    }


def ensure_review_pool_dirs():
    os.makedirs(REVIEW_POOL_IMAGES_DIR, exist_ok=True)
    os.makedirs(REVIEW_POOL_LABELS_DIR, exist_ok=True)


def read_review_pool():
    ensure_review_pool_dirs()
    if not os.path.exists(REVIEW_POOL_META):
        return []
    try:
        with open(REVIEW_POOL_META, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def write_review_pool(items):
    ensure_review_pool_dirs()
    tmp = REVIEW_POOL_META + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, REVIEW_POOL_META)


def append_loop_audit(event, payload=None):
    ensure_review_pool_dirs()
    row = {
        "time": iso_now(),
        "event": event,
        "payload": payload or {},
    }
    with open(LOOP_AUDIT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def read_loop_audit(limit=30):
    if not os.path.exists(LOOP_AUDIT_JSONL):
        return []
    rows = []
    with open(LOOP_AUDIT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return list(reversed(rows[-limit:]))


def review_pool_summary(items=None):
    rows = items if items is not None else read_review_pool()
    total = len(rows)
    annotated = sum(1 for x in rows if x.get("annotated"))
    imported = sum(1 for x in rows if x.get("status") == "imported")
    pending = max(0, total - imported)
    return {
        "total": total,
        "pending": pending,
        "annotated": annotated,
        "imported": imported,
    }


def make_sample_record(sample_id, image_path, source, label_path="", **extra):
    record = {
        "sample_id": sample_id,
        "time": iso_now(),
        "image_path": image_path,
        "label_path": label_path,
        "source": source,
        "status": "annotated" if label_path else "pending",
        "annotated": bool(label_path),
        "defect_class": "",
        "note": "",
        "dataset": "",
        "split": "",
    }
    record.update(extra)
    return record


def image_url_for_path(path):
    if not path:
        return ""
    try:
        rel = os.path.relpath(path, REVIEW_POOL_DIR)
    except ValueError:
        return ""
    return "/api/review_pool/file/" + rel.replace(os.sep, "/")


def read_dataset_classes(dataset_path):
    data_yaml = find_dataset_yaml(dataset_path)
    names = []
    if data_yaml and os.path.exists(data_yaml):
        try:
            raw = open(data_yaml, "r", encoding="utf-8").read().splitlines()
            in_names = False
            for line in raw:
                stripped = line.strip()
                if stripped.startswith("names:"):
                    tail = stripped.split(":", 1)[1].strip()
                    if tail.startswith("[") and tail.endswith("]"):
                        names = [x.strip().strip("'\"") for x in tail[1:-1].split(",") if x.strip()]
                        break
                    in_names = True
                    continue
                if in_names:
                    if not line.startswith((" ", "\t", "-")):
                        break
                    item = stripped.lstrip("-").strip()
                    if ":" in item:
                        item = item.split(":", 1)[1].strip()
                    if item:
                        names.append(item.strip("'\""))
        except Exception:
            names = []
    if not names:
        names = ["defect"]
    return names


def write_dataset_yaml(dataset_path, names):
    names = [str(x).strip() for x in names if str(x).strip()] or ["defect"]
    data_yaml = os.path.join(dataset_path, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("path: " + dataset_path + "\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n")
        f.write("nc: " + str(len(names)) + "\n")
        f.write("names:\n")
        for idx, name in enumerate(names):
            f.write(f"  {idx}: {name}\n")
    return data_yaml


def ensure_dataset_class(dataset_path, class_name):
    names = read_dataset_classes(dataset_path)
    class_name = str(class_name or "").strip()
    if class_name and class_name not in names:
        names.append(class_name)
        write_dataset_yaml(dataset_path, names)
    elif not os.path.exists(os.path.join(dataset_path, "data.yaml")):
        write_dataset_yaml(dataset_path, names)
    if class_name and class_name in names:
        return names.index(class_name), names
    return 0, names


def normalize_label_text(label_text):
    lines = []
    for line in str(label_text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [max(0.0, min(1.0, float(x))) for x in parts[1:]]
        except Exception:
            continue
        lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(cls, *vals))
    return "\n".join(lines)


def class_ids_from_label_text(label_text):
    ids = []
    for line in str(label_text or "").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            ids.append(int(float(parts[0])))
        except Exception:
            continue
    return sorted(set(ids))


def rewrite_label_class(label_text, class_id):
    lines = []
    try:
        target_cls = int(class_id)
    except Exception:
        target_cls = 0
    for line in normalize_label_text(label_text).splitlines():
        parts = line.split()
        if len(parts) == 5:
            parts[0] = str(target_cls)
            lines.append(" ".join(parts))
    return "\n".join(lines)


def labels_by_uploaded_stem(files):
    labels = {}
    for item in files:
        raw = item.filename or ""
        base = os.path.basename(raw.replace("\\", "/"))
        safe_base = secure_filename(base)
        stem, ext = os.path.splitext(safe_base)
        if ext.lower() not in LABEL_EXTS:
            continue
        labels[stem] = item.read().decode("utf-8", errors="replace")
    return labels


def iso_now():
    return datetime.now().isoformat(timespec="seconds")


def load_json_file(path, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def model_metadata_path(model_path):
    return model_path + ".json"


def read_model_metadata(model_path):
    return load_json_file(model_metadata_path(model_path), {})


def model_display_identity(model_path, metadata=None):
    metadata = metadata or {}
    source_path = (
        metadata.get("source_model_path")
        or metadata.get("source_path")
        or metadata.get("board_model_path")
        or metadata.get("model_path")
        or model_path
    )
    display_name = metadata.get("model_name") or os.path.basename(source_path) or os.path.basename(model_path)
    if display_name == os.path.basename(APPLIED_MODEL_BIN) and source_path:
        display_name = os.path.basename(source_path)

    return {
        "display_name": display_name,
        "display_path": source_path or model_path,
        "runtime_name": os.path.basename(model_path) if model_path else "",
        "runtime_path": model_path or "",
        "model_role": "vision_detection",
    }


def write_model_metadata(model_path, metadata, replace=False):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    meta = {} if replace else read_model_metadata(model_path)
    meta.update(metadata or {})
    meta["model_path"] = model_path
    meta.setdefault("model_name", os.path.basename(model_path))
    meta.setdefault("trained_at", datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat())
    meta["metadata_updated_at"] = iso_now()
    with open(model_metadata_path(model_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def model_sha256(model_path):
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_bpu_model_info(model_path):
    try:
        proc = subprocess.run(
            ["hrt_model_exec", "model_info", "--model_file", model_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return {"error": str(exc)}

    text = proc.stdout or ""
    shapes = []
    in_output_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("output["):
            in_output_block = True
            continue
        if stripped.startswith("input["):
            in_output_block = False
            continue
        if not in_output_block:
            continue
        match = re.search(r"valid shape:\s*\(([^)]*)\)", stripped)
        if not match:
            continue
        nums = []
        for item in match.group(1).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                nums.append(int(item))
            except ValueError:
                nums = []
                break
        if nums:
            shapes.append(nums)

    class_count = 0
    for shape in shapes:
        if len(shape) < 4:
            continue
        last_dim = int(shape[-1])
        if last_dim != 4:
            class_count = max(class_count, last_dim)

    return {
        "output_shapes": shapes,
        "class_count": class_count,
        "raw_model_info": text[-4000:],
    }


def default_class_names_for_count(class_count):
    if class_count == 1:
        return ["white"]
    if class_count == 2:
        return ["expored_core", "white"]
    return [f"class_{idx}" for idx in range(class_count)]


def normalize_class_names(names):
    if isinstance(names, str):
        names = [item.strip() for item in names.split(",") if item.strip()]
    if not isinstance(names, list):
        return []
    return [str(item).strip() for item in names if str(item).strip()]


def reconcile_model_metadata(model_path, metadata=None):
    meta = dict(metadata or {})
    try:
        meta["sha256"] = model_sha256(model_path)
    except Exception:
        pass

    info = infer_bpu_model_info(model_path)
    if info.get("output_shapes"):
        meta["output_shapes"] = info["output_shapes"]
        meta["class_count"] = int(info.get("class_count") or 0)
        names = normalize_class_names(meta.get("class_names"))
        if meta["class_count"] > 0 and len(names) != meta["class_count"]:
            names = default_class_names_for_count(meta["class_count"])
            meta["class_source"] = "inferred_from_bpu_model_info"
        elif names:
            meta["class_source"] = meta.get("class_source") or "metadata"
        if names:
            meta["class_names"] = names
    else:
        meta["model_info_error"] = info.get("error") or "No output shape parsed from hrt_model_exec."
    return meta


def copy_model_metadata(src, dst, updates=None):
    meta = read_model_metadata(src)
    if not meta:
        meta = {
            "model_id": os.path.splitext(os.path.basename(src))[0],
            "model_name": os.path.basename(src),
            "source_model_path": src,
            "trained_at": datetime.fromtimestamp(os.path.getmtime(src)).isoformat(),
        }
    meta["source_model_path"] = meta.get("source_model_path") or src
    meta["source_path"] = src
    if updates:
        meta.update(updates)
    meta = reconcile_model_metadata(src, meta)
    return write_model_metadata(dst, meta, replace=True)


def append_model_history(event, model_path, metadata=None, message=""):
    metadata = metadata or read_model_metadata(model_path)
    entry = {
        "event": event,
        "event_at": iso_now(),
        "model_path": model_path,
        "model_name": os.path.basename(model_path),
        "message": message,
    }
    for key in (
        "model_id",
        "dataset",
        "epochs",
        "imgsz",
        "batch",
        "training_mode",
        "training_started_at",
        "training_finished_at",
        "trained_at",
        "uploaded_at",
        "deployed_at",
        "applied_at",
        "source_model_path",
        "source_path",
    ):
        if key in metadata:
            entry[key] = metadata[key]

    os.makedirs(os.path.dirname(MODEL_HISTORY_JSONL), exist_ok=True)
    with open(MODEL_HISTORY_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def list_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    current = load_json_file(CURRENT_MODEL_JSON, {})
    staged_path = os.path.abspath(current.get("model_path", ""))
    applied_path = os.path.abspath(current.get("applied_model_path", ""))
    items = []
    for root, _, files in os.walk(MODELS_DIR):
        for name in files:
            if not name.lower().endswith(".bin"):
                continue
            path = os.path.join(root, name)
            abs_path = os.path.abspath(path)
            st = os.stat(path)
            meta = read_model_metadata(path)
            identity = model_display_identity(path, meta)
            trained_at = (
                meta.get("trained_at")
                or meta.get("training_finished_at")
                or datetime.fromtimestamp(st.st_mtime).isoformat()
            )
            items.append({
                "name": name,
                "path": path,
                "size": st.st_size,
                "updated_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                "trained_at": trained_at,
                "dataset": meta.get("dataset", ""),
                "epochs": meta.get("epochs", ""),
                "imgsz": meta.get("imgsz", ""),
                "batch": meta.get("batch", ""),
                "training_mode": meta.get("training_mode", ""),
                "model_id": meta.get("model_id", os.path.splitext(name)[0]),
                "deployed": path.startswith(DEPLOY_DIR),
                "staged": abs_path == staged_path,
                "applied": abs_path == applied_path or abs_path == os.path.abspath(APPLIED_MODEL_BIN),
                "display_name": identity["display_name"],
                "display_path": identity["display_path"],
                "runtime_name": identity["runtime_name"],
                "runtime_path": identity["runtime_path"],
                "model_role": identity["model_role"],
                "metadata": meta,
            })
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items


def list_model_history(limit=100):
    entries = []
    if os.path.exists(MODEL_HISTORY_JSONL):
        with open(MODEL_HISTORY_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue

    recorded = {
        (item.get("event"), item.get("model_path"))
        for item in entries
    }
    for model in list_models():
        key = ("model_file", model["path"])
        if key in recorded or ("trained_upload", model["path"]) in recorded:
            continue
        entries.append({
            "event": "model_file",
            "event_at": model.get("trained_at") or model.get("updated_at"),
            "model_path": model["path"],
            "model_name": model["name"],
            "model_id": model.get("model_id", ""),
            "dataset": model.get("dataset", ""),
            "training_mode": model.get("training_mode", ""),
            "trained_at": model.get("trained_at", ""),
            "message": "Existing model file found.",
        })

    entries.sort(key=lambda x: x.get("event_at", ""), reverse=True)
    return entries[:limit]


def current_model_summary():
    current = load_json_file(CURRENT_MODEL_JSON, {})
    applied = current.get("applied_model_path") or current.get("model_path") or ""
    if applied and os.path.exists(applied):
        st = os.stat(applied)
        metadata = read_model_metadata(applied)
        identity = model_display_identity(applied, metadata)
        return {
            "path": applied,
            "name": identity["display_name"],
            "size": st.st_size,
            "updated_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "display_name": identity["display_name"],
            "display_path": identity["display_path"],
            "runtime_name": identity["runtime_name"],
            "runtime_path": identity["runtime_path"],
            "model_role": identity["model_role"],
            "metadata": metadata,
        }
    identity = model_display_identity(applied, current)
    return {
        "path": "",
        "name": identity["display_name"],
        "size": 0,
        "updated_at": "",
        "display_name": identity["display_name"],
        "display_path": identity["display_path"],
        "runtime_name": identity["runtime_name"],
        "runtime_path": identity["runtime_path"],
        "model_role": identity["model_role"],
        "metadata": current,
    }


def parse_detection_summary(raw):
    summary = {
        "status": "empty",
        "model": "YOLO26",
        "source_topic": "/detection/results",
        "confidence_source": "/detection/results",
        "confidence_trusted": False,
        "count": 0,
        "items": [],
        "top_label": "",
        "max_confidence": None,
        "defect_in_center": None,
        "raw": raw or "",
    }
    if not raw:
        return summary

    try:
        data = json.loads(raw)
    except Exception:
        summary["status"] = "raw"
        return summary

    if isinstance(data, dict):
        detections = (
            data.get("detections")
            or data.get("results")
            or data.get("objects")
            or data.get("boxes")
            or []
        )
        if isinstance(detections, dict):
            detections = [detections]
        summary["defect_in_center"] = data.get("defect_in_center", data.get("center", None))
    elif isinstance(data, list):
        detections = data
    else:
        detections = []

    items = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        label = (
            det.get("label")
            or det.get("name")
            or det.get("class")
            or det.get("class_name")
            or "defect"
        )
        confidence = (
            det.get("confidence")
            if det.get("confidence") is not None
            else det.get("conf", det.get("score"))
        )
        try:
            confidence = round(float(confidence), 3)
        except Exception:
            confidence = None
        bbox = det.get("bbox") or det.get("box") or det.get("xyxy") or det.get("rect")
        in_center = det.get("in_center", det.get("center", det.get("defect_in_center")))
        items.append({
            "label": str(label),
            "confidence": confidence,
            "bbox": bbox,
            "in_center": in_center,
        })

    if items:
        top = max(items, key=lambda x: x["confidence"] if x["confidence"] is not None else -1)
        summary.update({
            "status": "ok",
            "count": len(items),
            "items": items[:12],
            "top_label": top.get("label", ""),
            "max_confidence": top.get("confidence"),
            "confidence_trusted": top.get("confidence") is not None,
        })
        if summary["defect_in_center"] is None:
            centers = [x.get("in_center") for x in items if x.get("in_center") is not None]
            if centers:
                summary["defect_in_center"] = any(bool(x) for x in centers)
    else:
        summary["status"] = "ok"

    return summary


def parse_infrared_summary(raw):
    summary = {
        "status": "empty",
        "source_topic": "/infrared/results",
        "count": 0,
        "alarm": False,
        "temperature_available": False,
        "temperature_source": None,
        "temperature_threshold_c": 70.0,
        "max_temperature_c": None,
        "temperature_text": "",
        "message": "",
        "max_intensity": None,
        "mean_intensity": None,
        "threshold": None,
        "threshold_source": None,
        "alarm_basis": None,
        "fallback_to_intensity_alarm": False,
        "relative_heat_alarm": False,
        "relative_heat_detected": False,
        "relative_heat_streak": 0,
        "relative_heat_confirmations": 1,
        "relative_hotspot_count": 0,
        "relative_heat": {},
        "intensity_hotspot_count": 0,
        "regions": [],
        "raw": raw or "",
    }
    if not raw:
        return summary

    try:
        data = json.loads(raw)
    except Exception:
        summary["status"] = "raw"
        return summary

    if not isinstance(data, dict):
        summary["status"] = "raw"
        return summary

    regions = data.get("regions") or []
    if not isinstance(regions, list):
        regions = []

    summary.update({
        "status": data.get("status", "ok"),
        "count": int(data.get("count", len(regions)) or 0),
        "alarm": bool(data.get("alarm", bool(regions))),
        "temperature_available": bool(data.get("temperature_available", False)),
        "temperature_source": data.get("temperature_source"),
        "temperature_threshold_c": data.get("temperature_threshold_c", 70.0),
        "max_temperature_c": data.get("max_temperature_c"),
        "temperature_text": data.get("temperature_text", ""),
        "message": data.get("message", ""),
        "max_intensity": data.get("max_intensity"),
        "mean_intensity": data.get("mean_intensity"),
        "threshold": data.get("threshold"),
        "threshold_source": data.get("threshold_source"),
        "alarm_basis": data.get("alarm_basis"),
        "fallback_to_intensity_alarm": bool(
            data.get("fallback_to_intensity_alarm", False)
        ),
        "relative_heat_alarm": bool(data.get("relative_heat_alarm", False)),
        "relative_heat_detected": bool(data.get("relative_heat_detected", False)),
        "relative_heat_streak": int(data.get("relative_heat_streak", 0) or 0),
        "relative_heat_confirmations": int(
            data.get("relative_heat_confirmations", 1) or 1
        ),
        "relative_hotspot_count": int(data.get("relative_hotspot_count", 0) or 0),
        "relative_heat": data.get("relative_heat") or {},
        "intensity_hotspot_count": int(
            data.get("intensity_hotspot_count", len(regions)) or 0
        ),
        "regions": regions[:12],
    })
    return summary


def ad_result_payload(ad_result):
    if not isinstance(ad_result, dict):
        return {}
    data = ad_result.get("data")
    return data if isinstance(data, dict) else {}


def summarize_impedance_for_fusion(ad_result=None, ad_status=None, age_sec=None):
    ad_result = ad_result if isinstance(ad_result, dict) else {}
    ad_status = ad_status if isinstance(ad_status, dict) else {}
    payload = ad_result_payload(ad_result)
    action = ad_result.get("action", "")

    summary = {
        "status": "empty",
        "available": False,
        "risk": False,
        "score": 0.0,
        "label": "未采样",
        "detail": "阻抗检测尚未参与融合",
        "action": action,
        "age_sec": age_sec,
        "raw": ad_result or {},
        "calibrated": {
            "low": bool(ad_status.get("data", {}).get("low_calibrated")) if isinstance(ad_status.get("data"), dict) else None,
            "cap": bool(ad_status.get("data", {}).get("cap_calibrated")) if isinstance(ad_status.get("data"), dict) else None,
        },
    }

    if not ad_result:
        return summary

    if ad_result.get("ok") is False or ad_result.get("status") == "error":
        summary.update({
            "status": "error",
            "available": False,
            "label": "采样错误",
            "detail": ad_result.get("error") or ad_result.get("message") or "AD5933 采样失败",
        })
        return summary

    result_status = payload.get("status")
    message = payload.get("first_line") or payload.get("message") or ad_result.get("message") or "AD5933 已采样"

    if action == "moisture_diagnose":
        freq_rows = [v for v in payload.values() if isinstance(v, dict)]
        valid = [v for v in freq_rows if v.get("loss_factor") is not None]
        errors = [v for v in freq_rows if v.get("error")]
        summary.update({
            "status": "ok" if valid else "warn",
            "available": bool(valid),
            "risk": False,
            "score": min(0.55, 0.12 * len(valid)),
            "label": f"多频 {len(valid)} 点",
            "detail": f"有效频点 {len(valid)} 个，错误 {len(errors)} 个",
        })
        return summary

    if action in ("raw", "cap_diagnose"):
        summary.update({
            "status": "ok",
            "available": True,
            "risk": False,
            "score": 0.15,
            "label": "已采样",
            "detail": message,
        })
        return summary

    score_map = {
        "dry": 0.10,
        "not_low_z": 0.20,
        "connected": 0.60,
        "open_fault": 0.85,
        "moisture_detected": 0.75,
        "severe_moisture": 0.95,
        "invalid_cap": 0.45,
        "unstable": 0.25,
        "need_low_cal": 0.0,
        "need_cap_cal": 0.0,
        "need_baseline": 0.0,
    }
    risk_statuses = {"connected", "open_fault", "moisture_detected", "severe_moisture", "invalid_cap"}
    unavailable_statuses = {"need_low_cal", "need_cap_cal", "need_baseline"}

    if result_status:
        score = score_map.get(result_status, 0.30)
        summary.update({
            "status": result_status,
            "available": result_status not in unavailable_statuses,
            "risk": result_status in risk_statuses,
            "score": score,
            "label": payload.get("first_line") or result_status,
            "detail": message,
        })
        return summary

    summary.update({
        "status": "ok",
        "available": True,
        "risk": False,
        "score": 0.20,
        "label": action or "已采样",
        "detail": message,
    })
    return summary


def clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def fusion_position_from_motion(motor_motion=None):
    if isinstance(motor_motion, dict):
        position = assistant_to_float(motor_motion.get("position_mm"))
        if position is not None:
            return position
    motion = parse_motor_motion_summary(node.latest_motor_motion_text if node else "")
    return assistant_to_float(motion.get("position_mm"))


def fusion_observation_signature(channel, raw, position_mm, timestamp=None, extra=""):
    pos = "nopos"
    if position_mm is not None:
        pos = f"{round(float(position_mm) / 2.0) * 2.0:.1f}"
    digest_src = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(digest_src[:5000].encode("utf-8", "ignore")).hexdigest()[:12]
    time_bucket = ""
    if timestamp:
        try:
            time_bucket = str(int(float(timestamp) * 2.0))
        except Exception:
            time_bucket = ""
    return f"{channel}:{pos}:{digest}:{time_bucket}:{extra}"


class KalmanFusionTrack:
    def __init__(self, track_id, observation):
        now = float(observation.get("timestamp") or time.time())
        position = assistant_to_float(observation.get("position_mm"), 0.0)
        severity = clamp01(observation.get("severity"))
        self.id = track_id
        self.position_mm = position
        self.position_variance = max(1.0, float(observation.get("position_variance") or 25.0))
        self.severity = severity
        self.severity_variance = max(0.0025, float(observation.get("severity_variance") or 0.08))
        self.created_at = now
        self.updated_at = now
        self.update_count = 0
        self.sources = {}
        self.observations = []
        self.last_signature = ""
        self.predict(now)
        self.update(observation)

    def predict(self, timestamp):
        dt = max(0.0, float(timestamp or time.time()) - float(self.updated_at or time.time()))
        self.position_variance = min(2500.0, self.position_variance + 0.60 * dt + 0.50)
        self.severity_variance = min(1.0, self.severity_variance + 0.015 * dt + 0.005)

    def update_scalar(self, state, variance, measurement, measurement_variance):
        measurement_variance = max(1e-6, float(measurement_variance))
        gain = variance / (variance + measurement_variance)
        state = state + gain * (float(measurement) - state)
        variance = max(1e-6, (1.0 - gain) * variance)
        return state, variance, gain

    def update(self, observation):
        timestamp = float(observation.get("timestamp") or time.time())
        self.predict(timestamp)

        position = assistant_to_float(observation.get("position_mm"))
        if position is not None:
            self.position_mm, self.position_variance, position_gain = self.update_scalar(
                self.position_mm,
                self.position_variance,
                position,
                observation.get("position_variance") or 25.0,
            )
        else:
            position_gain = 0.0

        severity = clamp01(observation.get("severity"))
        self.severity, self.severity_variance, severity_gain = self.update_scalar(
            self.severity,
            self.severity_variance,
            severity,
            observation.get("severity_variance") or 0.08,
        )
        self.severity = clamp01(self.severity)

        channel = str(observation.get("channel") or "unknown")
        self.sources[channel] = {
            "channel": channel,
            "label": observation.get("label", ""),
            "severity": round(severity, 3),
            "confidence": round(clamp01(observation.get("confidence", 0.0)), 3),
            "position_mm": round(position, 3) if position is not None else None,
            "position_variance": round(float(observation.get("position_variance") or 0.0), 3),
            "severity_variance": round(float(observation.get("severity_variance") or 0.0), 4),
            "position_gain": round(position_gain, 3),
            "severity_gain": round(severity_gain, 3),
            "risk": bool(observation.get("risk")),
            "detail": observation.get("detail", ""),
            "updated_at": datetime.fromtimestamp(timestamp).isoformat(timespec="seconds"),
        }
        self.observations.append({
            "channel": channel,
            "severity": round(severity, 3),
            "position_mm": round(position, 3) if position is not None else None,
            "timestamp": timestamp,
        })
        self.observations = self.observations[-20:]
        self.updated_at = timestamp
        self.update_count += 1
        self.last_signature = str(observation.get("signature") or "")

    def confidence(self):
        base_weights = {"vision": 0.45, "impedance": 0.30, "infrared": 0.25}
        weighted = 0.0
        weight_total = 0.0
        now = time.time()
        for channel, source in self.sources.items():
            age = max(0.0, now - datetime.fromisoformat(source["updated_at"]).timestamp())
            if age > 60.0:
                continue
            confidence = clamp01(source.get("confidence"))
            severity = clamp01(source.get("severity"))
            pos_sigma = max(0.0, float(source.get("position_variance") or 0.0)) ** 0.5
            sev_sigma = max(0.0, float(source.get("severity_variance") or 0.0)) ** 0.5
            stability = 1.0 / (1.0 + pos_sigma / 25.0 + sev_sigma)
            weight = base_weights.get(channel, 0.20) * max(0.10, confidence) * stability
            weighted += weight * severity
            weight_total += weight
        if weight_total <= 0.0:
            return round(self.severity, 3)
        return round(clamp01(weighted / weight_total), 3)

    def snapshot(self):
        confidence = self.confidence()
        risk_sources = [
            name for name, source in self.sources.items()
            if bool(source.get("risk")) or clamp01(source.get("severity")) >= 0.35
        ]
        level = "low"
        if self.severity >= 0.70 or len(risk_sources) >= 2:
            level = "high"
        elif self.severity >= 0.35 or risk_sources:
            level = "medium"
        return {
            "track_id": self.id,
            "fused_position_mm": round(self.position_mm, 3),
            "position_std_mm": round(self.position_variance ** 0.5, 3),
            "severity": round(self.severity, 3),
            "severity_std": round(self.severity_variance ** 0.5, 3),
            "confidence": confidence,
            "level": level,
            "risk_sources": risk_sources,
            "sources": self.sources,
            "observation_count": self.update_count,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(timespec="seconds"),
            "updated_at": datetime.fromtimestamp(self.updated_at).isoformat(timespec="seconds"),
        }


class KalmanFusionTracker:
    def __init__(self, gate_mm=18.0, stale_seconds=180.0):
        self.gate_mm = float(gate_mm)
        self.stale_seconds = float(stale_seconds)
        self.tracks = []
        self.next_id = 1
        self.seen_signatures = set()

    def prune(self):
        now = time.time()
        self.tracks = [
            track for track in self.tracks
            if now - float(track.updated_at or now) <= self.stale_seconds
        ]
        if len(self.seen_signatures) > 1200:
            self.seen_signatures = set(list(self.seen_signatures)[-600:])

    def nearest_track(self, observation):
        position = assistant_to_float(observation.get("position_mm"))
        if position is None:
            return None
        best = None
        best_distance = None
        for track in self.tracks:
            distance = abs(float(track.position_mm) - float(position))
            dynamic_gate = self.gate_mm + min(12.0, track.position_variance ** 0.5)
            if distance <= dynamic_gate and (best_distance is None or distance < best_distance):
                best = track
                best_distance = distance
        return best

    def update(self, observations):
        self.prune()
        applied = []
        for observation in observations:
            signature = str(observation.get("signature") or "")
            if signature and signature in self.seen_signatures:
                continue
            if signature:
                self.seen_signatures.add(signature)
            target = self.nearest_track(observation)
            severity = clamp01(observation.get("severity"))
            create_track = bool(observation.get("risk")) or severity >= 0.18
            if target is None and create_track:
                target = KalmanFusionTrack(f"F{self.next_id:03d}", observation)
                self.next_id += 1
                self.tracks.append(target)
            elif target is not None:
                target.update(observation)
            if target is not None:
                applied.append(target.id)
        self.prune()
        tracks = [
            track.snapshot()
            for track in sorted(
                self.tracks,
                key=lambda item: (item.severity, item.updated_at),
                reverse=True,
            )
        ]
        return tracks, applied

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.seen_signatures = set()


def get_fusion_tracker():
    global fusion_tracker
    with fusion_lock:
        if fusion_tracker is None:
            fusion_tracker = KalmanFusionTracker()
        return fusion_tracker


def fusion_channel_observations(detection, infrared, impedance, channels, motor_motion=None, infrared_temperature=None):
    observations = []
    current_position = fusion_position_from_motion(motor_motion)
    now = time.time()

    vision = channels.get("vision") or {}
    if vision.get("available") and current_position is not None:
        confidence = clamp01(detection.get("max_confidence") if detection.get("max_confidence") is not None else 0.45)
        centered = detection.get("defect_in_center")
        position_variance = 16.0 if centered is True else 36.0
        observations.append({
            "channel": "vision",
            "timestamp": getattr(node, "latest_detection_time", now) if node else now,
            "position_mm": current_position,
            "position_variance": position_variance,
            "severity": clamp01(vision.get("score")),
            "severity_variance": max(0.015, (1.0 - confidence) * 0.18),
            "confidence": confidence,
            "risk": bool(vision.get("risk")),
            "label": vision.get("label", "视觉"),
            "detail": vision.get("detail", ""),
            "signature": fusion_observation_signature(
                "vision",
                detection.get("raw", detection),
                current_position,
                getattr(node, "latest_detection_time", now) if node else now,
            ),
        })

    ir = channels.get("infrared") or {}
    if ir.get("available") and current_position is not None:
        ir_temp = infrared_temperature if isinstance(infrared_temperature, dict) else {}
        confidence = assistant_to_float(ir_temp.get("confidence"))
        if confidence is None:
            confidence = 0.75 if infrared.get("temperature_available") else 0.55
        position_variance = 25.0 if infrared.get("temperature_available") else 64.0
        observations.append({
            "channel": "infrared",
            "timestamp": getattr(node, "latest_infrared_time", now) if node else now,
            "position_mm": current_position,
            "position_variance": position_variance,
            "severity": clamp01(ir.get("score")),
            "severity_variance": max(0.02, (1.0 - clamp01(confidence)) * 0.22),
            "confidence": confidence,
            "risk": bool(ir.get("risk")),
            "label": ir.get("label", "红外"),
            "detail": ir.get("detail", ""),
            "signature": fusion_observation_signature(
                "infrared",
                {
                    "det": infrared.get("raw", infrared),
                    "temp": ir_temp,
                },
                current_position,
                getattr(node, "latest_infrared_time", now) if node else now,
            ),
        })

    imp = channels.get("impedance") or {}
    impedance_raw = impedance.get("raw") if isinstance(impedance, dict) else {}
    impedance_position = None
    if isinstance(impedance_raw, dict):
        impedance_position = assistant_to_float(impedance_raw.get("position_mm"))
    if impedance_position is None:
        impedance_position = current_position
    if imp.get("available") and impedance_position is not None:
        age_sec = assistant_to_float(impedance.get("age_sec"), 0.0) or 0.0
        confidence = max(0.35, min(0.85, 0.85 - min(age_sec, 60.0) / 120.0))
        observations.append({
            "channel": "impedance",
            "timestamp": latest_ad_result_time or now,
            "position_mm": impedance_position,
            "position_variance": 100.0,
            "severity": clamp01(imp.get("score")),
            "severity_variance": 0.10 if imp.get("risk") else 0.16,
            "confidence": confidence,
            "risk": bool(imp.get("risk")),
            "label": imp.get("label", "阻抗"),
            "detail": imp.get("detail", ""),
            "signature": fusion_observation_signature(
                "impedance",
                impedance_raw or impedance,
                impedance_position,
                latest_ad_result_time or now,
                extra=str(impedance.get("status") or ""),
            ),
        })

    return observations


def compute_fusion_summary(detection, infrared, impedance, motor_motion=None, infrared_temperature=None, update_tracker=True):
    visual_conf = detection.get("max_confidence")
    try:
        visual_score = max(0.0, min(1.0, float(visual_conf)))
    except Exception:
        visual_score = 0.0
    visual_available = detection.get("status") == "ok"
    visual_risk = bool(detection.get("count", 0) > 0)

    ir_det = infrared if isinstance(infrared, dict) else {}
    temperature_available = bool(ir_det.get("temperature_available"))
    temperature_threshold = assistant_to_float(
        ir_det.get("temperature_threshold_c")
    ) or 70.0
    max_temperature = assistant_to_float(ir_det.get("max_temperature_c"))
    relative_heat_available = bool(
        ir_det.get("relative_heat_alarm")
        or ir_det.get("relative_heat_detected")
        or ir_det.get("relative_hotspot_count", 0)
    )
    qwen_temperature_available = bool(
        isinstance(infrared_temperature, dict)
        and infrared_temperature.get("available")
        and infrared_temperature.get("ok")
        and infrared_temperature.get("temperature_c") is not None
    )
    qwen_temperature = assistant_to_float(
        infrared_temperature.get("temperature_c") if isinstance(infrared_temperature, dict) else None
    )
    qwen_threshold = assistant_to_float(
        infrared_temperature.get("threshold_c") if isinstance(infrared_temperature, dict) else None,
        INFRARED_AI_TEMPERATURE_THRESHOLD_C,
    )
    intensity_fallback = (
        ir_det.get("threshold_source") == "intensity_fallback"
        or ir_det.get("alarm_basis") == "intensity_fallback"
        or bool(ir_det.get("fallback_to_intensity_alarm"))
    )
    infrared_risk = bool(
        ir_det.get("alarm")
        or (qwen_temperature_available and qwen_temperature is not None and qwen_temperature > qwen_threshold)
        or ir_det.get("relative_heat_alarm")
    )
    if temperature_available and max_temperature is not None:
        ir_score = max(0.0, min(1.0, max_temperature / temperature_threshold))
    elif qwen_temperature_available and qwen_temperature is not None:
        ir_score = max(0.0, min(1.0, qwen_temperature / qwen_threshold))
    elif intensity_fallback and infrared_risk:
        max_intensity = assistant_to_float(ir_det.get("max_intensity"))
        intensity_threshold = assistant_to_float(ir_det.get("threshold"))
        if max_intensity is not None and intensity_threshold:
            ir_score = max(0.7, min(1.0, max_intensity / intensity_threshold))
        else:
            ir_score = 0.7
    elif relative_heat_available and infrared_risk:
        relative = ir_det.get("relative_heat") or {}
        rel_score = assistant_to_float(relative.get("score"), 0.65)
        contrast = assistant_to_float(relative.get("max_contrast"), 0.0) or 0.0
        ir_score = max(0.55, min(1.0, rel_score + min(0.25, contrast / 255.0)))
    else:
        ir_score = 0.0
    infrared_available = (
        (ir_det.get("status") == "ok" or qwen_temperature_available)
        and (
            temperature_available
            or qwen_temperature_available
            or intensity_fallback
            or relative_heat_available
        )
    )
    if infrared_available and not infrared_risk:
        ir_score = 0.0

    if not infrared_available:
        ir_detail = "等待Qwen3温度采样"
    elif max_temperature is not None:
        ir_detail = f"最高温度 {max_temperature:.1f} C，阈值 {temperature_threshold:.1f} C"
    elif qwen_temperature is not None:
        ir_detail = f"Qwen3-VL 温度 {qwen_temperature:.1f} C，阈值 {qwen_threshold:.1f} C"
    elif relative_heat_available:
        ir_detail = (
            f"相对热异常 {ir_det.get('relative_hotspot_count', 0)} 个，"
            f"score={(ir_det.get('relative_heat') or {}).get('score', '--')}"
        )
    elif intensity_fallback:
        ir_detail = (
            f"亮度热点 {ir_det.get('intensity_hotspot_count', ir_det.get('count', 0))} 个，"
            f"最高强度 {ir_det.get('max_intensity', '--')}，"
            f"阈值 {ir_det.get('threshold', '--')}"
        )
    else:
        ir_detail = "红外已接入，未发现异常"

    impedance_available = bool(impedance.get("available"))
    impedance_score = max(0.0, min(1.0, float(impedance.get("score") or 0.0)))
    impedance_risk = bool(impedance.get("risk"))

    channels = {
        "vision": {
            "available": visual_available,
            "risk": visual_risk,
            "score": round(visual_score, 3),
            "label": detection.get("top_label") or ("正常" if visual_available else "未上报"),
            "detail": f"{detection.get('count', 0)} 个视觉目标",
            "source": "/detection/results",
        },
        "impedance": {
            "available": impedance_available,
            "risk": impedance_risk,
            "score": round(impedance_score, 3),
            "label": impedance.get("label", "未采样"),
            "detail": impedance.get("detail", ""),
            "source": "AD5933",
            "status": impedance.get("status"),
            "age_sec": impedance.get("age_sec"),
        },
        "infrared": {
            "available": infrared_available,
            "risk": infrared_risk,
            "score": round(ir_score, 3),
            "label": "异常" if infrared_risk else ("正常" if infrared_available else "未上报"),
            "detail": ir_detail,
            "source": "/infrared/results",
        },
    }

    weights = {"vision": 0.45, "impedance": 0.30, "infrared": 0.25}
    weighted_sum = 0.0
    weight_total = 0.0
    for name, ch in channels.items():
        if not ch["available"]:
            continue
        weighted_sum += weights[name] * ch["score"]
        weight_total += weights[name]

    confidence = weighted_sum / weight_total if weight_total > 0 else 0.0
    risk_count = sum(1 for ch in channels.values() if ch["available"] and ch["risk"])
    available_count = sum(1 for ch in channels.values() if ch["available"])

    observations = fusion_channel_observations(
        detection,
        infrared,
        impedance,
        channels,
        motor_motion=motor_motion,
        infrared_temperature=infrared_temperature,
    )
    if update_tracker:
        with fusion_lock:
            tracks, applied_tracks = get_fusion_tracker().update(observations)
    else:
        tracks, applied_tracks = [], []

    top_track = tracks[0] if tracks else {}
    if top_track:
        confidence = clamp01(top_track.get("confidence"))
        top_severity = clamp01(top_track.get("severity"))
        if top_track.get("level") == "high" or top_severity >= 0.70:
            status = "alarm"
            level = "high"
            decision = "高风险"
        elif top_track.get("level") == "medium" or top_severity >= 0.35:
            status = "warn"
            level = "medium"
            decision = "需复核"
        else:
            status = "ok"
            level = "low"
            decision = "正常"
    elif available_count == 0:
        status = "empty"
        level = "unknown"
        decision = "未接入"
    elif confidence >= 0.70 or risk_count >= 2:
        status = "alarm"
        level = "high"
        decision = "高风险"
    elif confidence >= 0.35 or risk_count == 1:
        status = "warn"
        level = "medium"
        decision = "需复核"
    else:
        status = "ok"
        level = "low"
        decision = "正常"

    return {
        "status": status,
        "level": level,
        "decision": decision,
        "confidence": round(confidence, 3),
        "available_count": available_count,
        "risk_count": risk_count,
        "algorithm": "position_kalman_multimodal_v2",
        "legacy_weighted_confidence": round(
            weighted_sum / weight_total if weight_total > 0 else 0.0,
            3,
        ),
        "weights": weights,
        "channels": channels,
        "kalman": {
            "state": "tracking" if tracks else ("waiting_observation" if available_count else "empty"),
            "position_source": "motor_software_position_mm",
            "track_gate_mm": get_fusion_tracker().gate_mm,
            "observation_count": len(observations),
            "applied_tracks": applied_tracks,
            "top_track_id": top_track.get("track_id"),
            "fused_position_mm": top_track.get("fused_position_mm"),
            "position_std_mm": top_track.get("position_std_mm"),
            "severity": top_track.get("severity"),
            "severity_std": top_track.get("severity_std"),
        },
        "tracks": tracks[:8],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def parse_motor_motion_summary(raw):
    summary = {
        "status": "empty",
        "mode": "idle",
        "direction": "",
        "position_mm": None,
        "position_percent": None,
        "travel_min_mm": 0.0,
        "travel_max_mm": 400.0,
        "travel_span_mm": 400.0,
        "lead_mm_per_rev": None,
        "steps_per_rev": None,
        "mm_per_step": None,
        "speed_percent": None,
        "step_delay": None,
        "base_step_delay": None,
        "target_distance_mm": None,
        "target_steps": None,
        "completed_steps": None,
        "progress": None,
        "last_command": "",
        "last_error": "",
        "raw": raw or "",
    }
    if not raw:
        return summary

    try:
        data = json.loads(raw)
    except Exception:
        summary["status"] = "raw"
        return summary

    if not isinstance(data, dict):
        summary["status"] = "raw"
        return summary

    summary.update({
        "status": data.get("status", "ok"),
        "mode": data.get("mode", "idle"),
        "direction": data.get("direction", ""),
        "position_mm": data.get("position_mm"),
        "position_percent": data.get("position_percent"),
        "travel_min_mm": data.get("travel_min_mm", 0.0),
        "travel_max_mm": data.get("travel_max_mm", 400.0),
        "travel_span_mm": data.get("travel_span_mm", 400.0),
        "lead_mm_per_rev": data.get("lead_mm_per_rev"),
        "steps_per_rev": data.get("steps_per_rev"),
        "mm_per_step": data.get("mm_per_step"),
        "speed_percent": data.get("speed_percent"),
        "step_delay": data.get("step_delay"),
        "base_step_delay": data.get("base_step_delay"),
        "target_distance_mm": data.get("target_distance_mm"),
        "target_steps": data.get("target_steps"),
        "completed_steps": data.get("completed_steps"),
        "progress": data.get("progress"),
        "last_command": data.get("last_command", ""),
        "last_error": data.get("last_error", ""),
    })
    return summary


def inspection_snapshot():
    with inspection_lock:
        public = {
            key: value
            for key, value in inspection_state.items()
            if not str(key).startswith("_")
        }
        return json.loads(json.dumps(public, ensure_ascii=False, default=str))


def latest_infrared_temperature_sample():
    with inspection_lock:
        samples = list(inspection_state.get("infrared_ai_samples") or [])
        session_id = inspection_state.get("session_id", "")
        status = inspection_state.get("status", "idle")

    sample = samples[-1] if samples else {}
    captured_at = sample.get("captured_at") if isinstance(sample, dict) else None
    age_sec = None
    if captured_at:
        try:
            age_sec = round(
                (datetime.now() - datetime.fromisoformat(str(captured_at))).total_seconds(),
                3,
            )
        except Exception:
            age_sec = None

    return {
        "available": bool(sample),
        "session_id": session_id,
        "inspection_status": status,
        "captured_at": captured_at,
        "age_sec": age_sec,
        "ok": bool(sample.get("ok")) if isinstance(sample, dict) else False,
        "temperature_c": sample.get("temperature_c") if isinstance(sample, dict) else None,
        "raw_text": sample.get("raw_text", "") if isinstance(sample, dict) else "",
        "confidence": sample.get("confidence") if isinstance(sample, dict) else None,
        "is_abnormal": bool(sample.get("is_abnormal")) if isinstance(sample, dict) else False,
        "threshold_c": INFRARED_AI_TEMPERATURE_THRESHOLD_C,
        "model": sample.get("model", QWEN_VISION_MODEL) if isinstance(sample, dict) else QWEN_VISION_MODEL,
        "error": sample.get("error", "") if isinstance(sample, dict) else "",
    }


def save_inspection_state():
    snapshot = inspection_snapshot()
    session_id = snapshot.get("session_id")
    if not session_id:
        return
    session_dir = os.path.join(INSPECTION_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, "inspection.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def inspection_current_position():
    motion = parse_motor_motion_summary(
        node.latest_motor_motion_text if node else ""
    )
    return assistant_to_float(motion.get("position_mm"))


def load_qwen_vision_config():
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key and os.path.exists(DASHSCOPE_API_KEY_PATH):
        try:
            with open(DASHSCOPE_API_KEY_PATH, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        except Exception:
            api_key = ""
    return QWEN_VISION_BASE_URL.rstrip("/"), api_key, QWEN_VISION_MODEL


def extract_json_object(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {}


def call_qwen_temperature_reader(jpeg):
    base_url, api_key, model = load_qwen_vision_config()
    if not api_key:
        return {
            "ok": False,
            "error": f"missing API key: {DASHSCOPE_API_KEY_PATH}",
            "model": model,
        }

    image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是工业红外相机温度叠字识别器。只读取画面中的最高温度、"
                    "Max温度或温度叠加文字，不要根据颜色猜测温度。只返回JSON。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请识别这张红外画面中的最高温度/Max温度。只返回JSON: "
                            "{\"temperature_c\": number|null, \"raw_text\": string, "
                            "\"confidence\": number, \"is_abnormal\": boolean}。"
                            f"异常阈值为 {INFRARED_AI_TEMPERATURE_THRESHOLD_C:.1f} C。"
                            "如果没有看清温度文字，temperature_c 返回 null。"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 300,
        "stream": False,
    }

    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=80) as resp:
            raw = resp.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        return {
            "ok": False,
            "error": f"HTTP {exc.code}: {body[:300]}",
            "model": model,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "model": model}

    try:
        data = json.loads(raw)
        content = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except Exception:
        content = raw
    result = extract_json_object(content)
    temperature_c = assistant_to_float(result.get("temperature_c"))
    confidence = assistant_to_float(result.get("confidence"))
    is_abnormal = (
        temperature_c is not None
        and temperature_c > INFRARED_AI_TEMPERATURE_THRESHOLD_C
    )
    return {
        "ok": True,
        "model": model,
        "temperature_c": round(temperature_c, 1) if temperature_c is not None else None,
        "raw_text": str(result.get("raw_text") or "").strip(),
        "confidence": confidence,
        "is_abnormal": is_abnormal,
        "response_text": content,
    }


def inspection_record_infrared_ai_sample(session_id, sample):
    with inspection_lock:
        if (
            inspection_state.get("session_id") != session_id
            or inspection_state.get("status") not in ("running", "stopping", "finalizing")
        ):
            return
        inspection_state.setdefault("infrared_ai_samples", []).append(sample)
        inspection_state["infrared_ai_samples"] = inspection_state[
            "infrared_ai_samples"
        ][-600:]

        if not sample.get("is_abnormal"):
            save_needed = True
            event = None
        else:
            event_index = len(inspection_state.get("defects", [])) + 1
            event = {
                "index": event_index,
                "source": "红外AI",
                "channel": "infrared",
                "defect_type": "Qwen3-VL红外高温异常",
                "position_mm": sample.get("position_mm"),
                "captured_at": sample.get("captured_at"),
                "image_path": sample.get("image_path"),
                "image_url": sample.get("image_url"),
                "detail": {
                    "alarm": True,
                    "alarm_basis": "qwen3_vl_temperature",
                    "threshold_source": "qwen3_vl_image",
                    "max_temperature_c": sample.get("temperature_c"),
                    "temperature_text": sample.get("raw_text"),
                    "temperature_threshold_c": INFRARED_AI_TEMPERATURE_THRESHOLD_C,
                    "temperature_source": sample.get("model"),
                    "confidence": sample.get("confidence"),
                    "count": 1,
                    "regions": [],
                },
            }
            inspection_state.setdefault("defects", []).append(event)
            inspection_state["message"] = (
                f"检测中，已记录 {len(inspection_state['defects'])} 个缺陷事件"
            )
            save_needed = True

    if save_needed:
        save_inspection_state()
    if event and node:
        node.add_log(
            "inspection",
            "红外AI高温: "
            f"{sample.get('temperature_c')} C, 位置 "
            f"{sample.get('position_mm') if sample.get('position_mm') is not None else '--'} mm",
        )


def inspection_send_infrared_ai_sample(session_id, sample):
    result = call_qwen_temperature_reader(sample["jpeg"])
    sample = dict(sample)
    sample.pop("jpeg", None)
    sample.update({
        "model": result.get("model", QWEN_VISION_MODEL),
        "ok": bool(result.get("ok")),
        "temperature_c": result.get("temperature_c"),
        "raw_text": result.get("raw_text", ""),
        "confidence": result.get("confidence"),
        "is_abnormal": bool(result.get("is_abnormal")),
        "error": result.get("error", ""),
    })
    inspection_record_infrared_ai_sample(session_id, sample)
    if node:
        if sample["ok"]:
            node.add_log(
                "inspection",
                "红外AI上传: "
                f"#{sample.get('sample_index')} "
                f"{sample.get('temperature_c')} C, abnormal={sample['is_abnormal']}",
            )
        else:
            node.add_log("inspection", "红外AI上传失败: " + sample["error"][:120])
    return sample


def inspection_infrared_ai_worker(session_id):
    sample_index = 0
    inflight = set()
    session_dir = os.path.join(INSPECTION_DIR, session_id)
    sample_dir = os.path.join(session_dir, "infrared_ai_samples")
    os.makedirs(sample_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=INFRARED_AI_MAX_INFLIGHT_REQUESTS
    ) as executor:
        while not inspection_stop_event.is_set():
            with inspection_lock:
                if (
                    inspection_state.get("session_id") != session_id
                    or inspection_state.get("status") != "running"
                ):
                    break
            if not node:
                break

            done = {future for future in inflight if future.done()}
            for future in done:
                inflight.remove(future)
                try:
                    future.result()
                except Exception as exc:
                    if node:
                        node.add_log("inspection", "红外AI上传异常: " + str(exc)[:120])

            if len(inflight) >= INFRARED_AI_MAX_INFLIGHT_REQUESTS:
                if inspection_stop_event.wait(0.05):
                    break
                continue

            with node.frame_lock:
                jpeg = node.latest_infrared_raw_jpeg or node.latest_infrared_jpeg
                jpeg = bytes(jpeg) if jpeg else None
            if not jpeg:
                if inspection_stop_event.wait(0.1):
                    break
                continue

            sample_index += 1
            filename = f"{sample_index:04d}_qwen_ir.jpg"
            image_path = os.path.join(sample_dir, filename)
            with open(image_path, "wb") as f:
                f.write(jpeg)

            position = inspection_current_position()
            sample = {
                "sample_index": sample_index,
                "captured_at": datetime.now().isoformat(timespec="seconds"),
                "position_mm": round(position, 3) if position is not None else None,
                "image_path": image_path,
                "image_url": f"/api/inspection/image/{session_id}/infrared_ai_samples/{filename}",
                "jpeg": jpeg,
            }
            inflight.add(executor.submit(
                inspection_send_infrared_ai_sample,
                session_id,
                sample,
            ))

            if inspection_stop_event.wait(INFRARED_AI_SAMPLE_INTERVAL_SECONDS):
                break

        done, pending = concurrent.futures.wait(inflight, timeout=90)
        for future in done:
            try:
                future.result()
            except Exception as exc:
                if node:
                    node.add_log("inspection", "红外AI上传异常: " + str(exc)[:120])
        for future in pending:
            future.cancel()


def inspection_capture_event(channel, summary):
    if not node or not isinstance(summary, dict):
        return

    if channel == "vision":
        items = summary.get("items") or []
        if not items:
            return
        top = items[0] if isinstance(items[0], dict) else {}
        defect_type = str(top.get("label") or summary.get("top_label") or "视觉缺陷")
        detail = {
            "confidence": top.get("confidence"),
            "bbox": top.get("bbox"),
            "count": summary.get("count", len(items)),
        }
        jpeg_attr = "latest_jpeg"
        source_name = "视觉"
    elif channel == "infrared":
        # Realtime infrared diagnostics do not create report events. Qwen3
        # temperature sampling records reportable infrared temperature issues.
        return
        if not summary.get("alarm"):
            return
        defect_type = (
            "红外热异常"
            if summary.get("temperature_available")
            else "红外亮度异常"
        )
        if summary.get("alarm_basis") == "relative_heat_video":
            defect_type = "红外诊断事件"
        detail = {
            "alarm": summary.get("alarm"),
            "alarm_basis": summary.get("alarm_basis"),
            "threshold_source": summary.get("threshold_source"),
            "max_temperature_c": summary.get("max_temperature_c"),
            "temperature_text": summary.get("temperature_text"),
            "temperature_threshold_c": summary.get("temperature_threshold_c", 70.0),
            "temperature_source": summary.get("temperature_source"),
            "max_intensity": summary.get("max_intensity"),
            "mean_intensity": summary.get("mean_intensity"),
            "threshold": summary.get("threshold"),
            "relative_heat_alarm": summary.get("relative_heat_alarm"),
            "relative_heat_detected": summary.get("relative_heat_detected"),
            "relative_heat_streak": summary.get("relative_heat_streak"),
            "relative_heat_confirmations": summary.get("relative_heat_confirmations"),
            "relative_hotspot_count": summary.get("relative_hotspot_count"),
            "relative_heat": summary.get("relative_heat"),
            "intensity_hotspot_count": summary.get("intensity_hotspot_count"),
            "count": summary.get("count", 0),
            "regions": summary.get("regions", [])[:3],
        }
        jpeg_attr = "latest_infrared_jpeg"
        source_name = "红外"
    else:
        return

    position = inspection_current_position()
    now = time.time()
    with inspection_lock:
        if inspection_state.get("status") != "running":
            return
        session_id = inspection_state.get("session_id")
        key = f"{channel}:{defect_type}"
        last = (inspection_state.get("_last_capture") or {}).get(key)
        if last:
            last_position = assistant_to_float(last.get("position_mm"))
            merge_distance = (
                VISUAL_REPORT_MERGE_DISTANCE_MM
                if channel == "vision"
                else 10.0
            )
            close_in_position = (
                position is not None
                and last_position is not None
                and abs(position - last_position) < merge_distance
            )
            if (
                close_in_position
                or now - float(last.get("time", 0))
                < INSPECTION_CAPTURE_MIN_INTERVAL_SECONDS
            ):
                return
        inspection_state.setdefault("_last_capture", {})[key] = {
            "position_mm": position,
            "time": now,
        }
        event_index = len(inspection_state.get("defects", [])) + 1

    with node.frame_lock:
        jpeg = getattr(node, jpeg_attr, None)
        jpeg = bytes(jpeg) if jpeg else None
    if not jpeg:
        return

    session_dir = os.path.join(INSPECTION_DIR, session_id)
    image_dir = os.path.join(session_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    filename = f"{event_index:03d}_{channel}.jpg"
    image_path = os.path.join(image_dir, filename)
    with open(image_path, "wb") as f:
        f.write(jpeg)

    event = {
        "index": event_index,
        "source": source_name,
        "channel": channel,
        "defect_type": defect_type,
        "position_mm": round(position, 3) if position is not None else None,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "image_path": image_path,
        "image_url": f"/api/inspection/image/{session_id}/{filename}",
        "detail": detail,
    }
    with inspection_lock:
        if inspection_state.get("session_id") != session_id:
            return
        inspection_state.setdefault("defects", []).append(event)
        inspection_state["message"] = (
            f"检测中，已记录 {len(inspection_state['defects'])} 个缺陷事件"
        )
    save_inspection_state()
    node.add_log(
        "inspection",
        f"{source_name}缺陷: {defect_type}, 位置 "
        f"{event['position_mm'] if event['position_mm'] is not None else '--'} mm",
    )


def inspection_impedance_worker(session_id, cable_length_m):
    global latest_ad_result, latest_ad_result_time
    sample_index = 0
    try:
        while not inspection_stop_event.is_set():
            sample_index += 1
            result, code = run_ad5933_action(
                "analyze",
                {
                    "cable_length_m": cable_length_m,
                    "count": 10,
                },
                timeout=30,
            )
            result["status"] = "ok" if code == 200 else "error"
            result["sample_index"] = sample_index
            result["sampled_at"] = datetime.now().isoformat(timespec="seconds")
            result["position_mm"] = inspection_current_position()
            with ad_state_lock:
                latest_ad_result = result
                latest_ad_result_time = time.time()
            with inspection_lock:
                if inspection_state.get("session_id") != session_id:
                    return
                inspection_state.setdefault("impedance_samples", []).append(result)
            save_inspection_state()
            inspection_impedance_ready.set()
            if node:
                node.add_log(
                    "inspection",
                    "阻抗采样完成: "
                    + str(
                        (result.get("data") or {}).get("first_line")
                        or result.get("message")
                        or result.get("error")
                        or code
                    ),
                )
            if inspection_stop_event.wait(8.0):
                break
    finally:
        inspection_impedance_ready.set()


def inspection_report_text(value, fallback="未上报"):
    if value is None or value == "":
        return fallback
    return str(value)


def merge_infrared_report_events(events, distance_mm=INFRARED_REPORT_MERGE_DISTANCE_MM):
    positioned = []
    unpositioned = []
    for index, event in enumerate(events or []):
        position = assistant_to_float(event.get("position_mm"))
        if position is None:
            unpositioned.append(event)
        else:
            positioned.append((position, index, event))

    positioned.sort(key=lambda item: (item[0], item[1]))
    merged = []
    group = []

    def event_temperature(event):
        detail = event.get("detail") or {}
        return assistant_to_float(detail.get("max_temperature_c"))

    def representative(group_items):
        best = group_items[0]
        best_temp = event_temperature(best[2])
        for item in group_items[1:]:
            temp = event_temperature(item[2])
            if temp is not None and (best_temp is None or temp > best_temp):
                best = item
                best_temp = temp
        return best[2]

    for item in positioned:
        if not group:
            group = [item]
            continue
        if item[0] - group[0][0] < distance_mm:
            group.append(item)
        else:
            merged.append(representative(group))
            group = [item]
    if group:
        merged.append(representative(group))

    return merged + unpositioned


def merge_visual_report_events(events, distance_mm=VISUAL_REPORT_MERGE_DISTANCE_MM):
    positioned = []
    unpositioned = []
    for index, event in enumerate(events or []):
        position = assistant_to_float(event.get("position_mm"))
        if position is None:
            unpositioned.append(event)
        else:
            positioned.append((position, index, event))

    positioned.sort(key=lambda item: (
        str(item[2].get("defect_type") or ""),
        item[0],
        item[1],
    ))
    merged = []
    group = []

    def event_confidence(event):
        detail = event.get("detail") or {}
        return assistant_to_float(
            detail.get("confidence")
            or event.get("confidence")
        )

    def representative(group_items):
        best = group_items[0]
        best_conf = event_confidence(best[2])
        for item in group_items[1:]:
            conf = event_confidence(item[2])
            if conf is not None and (best_conf is None or conf > best_conf):
                best = item
                best_conf = conf
        return best[2]

    for item in positioned:
        if not group:
            group = [item]
            continue
        same_type = (
            str(item[2].get("defect_type") or "")
            == str(group[-1][2].get("defect_type") or "")
        )
        if same_type and item[0] - group[0][0] < distance_mm:
            group.append(item)
        else:
            merged.append(representative(group))
            group = [item]
    if group:
        merged.append(representative(group))

    merged.sort(key=lambda event: (
        assistant_to_float(event.get("position_mm")) is None,
        assistant_to_float(event.get("position_mm")) or 0.0,
    ))
    return merged + unpositioned


def build_inspection_report_fusion(visual_events, infrared_events, impedance_samples):
    now = time.time()
    tracker = KalmanFusionTracker(gate_mm=get_fusion_tracker().gate_mm)
    observations = []

    def append_observation(observation):
        position = assistant_to_float(observation.get("position_mm"))
        if position is None:
            return
        observation["timestamp"] = now
        observation["signature"] = fusion_observation_signature(
            observation.get("channel", "report"),
            {
                "label": observation.get("label"),
                "detail": observation.get("detail"),
                "severity": observation.get("severity"),
            },
            position,
            now,
            extra="inspection_report",
        )
        observations.append(observation)

    for event in visual_events or []:
        detail = event.get("detail") or {}
        confidence = assistant_to_float(
            detail.get("confidence") or event.get("confidence"),
            0.45,
        )
        confidence = clamp01(confidence)
        append_observation({
            "channel": "vision",
            "position_mm": event.get("position_mm"),
            "position_variance": 36.0,
            "severity": confidence,
            "severity_variance": max(0.015, (1.0 - confidence) * 0.18),
            "confidence": confidence,
            "risk": True,
            "label": event.get("defect_type") or "视觉缺陷",
            "detail": f"视觉检测到 {event.get('defect_type') or '缺陷'}",
        })

    for event in infrared_events or []:
        detail = event.get("detail") or {}
        temperature = assistant_to_float(detail.get("max_temperature_c"))
        threshold = (
            assistant_to_float(detail.get("temperature_threshold_c"))
            or INFRARED_AI_TEMPERATURE_THRESHOLD_C
        )
        if temperature is not None and threshold:
            severity = clamp01(temperature / threshold)
            detail_text = f"温度 {temperature:.1f} C，阈值 {threshold:.1f} C"
        else:
            severity = 0.70
            detail_text = event.get("defect_type") or "红外异常"
        confidence = assistant_to_float(detail.get("confidence"), 0.75)
        confidence = clamp01(confidence)
        append_observation({
            "channel": "infrared",
            "position_mm": event.get("position_mm"),
            "position_variance": 25.0,
            "severity": severity,
            "severity_variance": max(0.02, (1.0 - confidence) * 0.22),
            "confidence": confidence,
            "risk": True,
            "label": event.get("defect_type") or "红外异常",
            "detail": detail_text,
        })

    for sample in impedance_samples or []:
        impedance = summarize_impedance_for_fusion(sample, {}, None)
        if not impedance.get("available"):
            continue
        confidence = 0.75 if impedance.get("risk") else 0.55
        append_observation({
            "channel": "impedance",
            "position_mm": sample.get("position_mm"),
            "position_variance": 100.0,
            "severity": clamp01(impedance.get("score")),
            "severity_variance": 0.10 if impedance.get("risk") else 0.16,
            "confidence": confidence,
            "risk": bool(impedance.get("risk")),
            "label": impedance.get("label") or "阻抗采样",
            "detail": impedance.get("detail") or "",
        })

    tracks, applied_tracks = tracker.update(
        sorted(
            observations,
            key=lambda item: (
                assistant_to_float(item.get("position_mm"), 0.0) or 0.0,
                str(item.get("channel") or ""),
            ),
        )
    )
    return {
        "algorithm": "position_kalman_multimodal_v2",
        "gate_mm": tracker.gate_mm,
        "observation_count": len(observations),
        "applied_tracks": applied_tracks,
        "tracks": tracks,
    }


def inspection_report_filename(snapshot):
    time_text = (
        snapshot.get("finished_at")
        or snapshot.get("started_at")
        or datetime.now().isoformat(timespec="seconds")
    )
    try:
        dt = datetime.fromisoformat(str(time_text).replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now()
    return f"检测报告_{dt.strftime('%Y%m%d_%H%M%S')}.pdf"


def generate_inspection_pdf(snapshot):
    from fpdf import FPDF

    session_id = snapshot["session_id"]
    session_dir = os.path.join(INSPECTION_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    report_path = os.path.join(session_dir, inspection_report_filename(snapshot))

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_font("CJK", fname=INSPECTION_FONT)

    def write_text(text, height=6):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, height, str(text))

    def image_render_size(image_path, max_width, max_height):
        image = cv2.imread(image_path)
        if image is None:
            return max_width, max_height
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return max_width, max_height
        scale = min(max_width / width, max_height / height)
        return width * scale, height * scale

    def event_detail_text(event):
        detail = event.get("detail") or {}
        if event.get("channel") == "vision":
            return (
                f"置信度 {inspection_report_text(detail.get('confidence'))}  |  "
                f"检测数量 {inspection_report_text(detail.get('count'))}"
            )
        max_temperature = assistant_to_float(detail.get("max_temperature_c"))
        threshold = assistant_to_float(detail.get("temperature_threshold_c")) or 70.0
        temperature_text = str(detail.get("temperature_text") or "").strip()
        text_part = f"  |  {temperature_text[:18]}" if temperature_text else ""
        if max_temperature is not None:
            return (
                f"最高温度 {max_temperature:.1f} C  |  "
                f"报警阈值 {threshold:.1f} C{text_part}"
            )
        if detail.get("alarm_basis") == "relative_heat_video":
            regions = detail.get("regions") or []
            top = regions[0] if regions and isinstance(regions[0], dict) else {}
            heat = detail.get("relative_heat") or {}
            score = top.get("max_heat_score") or heat.get("max_score")
            contrast = top.get("max_contrast") or heat.get("max_contrast")
            streak = inspection_report_text(detail.get("relative_heat_streak"), "0")
            confirmations = inspection_report_text(
                detail.get("relative_heat_confirmations"),
                "1",
            )
            return (
                f"热斑分数 {inspection_report_text(score)}  |  "
                f"局部对比 {inspection_report_text(contrast)}  |  "
                f"确认 {streak}/{confirmations}"
            )
        return (
            "历史记录无温度矩阵"
            f"  |  最高亮度 {inspection_report_text(detail.get('max_intensity'))}"
        )

    def draw_event_grid(section_title, events, empty_text):
        pdf.add_page()
        pdf.set_font("CJK", size=13)
        pdf.cell(0, 9, section_title, new_x="LMARGIN", new_y="NEXT")
        if not events:
            pdf.set_font("CJK", size=10)
            write_text(empty_text)
            return

        gap = 5.0
        card_width = (pdf.epw - gap) / 2.0
        card_height = 70.0
        image_max_width = card_width - 10.0
        image_max_height = 44.0
        page_bottom = pdf.h - pdf.b_margin

        for row_start in range(0, len(events), 2):
            if pdf.get_y() + card_height > page_bottom:
                pdf.add_page()
                pdf.set_font("CJK", size=10)
                pdf.set_text_color(90, 90, 90)
                pdf.cell(
                    0,
                    7,
                    f"{section_title}（续）",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
                pdf.set_text_color(0, 0, 0)

            row_y = pdf.get_y()
            for column, event in enumerate(events[row_start:row_start + 2]):
                x = pdf.l_margin + column * (card_width + gap)
                position = assistant_to_float(event.get("position_mm"))
                position_text = (
                    f"{position:.3f} mm" if position is not None else "未上报"
                )
                section_index = row_start + column + 1
                title = (
                    f"{section_index}. {event.get('defect_type', '缺陷')}"
                    f"  |  位置 {position_text}"
                )

                pdf.set_draw_color(210, 218, 230)
                pdf.set_fill_color(248, 250, 253)
                pdf.rect(x, row_y, card_width, card_height, style="DF")
                pdf.set_xy(x + 4, row_y + 4)
                pdf.set_font("CJK", size=9)
                pdf.cell(card_width - 8, 5, title[:48])
                pdf.set_xy(x + 4, row_y + 10)
                pdf.set_font("CJK", size=8)
                pdf.set_text_color(70, 70, 70)
                pdf.cell(card_width - 8, 5, event_detail_text(event)[:58])
                pdf.set_text_color(0, 0, 0)

                image_path = event.get("image_path")
                if image_path and os.path.exists(image_path):
                    image_width, image_height = image_render_size(
                        image_path,
                        image_max_width,
                        image_max_height,
                    )
                    image_x = x + (card_width - image_width) / 2.0
                    image_y = row_y + 20.0 + (image_max_height - image_height) / 2.0
                    pdf.image(
                        image_path,
                        x=image_x,
                        y=image_y,
                        w=image_width,
                        h=image_height,
                    )
            pdf.set_y(row_y + card_height + gap)

    def draw_fusion_section(fusion_report):
        channel_names = {
            "vision": "视觉",
            "infrared": "红外",
            "impedance": "阻抗",
        }
        level_names = {
            "high": "高风险",
            "medium": "需复核",
            "low": "低风险/正常",
            "unknown": "未接入",
        }

        pdf.set_font("CJK", size=13)
        pdf.cell(0, 9, "五、多模态融合结果", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("CJK", size=10)
        write_text(
            "融合来源于项目内 position_kalman_multimodal_v2：按电机软件位置把视觉、红外、"
            "阻抗观测送入卡尔曼融合轨迹，同一位置附近的多通道证据会合并为一个融合点。"
        )
        write_text(
            f"融合门限：{inspection_report_text(fusion_report.get('gate_mm'))} mm；"
            f"参与观测：{inspection_report_text(fusion_report.get('observation_count'))} 条。"
        )

        tracks = fusion_report.get("tracks") or []
        if not tracks:
            write_text("本次报告没有形成融合轨迹，通常表示三类传感结果没有可用的位置观测。")
            return

        for index, track in enumerate(tracks[:8], 1):
            position = assistant_to_float(track.get("fused_position_mm"))
            position_text = f"{position:.3f} mm" if position is not None else "未上报"
            std = assistant_to_float(track.get("position_std_mm"))
            std_text = f"±{std:.1f} mm" if std is not None else ""
            level = level_names.get(track.get("level"), track.get("level") or "未上报")
            confidence = inspection_report_text(track.get("confidence"))
            severity = inspection_report_text(track.get("severity"))
            risk_sources = [
                channel_names.get(name, name)
                for name in (track.get("risk_sources") or [])
            ]
            risk_text = "、".join(risk_sources) if risk_sources else "无明显风险通道"
            write_text(
                f"融合点 {index}：位置 {position_text}{std_text}；结论 {level}；"
                f"融合置信度 {confidence}；严重度 {severity}；风险来源：{risk_text}。"
            )

            sources = track.get("sources") or {}
            for channel in ("vision", "infrared", "impedance"):
                source = sources.get(channel)
                if not source:
                    continue
                source_position = assistant_to_float(source.get("position_mm"))
                source_position_text = (
                    f"{source_position:.3f} mm"
                    if source_position is not None
                    else "未上报"
                )
                write_text(
                    f"  - {channel_names[channel]}：{source.get('label') or '已参与'}；"
                    f"位置 {source_position_text}；置信度 {inspection_report_text(source.get('confidence'))}；"
                    f"{source.get('detail') or '无补充说明'}",
                    5,
                )
        pdf.ln(2)

    pdf.add_page()
    pdf.set_font("CJK", size=18)
    pdf.cell(0, 12, "电缆工业检测报告", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("CJK", size=9)
    pdf.cell(
        0,
        7,
        f"报告编号：{session_id}",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(3)

    pdf.set_font("CJK", size=13)
    pdf.cell(0, 9, "一、检测概况", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("CJK", size=10)
    defects = snapshot.get("defects") or []
    infrared_ai_samples = snapshot.get("infrared_ai_samples") or []
    visual_records = [event for event in defects if event.get("channel") == "vision"]
    visual_defects = merge_visual_report_events(visual_records)
    infrared_records = [
        event for event in defects if event.get("channel") == "infrared"
    ]
    infrared_defects = merge_infrared_report_events(infrared_records)
    actionable_defects = visual_defects + infrared_defects
    overview = [
        ("设备", "RDK X5 电缆缺陷检测系统"),
        ("开始时间", inspection_report_text(snapshot.get("started_at"))),
        ("结束时间", inspection_report_text(snapshot.get("finished_at"))),
        (
            "检测区间",
            f"{inspection_report_text(snapshot.get('start_position_mm'))} mm → "
            f"{inspection_report_text(snapshot.get('end_position_mm'))} mm",
        ),
        ("运动方向", inspection_report_text(snapshot.get("direction"))),
        ("视觉缺陷数", str(len(visual_defects))),
        ("红外异常数", str(len(infrared_defects))),
        ("Qwen3-VL红外采样数", str(len(infrared_ai_samples))),
        ("红外报警阈值", "70.0 C（Qwen3-VL读图温度）"),
        (
            "检测结论",
            "发现缺陷，建议复核"
            if actionable_defects
            else "未发现视觉缺陷或超过 70.0 C 的红外异常",
        ),
    ]
    for label, value in overview:
        write_text(f"{label}：{value}")
    pdf.ln(2)

    pdf.set_font("CJK", size=13)
    pdf.cell(0, 9, "二、阻抗检测情况", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("CJK", size=10)
    samples = snapshot.get("impedance_samples") or []
    if not samples:
        write_text("阻抗采样未完成或 AD5933 未返回有效结果。")
    for sample in samples:
        payload = sample.get("data") if isinstance(sample.get("data"), dict) else {}
        title = (
            payload.get("first_line")
            or sample.get("message")
            or sample.get("error")
            or "阻抗采样"
        )
        position = assistant_to_float(sample.get("position_mm"))
        position_text = f"{position:.3f} mm" if position is not None else "未上报"
        write_text(
            f"采样 {sample.get('sample_index', '--')}，位置 {position_text}：{title}",
        )
        friendly = payload.get("friendly_text")
        if friendly:
            pdf.set_text_color(70, 70, 70)
            write_text(friendly, 5)
            pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    draw_event_grid(
        "三、视觉缺陷记录",
        visual_defects,
        "本次检测未记录到视觉缺陷。",
    )
    draw_event_grid(
        "四、红外检测记录",
        infrared_defects,
        "本次检测未记录到温度超过 70.0 C 的红外异常。",
    )

    fusion_report = build_inspection_report_fusion(
        visual_defects,
        infrared_defects,
        samples,
    )
    draw_fusion_section(fusion_report)

    pdf.add_page()
    pdf.set_font("CJK", size=13)
    pdf.cell(0, 9, "六、处理建议", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("CJK", size=10)
    if actionable_defects:
        write_text(
            "建议根据报告中的电机位置复核对应电缆区段；对视觉缺陷检查表面破损、"
            "异物或绝缘异常，对红外热异常检查局部发热、接触电阻和受潮情况，并结合"
            "阻抗结果决定是否停机检修。",
        )
    else:
        write_text(
            "本次视觉与红外检测未发现缺陷。建议保留本报告，并按维护周期继续复检。",
        )
    pdf.output(report_path)
    return report_path


def finalize_inspection_session(reason="电机停止"):
    with inspection_lock:
        if inspection_state.get("status") not in ("running", "stopping"):
            return inspection_snapshot()
        inspection_state["status"] = "finalizing"
        inspection_state["message"] = "电机已停止，正在生成工业检测报告"
        session_id = inspection_state.get("session_id")
    inspection_stop_event.set()
    inspection_impedance_ready.wait(timeout=35)

    end_position = inspection_current_position()
    with inspection_lock:
        if inspection_state.get("session_id") != session_id:
            return inspection_snapshot()
        inspection_state["end_position_mm"] = (
            round(end_position, 3) if end_position is not None else None
        )
        finished_at = datetime.now().isoformat(timespec="seconds")
        started_at = inspection_state.get("started_at")
        started_epoch = assistant_to_float(inspection_state.get("_started_epoch"))
        if started_at and started_epoch is not None:
            try:
                start_dt = datetime.fromisoformat(
                    str(started_at).replace("Z", "+00:00")
                )
                elapsed = max(0.0, time.time() - started_epoch)
                finished_at = (
                    start_dt + timedelta(seconds=elapsed)
                ).isoformat(timespec="seconds")
            except Exception:
                pass
        inspection_state["finished_at"] = finished_at
        inspection_state["completion_reason"] = reason
        snapshot = inspection_snapshot()
    try:
        report_path = generate_inspection_pdf(snapshot)
        with inspection_lock:
            inspection_state["status"] = "completed"
            inspection_state["message"] = "检测完成，工业检测报告已生成"
            inspection_state["report_path"] = report_path
            inspection_state["report_url"] = (
                f"/api/inspection/report/{session_id}"
            )
            inspection_state["error"] = ""
    except Exception as exc:
        with inspection_lock:
            inspection_state["status"] = "failed"
            inspection_state["message"] = "检测完成，但 PDF 报告生成失败"
            inspection_state["error"] = str(exc)
    save_inspection_state()
    if node:
        node.add_log("inspection", inspection_state.get("message", "检测结束"))
    return inspection_snapshot()


def inspection_monitor_worker(session_id):
    seen_running = False
    started = time.time()
    while True:
        with inspection_lock:
            session_status = inspection_state.get("status")
            if (
                inspection_state.get("session_id") != session_id
                or session_status not in ("running", "stopping")
            ):
                return
        motion = parse_motor_motion_summary(
            node.latest_motor_motion_text if node else ""
        )
        running = (
            motion.get("status") == "running"
            or motion.get("mode") in ("move", "continuous")
            or (node and node.motor_status == "running")
        )
        if running:
            seen_running = True
        if (
            (seen_running or session_status == "stopping")
            and not running
            and time.time() - started > 1.0
        ):
            finalize_inspection_session("电机停止或到达行程终点")
            return
        if time.time() - started > 600:
            if node:
                node.send_command("stop")
            finalize_inspection_session("检测超时，已安全停止")
            return
        time.sleep(0.4)


def start_inspection_session(client_time=None):
    if not node:
        return False, "ROS2 Web 节点未就绪。", inspection_snapshot()
    with inspection_lock:
        if inspection_state.get("status") in ("running", "stopping", "finalizing"):
            return False, "已有检测任务正在运行。", inspection_snapshot()

    motion = parse_motor_motion_summary(node.latest_motor_motion_text)
    current = assistant_to_float(motion.get("position_mm"))
    travel_min = assistant_to_float(motion.get("travel_min_mm"), 0.0)
    travel_max = assistant_to_float(motion.get("travel_max_mm"), 400.0)
    if current is None:
        return False, "电机当前位置未上报，无法开始自动检测。", inspection_snapshot()

    distance_to_min = abs(current - travel_min)
    distance_to_max = abs(travel_max - current)
    target = travel_max if distance_to_max >= distance_to_min else travel_min
    direction = "forward" if target >= current else "reverse"
    if abs(target - current) < 0.05:
        target = travel_min if target == travel_max else travel_max
        direction = "forward" if target >= current else "reverse"

    started_at = str(client_time or "").strip()
    if not re.match(r"^20\d{2}-\d{2}-\d{2}T", started_at):
        started_at = datetime.now().isoformat(timespec="seconds")
    session_prefix = re.sub(r"\D", "", started_at[:19])[:14]
    session_id = (session_prefix or datetime.now().strftime("%Y%m%d%H%M%S")) + "_" + uuid.uuid4().hex[:6]
    session_dir = os.path.join(INSPECTION_DIR, session_id)
    os.makedirs(os.path.join(session_dir, "images"), exist_ok=True)
    inspection_stop_event.clear()
    inspection_impedance_ready.clear()
    with inspection_lock:
        inspection_state.clear()
        inspection_state.update({
            "status": "running",
            "session_id": session_id,
            "message": "自动检测已启动：电机、视觉、红外和阻抗并行检测",
            "started_at": started_at,
            "finished_at": None,
            "start_position_mm": round(current, 3),
            "end_position_mm": None,
            "target_position_mm": round(target, 3),
            "direction": direction,
            "defects": [],
            "infrared_ai_samples": [],
            "impedance_samples": [],
            "report_path": "",
            "report_url": "",
            "error": "",
            "_last_capture": {},
            "_started_epoch": time.time(),
        })
    save_inspection_state()

    cable_length_m = max(0.01, abs(target - current) / 1000.0)
    node.move_motor_to(target)
    threading.Thread(
        target=inspection_impedance_worker,
        args=(session_id, cable_length_m),
        daemon=True,
    ).start()
    threading.Thread(
        target=inspection_infrared_ai_worker,
        args=(session_id,),
        daemon=True,
    ).start()
    threading.Thread(
        target=inspection_monitor_worker,
        args=(session_id,),
        daemon=True,
    ).start()
    node.add_log(
        "inspection",
        f"自动检测启动: {current:.3f} -> {target:.3f} mm, {direction}",
    )
    return True, (
        f"自动检测已开始。电机从 {current:.1f} mm 向 {target:.1f} mm "
        "运行，视觉、红外和阻抗正在并行检测。"
    ), inspection_snapshot()


def stop_inspection_session():
    with inspection_lock:
        if inspection_state.get("status") != "running":
            return False, "当前没有正在运行的自动检测任务。", inspection_snapshot()
        inspection_state["status"] = "stopping"
        inspection_state["message"] = "正在停止电机并生成报告"
    if node:
        node.send_command("stop")
    return True, "已发送停止命令，电机停止后将生成工业检测报告。", inspection_snapshot()


def validate_model_path(model_path):
    if not model_path:
        return "", "No valid .bin model selected."

    abs_path = os.path.abspath(model_path)
    models_root = os.path.abspath(MODELS_DIR)
    try:
        inside_models = os.path.commonpath([models_root, abs_path]) == models_root
    except ValueError:
        inside_models = False

    if not inside_models:
        return "", "Model must be inside the current workspace models directory."
    if not abs_path.endswith(".bin") or not os.path.exists(abs_path):
        return "", "No valid .bin model selected."
    return abs_path, ""


def select_model_path(data):
    model_path = data.get("model") or data.get("model_path") or ""

    if not model_path and os.path.exists(CURRENT_MODEL_JSON):
        try:
            with open(CURRENT_MODEL_JSON, "r", encoding="utf-8") as f:
                current = json.load(f)
            model_path = current.get("model_path") or current.get("applied_model_path") or ""
        except Exception:
            model_path = ""

    if not model_path:
        models = list_models()
        if models:
            model_path = models[0]["path"]

    return model_path


def write_current_model_meta(meta):
    os.makedirs(os.path.dirname(CURRENT_MODEL_JSON), exist_ok=True)
    with open(CURRENT_MODEL_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def stage_model_file(model_path, note):
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    dst = os.path.join(DEPLOY_DIR, os.path.basename(model_path))
    if os.path.abspath(model_path) != os.path.abspath(dst):
        shutil.copy2(model_path, dst)

    meta = copy_model_metadata(model_path, dst, {
        "source_path": model_path,
        "deployed_at": iso_now(),
        "note": note,
    })
    write_current_model_meta(meta)
    append_model_history("model_staged", dst, meta, "Model staged for deployment.")
    return dst, meta


def find_detection_node_pids():
    pids = []
    proc_dir = "/proc"
    for name in os.listdir(proc_dir):
        if not name.isdigit():
            continue
        path = os.path.join(proc_dir, name, "cmdline")
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            continue
        if not raw:
            continue
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        if "detection_node" not in cmd:
            continue
        if "infrared_detection_node" in cmd:
            continue
        if "web_control_node" in cmd:
            continue
        pids.append(int(name))
    return sorted(set(pids))


def stop_detection_node():
    pids = find_detection_node_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + 4.0
    while time.time() < deadline:
        if not find_detection_node_pids():
            return pids
        time.sleep(0.2)

    for pid in find_detection_node_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(0.5)
    return pids


def detection_class_names_for_model(model_path):
    meta = read_model_metadata(model_path)
    names = meta.get("class_names")
    if isinstance(names, str):
        names = [item.strip() for item in names.split(",") if item.strip()]
    if not isinstance(names, list):
        names = []
    names = [str(item).strip() for item in names if str(item).strip()]

    class_count = 0
    try:
        class_count = int(meta.get("class_count") or 0)
    except Exception:
        class_count = 0
    if not class_count:
        for shape in meta.get("output_shapes") or []:
            try:
                last_dim = int(shape[-1])
            except Exception:
                continue
            if last_dim != 4:
                class_count = max(class_count, last_dim)

    if class_count > 0 and len(names) != class_count:
        names = [f"class_{idx}" for idx in range(class_count)]
    if not names:
        names = ["defect"]
    return names


def start_detection_node(model_path):
    class_arg = ",".join(detection_class_names_for_model(model_path))
    setup_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"source {shlex.quote(BASE_DIR)}/install/setup.bash && "
        "exec ros2 run detection_pkg detection_node --ros-args "
        f"-p model_path:={shlex.quote(model_path)} "
        f"-p class_names:={shlex.quote(class_arg)} "
        "-p conf_thresh:=0.7 "
        "-p nms_thresh:=0.45 "
        "-p center_threshold:=100"
    )

    os.makedirs(os.path.dirname(DETECTION_LOG), exist_ok=True)
    with open(DETECTION_LOG, "ab") as log:
        subprocess.Popen(
            ["/bin/bash", "-lc", setup_cmd],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    time.sleep(2.0)
    return find_detection_node_pids()


def safe_dataset_name(name):
    base = secure_filename(name or "")
    if not base:
        base = "dataset_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    return base


def normalize_dataset_split(value):
    split = str(value or "train").strip().lower()
    if split in ("val", "valid", "validation"):
        return "valid"
    if split == "test":
        return "test"
    return "train"


def validate_dataset_path(dataset_name):
    if not dataset_name:
        return "", "No dataset selected."

    safe_name = safe_dataset_name(dataset_name)
    dataset_path = os.path.abspath(os.path.join(DATASETS_DIR, safe_name))
    datasets_root = os.path.abspath(DATASETS_DIR)
    try:
        inside = os.path.commonpath([datasets_root, dataset_path]) == datasets_root
    except ValueError:
        inside = False

    if not inside or not os.path.isdir(dataset_path):
        return "", "Dataset must exist inside the current workspace datasets directory."
    return dataset_path, ""


def default_dataset_name():
    datasets = list_datasets()
    trainable = [
        item for item in datasets
        if item.get("data_yaml")
        and (item.get("has_best_pt") or item.get("has_yolo26n"))
        and int(item.get("val_images") or 0) > 0
    ]
    if trainable:
        return trainable[0]["name"]
    receivable = [item for item in datasets if item.get("data_yaml")]
    if receivable:
        return receivable[0]["name"]
    with_images = [item for item in datasets if int(item.get("images") or 0) > 0]
    if with_images:
        return with_images[0]["name"]
    if datasets:
        return datasets[0]["name"]
    return ""


def ensure_yolo_split_dirs(dataset_path, split):
    images_dir = os.path.join(dataset_path, split, "images")
    labels_dir = os.path.join(dataset_path, split, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return images_dir, labels_dir


def clean_upload_parts(raw_name, dataset_name=""):
    parts = [
        secure_filename(p)
        for p in str(raw_name or "").replace("\\", "/").split("/")
        if p and p not in (".", "..")
    ]
    if len(parts) > 1 and dataset_name and parts[0] == safe_dataset_name(dataset_name):
        parts = parts[1:]
    return parts


def upload_destination_parts(parts, dataset_name="", split="train"):
    parts = list(parts or [])
    if not parts:
        return []

    lower = [p.lower() for p in parts]
    for marker in ("train", "valid", "val", "test"):
        if marker in lower:
            idx = lower.index(marker)
            parts = parts[idx:]
            if parts and parts[0].lower() == "val":
                parts[0] = "valid"
            return parts

    filename = parts[-1]
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTS:
        return [normalize_dataset_split(split), "images", filename]
    if ext in LABEL_EXTS:
        return [normalize_dataset_split(split), "labels", filename]
    if len(parts) == 1:
        return [filename]

    return parts


def safe_join_under(root, parts):
    dest = os.path.abspath(os.path.join(root, *parts))
    root_abs = os.path.abspath(root)
    try:
        inside = os.path.commonpath([root_abs, dest]) == root_abs
    except ValueError:
        inside = False
    if not inside:
        return ""
    return dest


def dataset_root_score(path):
    if not os.path.isdir(path):
        return 0
    score = 0
    if find_dataset_yaml(path):
        score += 100
    if os.path.isdir(os.path.join(path, "images")):
        score += 30 + count_files(os.path.join(path, "images"), IMAGE_EXTS)
    if os.path.isdir(os.path.join(path, "labels")):
        score += 20 + count_files(os.path.join(path, "labels"), LABEL_EXTS)
    for split in ("train", "valid", "val", "test"):
        split_dir = os.path.join(path, split)
        image_dir = os.path.join(split_dir, "images")
        label_dir = os.path.join(split_dir, "labels")
        if os.path.isdir(image_dir):
            score += 40 + count_files(image_dir, IMAGE_EXTS)
        if os.path.isdir(label_dir):
            score += 20 + count_files(label_dir, LABEL_EXTS)
    return score


def find_best_dataset_root(target):
    best = target
    best_score = dataset_root_score(target)
    target_depth = target.rstrip(os.sep).count(os.sep)
    for root, dirs, _ in os.walk(target):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__MACOSX",)]
        if root != target:
            depth = root.rstrip(os.sep).count(os.sep) - target_depth
            if depth > 4:
                dirs[:] = []
                continue
        score = dataset_root_score(root)
        if score > best_score:
            best = root
            best_score = score
    return best if best_score > 0 else target


def merge_path_into(src, dst):
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(src):
            merge_path_into(os.path.join(src, name), os.path.join(dst, name))
        try:
            os.rmdir(src)
        except OSError:
            pass
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    final = dst
    if os.path.exists(final):
        stem, ext = os.path.splitext(dst)
        final = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}"
    shutil.move(src, final)


def promote_dataset_root(target, source):
    target = os.path.abspath(target)
    source = os.path.abspath(source)
    if source == target:
        return
    for name in os.listdir(source):
        merge_path_into(os.path.join(source, name), os.path.join(target, name))
    try:
        os.rmdir(source)
    except OSError:
        pass


def normalize_uploaded_yolo_dataset(target):
    best_root = find_best_dataset_root(target)
    promote_dataset_root(target, best_root)

    yml_path = os.path.join(target, "data.yml")
    yaml_path = os.path.join(target, "data.yaml")
    if os.path.exists(yml_path) and not os.path.exists(yaml_path):
        shutil.copy2(yml_path, yaml_path)

    root_images = os.path.join(target, "images")
    root_labels = os.path.join(target, "labels")
    has_split_images = any(
        os.path.isdir(os.path.join(target, split, "images"))
        for split in ("train", "valid", "test")
    )
    if os.path.isdir(root_images) and not has_split_images:
        merge_path_into(root_images, os.path.join(target, "train", "images"))
        if os.path.isdir(root_labels):
            merge_path_into(root_labels, os.path.join(target, "train", "labels"))

    if int(dataset_summary(target).get("images") or 0) > 0 and not os.path.exists(yaml_path):
        write_dataset_yaml(target, ["defect"])


def extract_uploaded_archive(archive_path, target, dataset_name):
    ext = os.path.splitext(archive_path)[1].lower()
    if ext == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                member_parts = upload_destination_parts(
                    clean_upload_parts(member.filename, dataset_name),
                    dataset_name,
                )
                member_dest = safe_join_under(target, member_parts)
                if not member_dest:
                    continue
                os.makedirs(os.path.dirname(member_dest), exist_ok=True)
                with zf.open(member) as src, open(member_dest, "wb") as out:
                    shutil.copyfileobj(src, out)
        return

    if ext in (".rar", ".7z"):
        extractor = shutil.which("7z") or shutil.which("7zz")
        if extractor:
            subprocess.run(
                [extractor, "x", "-y", f"-o{target}", archive_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return
        if ext == ".rar":
            unrar = shutil.which("unrar")
            if unrar:
                subprocess.run(
                    [unrar, "x", "-o+", archive_path, target + os.sep],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                return
        raise RuntimeError("当前板端未安装 unrar/7z，暂不能解压 .rar/.7z；请上传 YOLO 数据集文件夹或 .zip 压缩包。")

    raise RuntimeError("Unsupported archive type.")


def append_pipeline_log(log_path, text):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def read_tail(path, max_bytes=12000):
    if not path or not os.path.exists(path):
        return ""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", errors="replace")


def update_pipeline_state(**kwargs):
    with pipeline_lock:
        pipeline_state.update(kwargs)


def module_available(name):
    return importlib.util.find_spec(name) is not None


def first_existing_tool(*names):
    for name in names:
        path = os.path.join(TOOLS_DIR, name)
        if os.path.exists(path):
            return path
    return os.path.join(TOOLS_DIR, names[0])




def find_existing_bpu_bin():
    candidates = [
        os.path.join(DEPLOY_DIR, "yolo26_bpu_bayese_640x640_nv12_int8.bin"),
        APPLIED_MODEL_BIN,
        os.path.join(DEPLOY_DIR, "yolo26dino3_bpu_bayese_640x640_nv12.bin"),
        os.path.join(MODELS_DIR, "best_yolo26_bpu_bayese_640x640_nv12.bin"),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    if os.path.isdir(MODELS_DIR):
        for root, _, files in os.walk(MODELS_DIR):
            for name in sorted(files):
                path = os.path.join(root, name)
                if name.lower().endswith(".bin") and os.path.getsize(path) > 0:
                    return path
    return ""


def preflight_action(key, title, detail, command="", required_for=""):
    return {
        "key": key,
        "title": title,
        "detail": detail,
        "command": command,
        "required_for": required_for,
    }


def local_yolo26_worker_url():
    configured = os.getenv("YOLO26_LOCAL_WORKER_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    fallback = os.getenv("YOLO26_LOCAL_WORKER_FALLBACK", "").strip()
    try:
        host = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
    except RuntimeError:
        host = ""
    if host and host not in ("127.0.0.1", "::1", "localhost"):
        return f"http://{host}:8765"
    return fallback.rstrip("/") if fallback else "http://192.168.128.100:8765"


def local_yolo26_worker_status(timeout=0.8):
    url = local_yolo26_worker_url()
    if not url:
        return None
    try:
        req = urllib.request.Request(url + "/api/train/status", method="GET")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        payload["worker_url"] = url
        return payload
    except Exception:
        return None


def local_yolo26_worker_train(payload, timeout=4.0):
    url = local_yolo26_worker_url()
    if not url:
        return None, 0
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url + "/api/train",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8", errors="replace"))
        body["worker_url"] = url
        return body, getattr(resp, "status", 200)


def pipeline_preflight(dataset_name="", weights="", skip_train=False, distill_teacher=""):
    datasets = list_datasets()
    if not dataset_name and datasets:
        dataset_name = datasets[0]["name"]
    dataset_path = os.path.join(DATASETS_DIR, dataset_name) if dataset_name else ""
    data_yaml = os.path.join(dataset_path, "data.yaml") if dataset_path else ""
    valid_images = os.path.join(dataset_path, "valid", "images") if dataset_path else ""
    yolo26n = os.path.join(dataset_path, "yolo26n.pt") if dataset_path else ""
    best_pt = os.path.join(dataset_path, "runs", "detect", "train", "weights", "best.pt") if dataset_path else ""

    commands = {
        "yolo": shutil.which("yolo") or "",
        "hb_mapper": shutil.which("hb_mapper") or "",
        "docker": shutil.which("docker") or "",
        "python3": shutil.which("python3") or "",
    }
    modules = {
        "ultralytics": module_available("ultralytics"),
        "torch": module_available("torch"),
        "transformers": module_available("transformers"),
        "onnx": module_available("onnx"),
        "onnxruntime": module_available("onnxruntime"),
        "cv2": module_available("cv2"),
        "numpy": module_available("numpy"),
    }
    export_script = first_existing_tool(
        "export_yolo26_distilled_to_bpu_onnx.py",
        "export_yolo26_detect_bpu.py",
    )
    mapper_script = first_existing_tool(
        "mapper_yolo26_raw_images_to_bpu.py",
        "mapper.py",
    )
    distill_script = first_existing_tool(
        "dinov3_distill_yolo26.py",
        "train_dinov3_yolo26_distill.py",
    )
    local_distill_script = os.path.join(TOOLS_DIR, "train_dinov3_yolo26_distill.py")
    fallback_bin = find_existing_bpu_bin()
    can_fallback_bin = False
    dino_teacher_dir = str(distill_teacher or os.getenv("DINOV3_TEACHER_DIR", "")).strip()
    dino_teacher_ready = (
        not dino_teacher_dir
        or (
            os.path.isdir(dino_teacher_dir)
            and os.path.exists(os.path.join(dino_teacher_dir, "config.json"))
            and os.path.exists(os.path.join(dino_teacher_dir, "model.safetensors"))
        )
    )

    files = {
        "dataset": bool(dataset_path and os.path.isdir(dataset_path)),
        "data_yaml": bool(data_yaml and os.path.exists(data_yaml)),
        "valid_images": bool(valid_images and os.path.isdir(valid_images) and os.listdir(valid_images)),
        "yolo26n": bool(yolo26n and os.path.exists(yolo26n)),
        "best_pt": bool(best_pt and os.path.exists(best_pt)),
        "weights": bool(weights and os.path.exists(weights)),
        "export_script": os.path.exists(export_script),
        "mapper_script": os.path.exists(mapper_script),
        "distill_script": os.path.exists(distill_script),
        "local_distill_script": os.path.exists(local_distill_script),
        "dino_teacher": dino_teacher_ready,
        "fallback_bin": can_fallback_bin,
    }

    export_runtime_ok = modules["ultralytics"] and modules["torch"]
    quant_runtime_ok = modules["onnxruntime"] and (commands["hb_mapper"] or commands["docker"])

    missing = []
    if not files["dataset"]:
        missing.append("dataset")
    if not files["data_yaml"]:
        missing.append("data.yaml")
    if not files["valid_images"]:
        missing.append("valid/images calibration images")
    if skip_train:
        if not files["weights"]:
            missing.append("uploaded .pt weights")
    else:
        if not files["best_pt"] and not files["yolo26n"]:
            missing.append("yolo26n.pt or existing best.pt")
        if not commands["yolo"] and not export_runtime_ok and not files["best_pt"]:
            missing.append("YOLO training runtime: yolo command or ultralytics+torch")
    if not export_runtime_ok and not can_fallback_bin:
        missing.append("ONNX export runtime: ultralytics+torch")
    if not modules["onnxruntime"] and not can_fallback_bin:
        missing.append("mapper dependency: onnxruntime")
    if not commands["hb_mapper"] and not commands["docker"] and not can_fallback_bin:
        missing.append("BPU quantization runtime: hb_mapper or docker")
    for key in ("export_script", "mapper_script", "distill_script"):
        if not files[key]:
            missing.append(key)
    if dino_teacher_dir:
        if not files["dino_teacher"]:
            missing.append("DINOV3_TEACHER_DIR with config.json and model.safetensors")
        if not files["local_distill_script"]:
            missing.append("train_dinov3_yolo26_distill.py")
        if not modules["transformers"]:
            missing.append("DINOv3 distillation runtime: transformers")

    actions = []
    if not files["dataset"] or not files["data_yaml"]:
        actions.append(preflight_action("dataset", "??????? YOLO26 ???", "??????? data.yaml???? train/valid/test ?????", "", "????"))
    if not files["valid_images"]:
        actions.append(preflight_action("valid_images", "?????/????", f"BIN ???????????{valid_images}", "", "ONNX -> BIN ??"))
    if skip_train and not files["weights"]:
        actions.append(preflight_action("uploaded_pt", "???? PT ??", "????? PT ???????????????? .pt ???", "", "PT -> ONNX"))
    if not skip_train and not files["best_pt"] and not files["yolo26n"]:
        actions.append(preflight_action("yolo26n", "?? yolo26n.pt ??? best.pt", f"?????????????{yolo26n}", "", "YOLO26n ??"))
    if not skip_train and not commands["yolo"] and not export_runtime_ok and not files["best_pt"]:
        actions.append(preflight_action("train_runtime", "?? YOLO ?????", "?? yolo ???? Python ???? import ultralytics ? torch?", "python3 -m pip install ultralytics torch", "YOLO26n ??"))
    if can_fallback_bin and (not export_runtime_ok or not quant_runtime_ok):
        actions.append(preflight_action("fallback_bin", "\u590d\u7528\u677f\u7aef\u73b0\u6709 BPU bin \u5b8c\u6210\u4e00\u952e\u6d41\u7a0b", "\u5f53\u524d\u677f\u7aef\u6ca1\u6709\u5b8c\u6574 PT->ONNX->BIN \u7f16\u8bd1\u73af\u5883\uff1b\u672c\u6b21\u4f1a\u751f\u6210\u65b0\u4ea7\u7269\u8bb0\u5f55\u5e76\u590d\u5236\u5df2\u6709\u53ef\u8fd0\u884c bin\uff1a" + fallback_bin, "", "BIN \u540e\u5907\u90e8\u7f72"))
    if not export_runtime_ok and not can_fallback_bin:
        actions.append(preflight_action("export_runtime", "?? ONNX ?????", "PT -> ONNX ?????? ultralytics ? torch?", "python3 -m pip install ultralytics torch onnx", "PT -> ONNX"))
    if not modules["onnxruntime"] and not can_fallback_bin:
        actions.append(preflight_action("onnxruntime", "?? ONNXRuntime", "mapper ????? ONNX ???????????????", "python3 -m pip install onnxruntime", "ONNX -> BIN ??"))
    if not commands["hb_mapper"] and not commands["docker"] and not can_fallback_bin:
        actions.append(preflight_action("bpu_toolchain", "????? BPU ?????", "?? hb_mapper????? OpenExplorer Docker ???", "hb_mapper --version  # ? docker images | grep -i openexplorer", "ONNX -> BIN ??"))
    for key, path, title in (("export_script", export_script, "?? YOLO26 ONNX ??????"), ("mapper_script", mapper_script, "?? YOLO26 BPU mapper ????"), ("distill_script", distill_script, "?? DINOv3 ????????")):
        if not files[key]:
            actions.append(preflight_action(key, title, path, "", "?? pipeline"))
    if dino_teacher_dir and not files["dino_teacher"]:
        actions.append(preflight_action("dino_teacher", "???? DINOv3 teacher ??", "??????? config.json ? model.safetensors??????? preprocessor_config.json?", "", "?? DINOv3 ??"))
    if dino_teacher_dir and not modules["transformers"]:
        actions.append(preflight_action("transformers", "?? DINOv3 teacher ????", "?? DINOv3 ???? transformers ? safetensors?", "python3 -m pip install transformers safetensors", "?? DINOv3 ??"))

    train_stage_ok = files["weights"] if skip_train else (files["best_pt"] or (files["yolo26n"] and (commands["yolo"] or export_runtime_ok)))
    train_stage_detail = weights if skip_train else (best_pt or yolo26n)

    stages = [
        {"stage": "prepare", "ok": files["dataset"] and files["data_yaml"], "detail": data_yaml},
        {"stage": "train_pt", "ok": bool(train_stage_ok), "detail": train_stage_detail},
        {"stage": "dinov3_distill", "ok": files["distill_script"] and files["dino_teacher"] and (not dino_teacher_dir or modules["transformers"]), "detail": dino_teacher_dir or distill_script},
        {"stage": "export_onnx", "ok": (files["export_script"] and export_runtime_ok) or can_fallback_bin, "detail": export_script if export_runtime_ok else ("fallback " + fallback_bin)},
        {"stage": "quantize_bin", "ok": (files["mapper_script"] and quant_runtime_ok) or can_fallback_bin, "detail": commands["hb_mapper"] or commands["docker"] or ("fallback " + fallback_bin)},
        {"stage": "auto_apply", "ok": os.path.isdir(DEPLOY_DIR) or os.access(os.path.dirname(DEPLOY_DIR), os.W_OK), "detail": APPLIED_MODEL_BIN},
    ]

    return {"ok": not missing, "dataset": dataset_name, "dataset_path": dataset_path, "commands": commands, "modules": modules, "files": files, "fallback_bin": fallback_bin, "missing": sorted(set(missing)), "actions": actions, "stages": stages, "generated_at": iso_now()}


def pipeline_worker(dataset_name, epochs, imgsz, batch, weights="", skip_train=False, distill=True, defect_note="", distill_teacher=""):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = os.path.join(RUNS_DIR, "pipeline", run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "pipeline.log")
    dataset_path = os.path.join(DATASETS_DIR, dataset_name)

    update_pipeline_state(
        running=True,
        stage="prepare",
        status="running",
        message="Preparing training pipeline",
        dataset=dataset_name,
        model="",
        training_mode="uploaded_pt_convert" if skip_train else "yolo26n_train_distill",
        defect_note=defect_note,
        distill_teacher=distill_teacher,
        log_path=log_path,
        started_at=datetime.now().isoformat(),
        finished_at=None,
    )

    cmd = [
        "python3",
        PIPELINE_SCRIPT,
        "--dataset", dataset_path,
        "--output-dir", run_dir,
        "--epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--batch", str(batch),
    ]
    if weights:
        cmd += ["--weights", weights]
    if skip_train:
        cmd += ["--skip-train"]
    if distill:
        cmd += ["--distill", distill_teacher or os.getenv("DINOV3_TEACHER_DIR", "") or "dinov3"]
    if defect_note:
        cmd += ["--defect-note", defect_note]

    append_pipeline_log(log_path, "[pipeline] " + " ".join(cmd))

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            append_pipeline_log(log_path, line)
            lower = line.lower()
            if "distill" in lower or "dinov3" in lower or "蒸馏" in line:
                update_pipeline_state(stage="distill", message=line.strip())
            elif "training" in lower or "train" in lower:
                update_pipeline_state(stage="train", message=line.strip())
            elif "onnx" in lower:
                update_pipeline_state(stage="export", message=line.strip())
            elif "mapper" in lower or ".bin" in lower:
                update_pipeline_state(stage="quantize", message=line.strip())

        rc = proc.wait()
        produced = []
        for root, _, files in os.walk(run_dir):
            for name in files:
                if name.lower().endswith(".bin"):
                    src = os.path.join(root, name)
                    dst = os.path.join(MODELS_DIR, name)
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    if os.path.abspath(src) != os.path.abspath(dst):
                        shutil.copy2(src, dst)
                    produced.append(dst)

        if rc == 0 and produced:
            produced_model = produced[0]
            output_info = load_json_file(os.path.join(run_dir, "pipeline_output.json"), {})
            is_fallback_bin = bool(output_info.get("fallback"))
            meta = write_model_metadata(produced_model, {
                "model_id": os.path.splitext(os.path.basename(produced_model))[0],
                "dataset": dataset_name,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "training_mode": "fallback_existing_bpu_bin" if is_fallback_bin else ("uploaded_pt_convert" if skip_train else "yolo26n_train_dinov3_distill"),
                "source_model_path": weights,
                "fallback_bin": is_fallback_bin,
                "fallback_source_bin": output_info.get("source_bin", ""),
                "fallback_reason": output_info.get("reason", ""),
                "defect_note": defect_note,
                "distill_teacher": distill_teacher or os.getenv("DINOV3_TEACHER_DIR", "") or "dinov3",
                "training_started_at": pipeline_state.get("started_at"),
                "training_finished_at": datetime.now().isoformat(),
                "auto_apply": True,
            })
            append_model_history(
                "pipeline_bin_generated",
                produced_model,
                meta,
                ("Fallback BPU BIN staged; " if is_fallback_bin else "YOLO26 pipeline generated BIN; ") + "supplemented defect: " + (defect_note or "not specified"),
            )
            update_pipeline_state(stage="deploy", message="Applying generated BIN to detection_node")
            os.makedirs(DEPLOY_DIR, exist_ok=True)
            tmp_current = APPLIED_MODEL_BIN + ".tmp"
            shutil.copy2(produced_model, tmp_current)
            os.replace(tmp_current, APPLIED_MODEL_BIN)
            applied_meta = copy_model_metadata(produced_model, APPLIED_MODEL_BIN, {
                "applied_model_path": APPLIED_MODEL_BIN,
                "applied_at": iso_now(),
            })
            stopped_pids = stop_detection_node()
            started_pids = start_detection_node(APPLIED_MODEL_BIN)
            applied_meta.update({
                "stopped_detection_pids": stopped_pids,
                "started_detection_pids": started_pids,
            })
            write_model_metadata(APPLIED_MODEL_BIN, applied_meta)
            write_current_model_meta(applied_meta)
            append_model_history(
                "model_auto_applied",
                APPLIED_MODEL_BIN,
                applied_meta,
                "Pipeline auto-applied generated BIN to the live detection model.",
            )
            update_pipeline_state(
                running=False,
                stage="done",
                status="done",
                message="BIN model generated and applied",
                model=APPLIED_MODEL_BIN,
                finished_at=datetime.now().isoformat(),
            )
        elif rc == 0:
            update_pipeline_state(
                running=False,
                stage="done",
                status="done",
                message="Pipeline finished but no BIN was found",
                finished_at=datetime.now().isoformat(),
            )
        else:
            update_pipeline_state(
                running=False,
                stage="failed",
                status="failed",
                message=f"Pipeline failed with exit code {rc}",
                finished_at=datetime.now().isoformat(),
            )
    except Exception as exc:
        append_pipeline_log(log_path, "[error] " + str(exc))
        update_pipeline_state(
            running=False,
            stage="failed",
            status="failed",
            message=str(exc),
            finished_at=datetime.now().isoformat(),
        )


def load_llm_config():
    """
    从 ~/.openclaw/openclaw.json 读取模型配置。

    你的配置里应该类似：
    models.providers.custom-gateway.baseUrl
    models.providers.custom-gateway.apiKey
    models.providers.custom-gateway.models[0].id

    同时支持环境变量覆盖：
    LLM_BASE_URL
    LLM_API_KEY
    LLM_MODEL
    """
    base_url = os.getenv("LLM_BASE_URL", "")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")

    provider_name = os.getenv("LLM_PROVIDER", "custom-gateway")

    try:
        with open(OPENCLAW_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        providers = cfg.get("models", {}).get("providers", {})
        provider = providers.get(provider_name)

        if not provider and providers:
            # 如果 custom-gateway 不存在，就取第一个 provider
            provider = list(providers.values())[0]

        if provider:
            if not base_url:
                base_url = provider.get("baseUrl", "")

            if not api_key:
                api_key = provider.get("apiKey", "")

            if not model:
                models = provider.get("models", [])
                if models:
                    model = models[0].get("id", "")

        # 如果 agent 默认模型写了 custom-gateway/xxx，也可以从这里取 xxx
        if not model:
            primary = (
                cfg.get("agents", {})
                .get("defaults", {})
                .get("model", {})
                .get("primary", "")
            )
            if "/" in primary:
                model = primary.split("/", 1)[1]
            elif primary:
                model = primary

    except Exception:
        # 配置读失败时，使用环境变量或默认值
        pass

    if not base_url:
        base_url = "https://cursor.scihub.edu.kg/api/v1"

    if not model:
        model = "claude-sonnet-4-6"

    return base_url.rstrip("/"), api_key, model


def clean_assistant_reply(text):
    """
    Keep the right-side assistant reply display-friendly.
    The chat UI is plain text, so strip common Markdown emphasis markers.
    """
    text = str(text or "").strip()
    for token in ("**", "__", "###", "##", "#"):
        text = text.replace(token, "")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("* ", "- ")):
            stripped = stripped[2:].strip()
        lines.append(stripped)
    return "\n".join(lines).strip()


def call_direct_llm(message, context=None):
    """
    直接调用模型 API，不走 OpenClaw Gateway。
    """
    base_url, api_key, model = load_llm_config()

    if not api_key:
        return (
            "模型 API Key 未配置。请检查 "
            "/home/sunrise/.openclaw/openclaw.json 里的 "
            "models.providers.custom-gateway.apiKey。"
        )

    system_prompt = """
你是部署在 RDK X5 电缆缺陷检测系统中的 AI Agent 智能助手。
你能读取网页后端给出的真实上下文：摄像头画面、电机软件位置、检测结果、误检闭环、数据集、模型、训练部署状态和系统资源。
回答必须以当前上下文为准，不确定就明确说未上报或未收到，不要编造节点、画面、训练结果或部署结果。
涉及电机、训练、部署、入库等动作时，只能说明 local_action_result 里已经由后端白名单执行或拦截的结果。
如果普通视觉摄像头没有画面，要直接指出并给出优先排查项；如果红外画面正常，也要说明两路摄像头状态不同。
回答要简洁、工程化、适合比赛现场展示；优先给结论，再给下一步建议。
"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt.strip() + (
                    "\n\nFormatting rules: answer in concise Chinese plain text. "
                    "Do not use Markdown headings, bold markers, or asterisks. "
                    "Prefer short lines and simple punctuation."
                )
            },
            {
                "role": "user",
                "content": (
                    "用户指令：{}\n\n"
                    "当前 RDK X5 系统上下文：{}"
                ).format(
                    message,
                    json.dumps(context or {}, ensure_ascii=False)
                )
            }
        ],
        "temperature": 0.2,
        "max_tokens": 800,
        "stream": False
    }

    url = base_url + "/chat/completions"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)

            content = data["choices"][0]["message"]["content"]

            # 兼容某些接口返回 content 为 list 的情况
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
                content = "".join(parts)

            return clean_assistant_reply(content)

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return "模型接口 HTTP 错误：{}，返回内容：{}".format(e.code, body)

    except Exception as e:
        return "连接模型接口失败：{}".format(e)


def assistant_llm_failed(reply):
    """Return True when the direct LLM layer did not produce a usable answer."""
    text = str(reply or "")
    lower = text.lower()
    markers = (
        "api key 未配置",
        "连接模型接口失败",
        "模型接口 http 错误",
        "internal server error",
        "timed out",
        "timeout",
        "connection refused",
        "invalid_api_key",
    )
    return any(marker in lower for marker in markers)


def assistant_to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def assistant_mm_from_match(match):
    if not match:
        return None
    value = assistant_to_float(match.group(1))
    if value is None:
        return None
    unit = (match.group(2) or "mm").lower()
    if unit in ("厘米", "cm"):
        return value * 10.0
    if unit in ("米", "m"):
        return value * 1000.0
    return value


def assistant_extract_first_distance(text):
    match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(毫米|mm|厘米|cm|米|m)?",
        text,
        re.IGNORECASE,
    )
    return assistant_mm_from_match(match)


def assistant_extract_target_distance(text):
    patterns = (
        r"(?:移动到|移到|定位到|到达|去到|目标位置|绝对位置|位置到)\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*(毫米|mm|厘米|cm|米|m)?",
        r"(?:move\s*to|goto|go\s*to)\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*(毫米|mm|厘米|cm|米|m)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        value = assistant_mm_from_match(match)
        if value is not None:
            return value
    return None


def assistant_age_text(age_sec):
    age = assistant_to_float(age_sec)
    if age is None:
        return "未上报"
    if age < 1:
        return "刚刚更新"
    return f"{age:.1f}s 前更新"


def assistant_mm_text(value, digits=1):
    number = assistant_to_float(value)
    if number is None:
        return "未上报"
    return f"{number:.{digits}f} mm"


def assistant_camera_state(has_image, age_sec):
    if not has_image:
        return "未收到画面"
    age = assistant_to_float(age_sec)
    if age is not None and age > 5:
        return "画面可能卡住"
    return "画面正常"


def assistant_detection_text(detection):
    detection = detection or {}
    count = int(detection.get("count") or 0)
    if count <= 0:
        return "未检出缺陷"
    label = detection.get("top_label") or "defect"
    conf = detection.get("max_confidence")
    if conf is None:
        return f"检出 {count} 个目标，最高类别 {label}"
    try:
        pct = float(conf) * 100.0 if float(conf) <= 1.0 else float(conf)
        return f"检出 {count} 个目标，最高 {label} {pct:.1f}%"
    except Exception:
        return f"检出 {count} 个目标，最高类别 {label}"


def assistant_collect_context(client_context=None):
    client_context = client_context if isinstance(client_context, dict) else {}
    now = time.time()
    image_age = round(now - node.last_image_time, 3) if node and node.last_image_time else None
    raw_image_age = round(now - node.last_raw_image_time, 3) if node and node.last_raw_image_time else None
    ir_image_age = round(now - node.last_infrared_image_time, 3) if node and node.last_infrared_image_time else None
    ir_raw_age = round(now - node.last_infrared_raw_image_time, 3) if node and node.last_infrared_raw_image_time else None

    detection = parse_detection_summary(node.latest_detection_text if node else "")
    infrared_detection = parse_infrared_summary(node.latest_infrared_text if node else "")
    infrared_temperature = latest_infrared_temperature_sample()
    motion = parse_motor_motion_summary(node.latest_motor_motion_text if node else "")

    with ad_state_lock:
        ad_status_snapshot = dict(latest_ad_status) if latest_ad_status else {}
        ad_result_snapshot = dict(latest_ad_result) if latest_ad_result else {}
        ad_age = round(now - latest_ad_result_time, 3) if latest_ad_result_time else None
    impedance = summarize_impedance_for_fusion(ad_result_snapshot, ad_status_snapshot, ad_age)
    fusion = compute_fusion_summary(
        detection,
        infrared_detection,
        impedance,
        motor_motion=motion,
        infrared_temperature=infrared_temperature,
    )

    memory = get_memory_info()
    disk = get_disk_info()
    bpu = get_bpu_info()
    temperature_c = get_temperature_c()
    if temperature_c is None and bpu.get("temperature_c") is not None:
        temperature_c = bpu.get("temperature_c")

    try:
        models = list_models()
    except Exception:
        models = []
    try:
        datasets_total = dataset_totals()
    except Exception:
        datasets_total = {}
    try:
        fp_summary = false_positive_summary(limit=5)
    except Exception:
        fp_summary = {"count": 0, "today": 0, "recent": []}

    process_counts = get_node_process_counts()
    ros_nodes = []
    for item in EXPECTED_ROS_NODES:
        name = item["name"]
        state = "online" if process_counts.get(name, 0) > 0 else "offline"
        if name == "camera_node" and node and (node.latest_jpeg is not None or node.latest_raw_jpeg is not None):
            state = "online"
        if name == "infrared_camera_node" and node and (
            node.latest_infrared_jpeg is not None or node.latest_infrared_raw_jpeg is not None
        ):
            state = "online"
        if name == "detection_node" and detection.get("status") not in ("empty", "raw"):
            state = "online"
        if name == "infrared_detection_node" and infrared_detection.get("status") not in ("empty", "raw"):
            state = "online"
        if name == "motor_control_node" and motion.get("status") not in ("empty", "raw"):
            state = "online"
        if name == "web_control_node":
            state = "online"
        ros_nodes.append({
            "name": name,
            "topic": item["topic"],
            "role": item["role"],
            "state": state,
        })

    with pipeline_lock:
        pipeline_snapshot = dict(pipeline_state)

    return {
        "device": "RDK X5",
        "workspace": BASE_DIR,
        "web_port": 8010,
        "assistant_mode": "context_agent_with_local_fallback",
        "system": {
            "cpu_percent": get_cpu_percent(),
            "bpu_percent": bpu.get("percent"),
            "memory_percent": memory.get("percent"),
            "disk_percent": disk.get("percent"),
            "temperature_c": temperature_c,
            "uptime": get_uptime_info(),
        },
        "ros": {
            "nodes": ros_nodes,
            "offline": [item["name"] for item in ros_nodes if item["state"] == "offline"],
        },
        "video": {
            "image_topic": node.image_topic if node else "unknown",
            "raw_image_topic": node.raw_image_topic if node else "unknown",
            "frame_id": node.frame_id if node else 0,
            "has_image": bool(node and node.latest_jpeg is not None),
            "has_raw_image": bool(node and node.latest_raw_jpeg is not None),
            "image_age_sec": image_age,
            "raw_image_age_sec": raw_image_age,
            "stream_fps": node.actual_fps() if node else None,
        },
        "infrared": {
            "image_topic": node.infrared_image_topic if node else "unknown",
            "raw_image_topic": node.infrared_raw_image_topic if node else "unknown",
            "results_topic": node.infrared_results_topic if node else "unknown",
            "frame_id": node.infrared_frame_id if node else 0,
            "has_image": bool(node and node.latest_infrared_jpeg is not None),
            "has_raw_image": bool(node and node.latest_infrared_raw_jpeg is not None),
            "image_age_sec": ir_image_age,
            "raw_image_age_sec": ir_raw_age,
            "stream_fps": node.actual_infrared_fps() if node else None,
            "detection": infrared_detection,
            "temperature": infrared_temperature,
        },
        "motor": {
            "status": node.motor_status if node else "unknown",
            "direction": node.motor_direction if node else "unknown",
            "motion": motion,
        },
        "detection": detection,
        "fusion": fusion,
        "impedance": impedance,
        "datasets": {
            "totals": datasets_total,
            "false_positive": fp_summary,
        },
        "models": {
            "count": len(models),
            "current": current_model_summary(),
            "recent": models[:5],
        },
        "pipeline": pipeline_snapshot,
        "inspection": inspection_snapshot(),
        "client": client_context,
        "logs": node.logs[:8] if node else [],
    }


def assistant_is_question_or_diagnostic(text):
    lower = text.lower()
    hard_question_terms = (
        "为什么",
        "什么原因",
        "原因",
        "怎么",
        "如何",
        "怎么办",
        "不动",
        "没反应",
        "没画面",
        "没连",
        "没连接",
        "读数为0",
        "为0",
    )
    if any(term in text for term in hard_question_terms):
        return True
    if "?" in lower or "？" in text:
        return True
    if ("吗" in text or "么" in text) and "停止" not in text:
        return True
    return False


def classify_assistant_intent(text):
    lower = text.lower()
    intent = {
        "name": "general",
        "skill": "general_chat",
        "action": "",
        "distance_mm": None,
        "target_mm": None,
        "execute": False,
    }

    if any(term in text for term in ("开始检测", "启动检测", "开始自动检测", "启动自动检测")):
        intent.update({
            "name": "inspection",
            "skill": "industrial_inspection",
            "action": "inspection_start",
            "execute": True,
        })
        return intent
    if any(term in text for term in ("停止检测", "结束检测", "终止检测")):
        intent.update({
            "name": "inspection",
            "skill": "industrial_inspection",
            "action": "inspection_stop",
            "execute": True,
        })
        return intent
    if any(term in text for term in ("检测报告", "查看报告", "下载报告", "工业报告")):
        intent.update({
            "name": "inspection",
            "skill": "industrial_inspection",
            "action": "inspection_report",
        })
        return intent
    if any(term in text for term in ("检测进度", "检测任务状态", "自动检测状态")):
        intent.update({
            "name": "inspection",
            "skill": "industrial_inspection",
            "action": "inspection_status",
        })
        return intent

    camera_terms = ("摄像头", "相机", "画面", "视频", "camera", "video")
    infrared_terms = ("红外", "ir", "infrared")
    motor_terms = ("电机", "丝杆", "螺母", "位置", "行程", "正转", "反转", "移动", "move", "motor")
    fp_terms = ("误检", "误报", "负样本", "hard negative", "入库", "数据集")
    train_terms = ("训练", "train")
    deploy_terms = ("部署", "deploy", "模型应用", "上线模型")
    impedance_terms = ("阻抗", "ad5933", "开路", "通断", "电容", "水浸", "moisture", "impedance")
    fusion_terms = ("融合", "置信度", "多模态", "kalman", "卡尔曼")
    log_terms = ("日志", "事件", "最近记录", "运行记录", "log")



    if any(term in text or term in lower for term in ("技能", "skill", "你会什么", "能做什么", "帮助", "help")):
        intent["name"] = "skills"
        intent["skill"] = "skill_catalog"
        return intent

    llm_chat_terms = (
        "你叫什么", "你是谁", "你的名字", "叫什么名字", "自我介绍", "介绍一下你自己",
        "写一句", "写一段", "帮我写", "生成一", "讲个", "闲聊", "聊天",
    )
    if any(term in text or term in lower for term in llm_chat_terms):
        return intent

    if assistant_is_question_or_diagnostic(text):
        if any(term in text or term in lower for term in camera_terms + infrared_terms):
            intent["name"] = "camera_diagnostic"
            intent["skill"] = "camera_doctor"
        elif any(term in text or term in lower for term in motor_terms):
            intent["name"] = "motor_diagnostic"
            intent["skill"] = "motor_doctor"
        elif any(term in text for term in fp_terms):
            intent["name"] = "false_positive"
            intent["skill"] = "false_positive_loop"
        elif any(term in text or term in lower for term in train_terms + deploy_terms):
            intent["name"] = "pipeline_question"
            intent["skill"] = "pipeline_status"
        elif any(term in text or term in lower for term in impedance_terms):
            intent["name"] = "impedance"
            intent["skill"] = "impedance_doctor"
        elif any(term in text or term in lower for term in fusion_terms):
            intent["name"] = "fusion"
            intent["skill"] = "fusion_status"
        elif any(term in text or term in lower for term in ("状态", "健康", "在线", "节点", "系统", "status")):
            intent["name"] = "diagnostic"
            intent["skill"] = "device_doctor"
        else:
            return intent
        return intent

    if any(term in text or term in lower for term in camera_terms + infrared_terms):
        intent["name"] = "camera_status"
        intent["skill"] = "camera_doctor"
        return intent

    if any(term in text or term in lower for term in impedance_terms):
        intent["name"] = "impedance"
        intent["skill"] = "impedance_doctor"
        return intent

    if any(term in text or term in lower for term in fusion_terms):
        intent["name"] = "fusion"
        intent["skill"] = "fusion_status"
        return intent

    if any(term in text or term in lower for term in log_terms):
        intent["name"] = "logs"
        intent["skill"] = "log_summary"
        return intent

    if any(term in text or term in lower for term in motor_terms):
        target_mm = assistant_extract_target_distance(text)
        distance_mm = assistant_extract_first_distance(text)
        intent["name"] = "motor_control"
        intent["skill"] = "motor_control"

        if "停止" in text or "stop" in lower:
            intent.update({"action": "stop", "execute": True})
            return intent
        if any(term in text for term in ("归零", "清零", "置零", "回零", "软件位置归零")) or "zero" in lower:
            intent.update({"action": "zero", "execute": True})
            return intent
        if target_mm is not None:
            intent.update({"action": "move_to", "target_mm": target_mm, "execute": True})
            return intent
        if "正转" in text or "forward" in lower:
            intent.update({"action": "forward", "execute": True})
            return intent
        if "反转" in text or "reverse" in lower:
            intent.update({"action": "reverse", "execute": True})
            return intent
        intent["name"] = "motor_status"
        intent["skill"] = "motor_position"
        return intent

    if any(term in text or term in lower for term in fp_terms):
        intent["name"] = "false_positive"
        intent["skill"] = "false_positive_loop"
        return intent

    if any(term in text or term in lower for term in train_terms):
        intent["name"] = "train"
        intent["skill"] = "pipeline_status"
        return intent

    if any(term in text or term in lower for term in deploy_terms):
        intent["name"] = "deploy"
        intent["skill"] = "model_deploy"
        return intent

    if "模型" in text or "model" in lower:
        intent["name"] = "model"
        intent["skill"] = "model_status"
        return intent

    if any(term in text for term in ("查看状态", "状态", "健康", "在线", "status")):
        intent["name"] = "status"
        intent["skill"] = "device_status"
        return intent

    return intent


def execute_assistant_intent(intent, context):
    if not intent.get("execute"):
        return ""
    if not node:
        return "未执行：ROS2 Web 节点未就绪。"

    action = intent.get("action")
    motion = (context.get("motor") or {}).get("motion") or {}
    travel_min = assistant_to_float(motion.get("travel_min_mm"), 0.0)
    travel_max = assistant_to_float(motion.get("travel_max_mm"), 400.0)
    current = assistant_to_float(motion.get("position_mm"))

    if action == "inspection_start":
        client_time = ((context.get("client") or {}).get("client_time"))
        _, message, _ = start_inspection_session(client_time)
        return message

    if action == "inspection_stop":
        _, message, _ = stop_inspection_session()
        return message

    if action == "stop":
        node.motor_status = "stopped"
        node.send_command("stop")
        return "已执行安全停止：向 /motor/control 发布 stop。"

    if action == "forward":
        node.motor_direction = "forward"
        node.motor_status = "running"
        node.send_command("forward")
        return "已执行正转：向 /motor/control 发布 forward。连续运行时请注意行程边界。"

    if action == "reverse":
        node.motor_direction = "reverse"
        node.motor_status = "running"
        node.send_command("reverse")
        return "已执行反转：向 /motor/control 发布 reverse。连续运行时请注意行程边界。"

    if action == "zero":
        node.zero_software_position()
        return "已执行软件位置归零：当前位置按 0 mm 重新计数。"

    if action == "move_to":
        target_mm = assistant_to_float(intent.get("target_mm"))
        if target_mm is None:
            return "未执行：没有解析到目标位置。"
        if target_mm < travel_min or target_mm > travel_max:
            return f"未执行：目标 {target_mm:.1f} mm 超出软件行程 {travel_min:.0f}-{travel_max:.0f} mm。"
        command = node.move_motor_to(target_mm)
        return f"已执行定位移动：目标 {target_mm:.1f} mm，命令 {command}。"

    return ""


def assistant_status_lines(context):
    video = context.get("video") or {}
    infrared = context.get("infrared") or {}
    motor = context.get("motor") or {}
    motion = motor.get("motion") or {}
    datasets = (context.get("datasets") or {}).get("totals") or {}
    fp = (context.get("datasets") or {}).get("false_positive") or {}
    models = context.get("models") or {}
    current_model = models.get("current") or {}
    pipeline = context.get("pipeline") or {}
    system = context.get("system") or {}
    offline = (context.get("ros") or {}).get("offline") or []

    lines = [
        f"视觉摄像头：{assistant_camera_state(video.get('has_image'), video.get('image_age_sec'))}，{assistant_age_text(video.get('image_age_sec'))}。",
        f"红外摄像头：{assistant_camera_state(infrared.get('has_image'), infrared.get('image_age_sec'))}，FPS {infrared.get('stream_fps') if infrared.get('stream_fps') is not None else '未上报'}。",
        f"电机：{motor.get('status', 'unknown')}，方向 {motor.get('direction', 'unknown')}，软件位置 {assistant_mm_text(motion.get('position_mm'))}。",
        f"检测：{assistant_detection_text(context.get('detection') or {})}；融合判断 {((context.get('fusion') or {}).get('decision') or '未上报')}。",
        f"数据闭环：数据集 {datasets.get('count', 0)} 个，图像 {datasets.get('images', 0)} 张，误检样本 {fp.get('count', 0)} 个，今日 {fp.get('today', 0)} 个。",
        f"模型：{current_model.get('display_name') or current_model.get('name') or '未应用'}；训练状态 {pipeline.get('status', 'idle')} / {pipeline.get('stage', 'idle')}。",
        f"系统：CPU {system.get('cpu_percent')}%，内存 {system.get('memory_percent')}%，温度 {system.get('temperature_c')}℃。",
    ]
    if offline:
        lines.append("需关注节点：" + "、".join(offline[:6]) + "。")
    return lines


def assistant_skill_catalog():
    return [
        {"name": "industrial_inspection", "title": "自动工业检测", "prompt": "开始检测"},
        {"name": "device_status", "title": "整机状态", "prompt": "查看RDK X5状态"},
        {"name": "camera_doctor", "title": "摄像头排查", "prompt": "普通摄像头为什么没画面"},
        {"name": "motor_position", "title": "电机位置", "prompt": "当前位置是多少"},
        {"name": "motor_control", "title": "电机控制", "prompt": "停止电机"},
        {"name": "false_positive_loop", "title": "误检闭环", "prompt": "分析当前误检"},
        {"name": "pipeline_status", "title": "训练状态", "prompt": "查看训练状态"},
        {"name": "model_status", "title": "模型状态", "prompt": "当前模型是什么"},
        {"name": "model_deploy", "title": "部署助手", "prompt": "部署模型怎么做"},
        {"name": "fusion_status", "title": "融合置信度", "prompt": "查看融合状态"},
        {"name": "impedance_doctor", "title": "阻抗检测", "prompt": "查看阻抗状态"},
        {"name": "log_summary", "title": "日志摘要", "prompt": "查看最近日志"},
    ]


def assistant_use_fast_skill(intent):
    return intent.get("skill") in {
        "industrial_inspection",
        "device_status",
        "camera_doctor",
        "motor_doctor",
        "motor_position",
        "motor_control",
        "false_positive_loop",
        "pipeline_status",
        "model_status",
        "model_deploy",
        "fusion_status",
        "impedance_doctor",
        "log_summary",
        "skill_catalog",
        "device_doctor",
    }


def assistant_local_reply(text, context, intent, action_result):
    name = intent.get("name", "general")
    lines = []
    if action_result:
        lines.append(action_result)

    video = context.get("video") or {}
    infrared = context.get("infrared") or {}
    motor = context.get("motor") or {}
    motion = motor.get("motion") or {}
    datasets = (context.get("datasets") or {}).get("totals") or {}
    fp = (context.get("datasets") or {}).get("false_positive") or {}
    models = context.get("models") or {}
    current_model = models.get("current") or {}
    pipeline = context.get("pipeline") or {}
    impedance = context.get("impedance") or {}
    fusion = context.get("fusion") or {}
    inspection = context.get("inspection") or inspection_snapshot()

    if name == "inspection":
        status = inspection.get("status", "idle")
        if status == "running":
            lines.append(
                f"自动检测正在运行：当前位置 "
                f"{assistant_mm_text(inspection_current_position())}，目标 "
                f"{assistant_mm_text(inspection.get('target_position_mm'))}，"
                f"已记录 {len(inspection.get('defects') or [])} 个缺陷事件。"
            )
        elif status in ("stopping", "finalizing"):
            lines.append("电机正在停止，系统正在汇总阻抗和缺陷记录并生成 PDF 报告。")
        elif status == "completed":
            lines.append(
                f"检测已完成，共记录 {len(inspection.get('defects') or [])} 个缺陷事件。"
            )
            lines.append(
                "工业检测报告："
                + inspection_report_text(inspection.get("report_url"), "报告地址未生成")
            )
        elif status == "failed":
            lines.append(
                "检测任务失败："
                + inspection_report_text(inspection.get("error"), inspection.get("message"))
            )
        else:
            lines.append("当前没有自动检测任务。输入“开始检测”即可启动完整检测流程。")
    elif name in ("status", "diagnostic"):
        lines.append("我已读取当前板端状态：")
        lines.extend(assistant_status_lines(context))
    elif name == "skills":
        lines.append("已加载本地快速技能，常用问题会直接读取板端状态秒回：")
        for item in assistant_skill_catalog():
            lines.append(f"{item['title']}：{item['prompt']}")
    elif name in ("camera_status", "camera_diagnostic"):
        lines.append(
            f"普通视觉摄像头当前状态：{assistant_camera_state(video.get('has_image'), video.get('image_age_sec'))}。"
        )
        lines.append(
            f"红外摄像头当前状态：{assistant_camera_state(infrared.get('has_image'), infrared.get('image_age_sec'))}。"
        )
        if not video.get("has_image"):
            lines.append(
                "结论：网页后端没有收到普通摄像头帧，优先检查 camera_node、/camera/image_raw、摄像头供电和 USB/CSI 连接。"
            )
        if infrared.get("has_image") and not video.get("has_image"):
            lines.append("红外画面仍在更新，说明网页和推流链路可用，问题更偏向普通摄像头采集链路。")
    elif name in ("motor_status", "motor_diagnostic", "motor_control"):
        lines.append(
            f"电机状态：{motor.get('status', 'unknown')}，方向 {motor.get('direction', 'unknown')}。"
        )
        lines.append(
            f"软件位置：{assistant_mm_text(motion.get('position_mm'))}，行程范围 "
            f"{assistant_mm_text(motion.get('travel_min_mm'), 0)}-{assistant_mm_text(motion.get('travel_max_mm'), 0)}。"
        )
        lines.append(
            "当前位置控制走纯软件计数：丝杆导程 10 mm，细分 400 步/圈，约 0.025 mm/步，不依赖磁编器。"
        )
        if motion.get("last_error"):
            lines.append("最近电机错误：" + str(motion.get("last_error")))
    elif name == "fusion":
        lines.append(
            f"多模态融合：{fusion.get('decision', '未上报')}，等级 {fusion.get('level', 'unknown')}，"
            f"置信度 {fusion.get('confidence', '未上报')}。"
        )
        lines.append(f"视觉：{assistant_detection_text(context.get('detection') or {})}。")
        lines.append(
            f"红外：异常区域 {((infrared.get('detection') or {}).get('count') or 0)} 个，"
            f"告警 {((infrared.get('detection') or {}).get('alarm') or False)}。"
        )
        lines.append(
            f"阻抗：{impedance.get('decision', '未上报')}，可用通道 {impedance.get('available_count', 0)} 个。"
        )
    elif name == "impedance":
        lines.append(
            f"阻抗检测：{impedance.get('decision', '未上报')}，状态 {impedance.get('status', 'empty')}，"
            f"风险等级 {impedance.get('level', 'unknown')}。"
        )
        lines.append(
            f"可用检测通道 {impedance.get('available_count', 0)} 个，风险通道 {impedance.get('risk_count', 0)} 个。"
        )
        lines.append("如果需要重新测量，请到“阻抗检测”页执行对应动作，避免助手误触发硬件采样。")
    elif name == "false_positive":
        lines.append(
            f"误检闭环当前有 {fp.get('count', 0)} 条样本记录，今日新增 {fp.get('today', 0)} 条。"
        )
        lines.append(
            f"当前检测结果：{assistant_detection_text(context.get('detection') or {})}。"
        )
        lines.append(
            "建议先在误检闭环页填写误检原因，再确认入库；负样本会生成空标签，用于下一轮训练压低类似误报。"
        )
    elif name == "train":
        lines.append(
            f"训练条件：数据集 {datasets.get('count', 0)} 个，图像 {datasets.get('images', 0)} 张，标签 {datasets.get('labels', 0)} 个。"
        )
        lines.append(
            f"训练状态：{pipeline.get('status', 'idle')} / {pipeline.get('stage', 'idle')}。"
        )
        lines.append("建议在“训练与部署”页确认数据集、epochs、imgsz、batch 后启动，避免误触发长任务。")
    elif name == "deploy":
        lines.append(
            f"当前可见模型 {models.get('count', 0)} 个，已应用模型：{current_model.get('display_name') or current_model.get('name') or '未应用'}。"
        )
        lines.append("建议先在模型列表选择目标模型，再执行部署或应用；部署结果会写入模型历史。")
    elif name == "model":
        lines.append(
            f"当前模型：{current_model.get('display_name') or current_model.get('name') or '未应用'}。"
        )
        lines.append(f"模型库数量：{models.get('count', 0)}。")
    elif name == "pipeline_question":
        lines.append(
            f"训练部署状态：{pipeline.get('status', 'idle')} / {pipeline.get('stage', 'idle')}。"
        )
        lines.append("训练负责把误检样本纳入模型迭代；部署负责把选中的 bin 模型暂存或应用到检测节点。")
    elif name == "logs":
        logs = context.get("logs") or []
        if not logs:
            lines.append("最近日志为空，网页后端暂未记录新的关键事件。")
        else:
            lines.append("最近关键日志：")
            for item in logs[:6]:
                lines.append(f"{item.get('time', '--:--:--')} {item.get('source', 'system')}：{item.get('message', '')}")
    else:
        lines.append("我已接入当前网页后端状态，可以回答摄像头、电机、误检闭环、训练部署和模型状态。")
        lines.extend(assistant_status_lines(context)[:4])

    return clean_assistant_reply("\n".join(lines))


def assistant_suggestions(intent, context):
    name = intent.get("name", "general")
    if name == "inspection":
        status = inspection_snapshot().get("status")
        if status == "running":
            return ["检测进度", "停止检测", "查看融合状态", "当前位置是多少"]
        if status == "completed":
            return ["查看检测报告", "开始检测", "查看融合状态", "查看最近日志"]
        return ["开始检测", "检测进度", "查看阻抗状态", "查看RDK X5状态"]
    if name == "skills":
        return [item["prompt"] for item in assistant_skill_catalog()[:4]]
    if name in ("camera_status", "camera_diagnostic"):
        return ["查看RDK X5状态", "查看红外画面状态", "普通摄像头怎么排查"]
    if name in ("motor_status", "motor_diagnostic", "motor_control"):
        return ["当前位置是多少", "停止电机", "软件位置归零", "查看行程范围"]
    if name == "false_positive":
        return ["分析当前误检", "加入数据集", "开始训练", "查看数据集状态"]
    if name in ("train", "deploy", "model", "pipeline_question"):
        return ["查看训练状态", "当前模型是什么", "部署模型怎么做", "查看模型历史"]
    if name == "fusion":
        return ["查看融合状态", "查看阻抗状态", "普通摄像头为什么没画面"]
    if name == "impedance":
        return ["查看阻抗状态", "查看融合状态", "查看最近日志"]
    if name == "logs":
        return ["查看RDK X5状态", "查看训练状态", "普通摄像头为什么没画面"]
    return ["查看RDK X5状态", "普通摄像头为什么没画面", "当前位置是多少", "更多技能"]


# =========================
# ROS2 Web 节点
# =========================

class WebControlNode(Node):
    def __init__(self):
        super().__init__("web_control_node")

        # =========================
        # 参数
        # =========================
        self.declare_parameter(
            "html_path",
            HTML_PATH
        )

        # 默认显示检测后的画面
        # 如果只想先看原始摄像头，启动时传：
        # -p image_topic:=/camera/image_raw
        self.declare_parameter("image_topic", "/detection/annotated_image")
        self.declare_parameter("raw_image_topic", "/camera/image_raw")
        self.declare_parameter("infrared_image_topic", "/infrared/annotated_image")
        self.declare_parameter("infrared_raw_image_topic", "/infrared/image_raw")
        self.declare_parameter("infrared_results_topic", "/infrared/results")
        self.declare_parameter("preview_width", 480)
        self.declare_parameter("preview_height", 360)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("stream_fps", 10.0)
        self.declare_parameter("encode_raw_stream", False)
        self.declare_parameter("encode_infrared_raw_stream", False)

        self.html_path = self.get_parameter("html_path").value
        self.image_topic = self.get_parameter("image_topic").value
        self.raw_image_topic = self.get_parameter("raw_image_topic").value
        self.infrared_image_topic = self.get_parameter("infrared_image_topic").value
        self.infrared_raw_image_topic = self.get_parameter("infrared_raw_image_topic").value
        self.infrared_results_topic = self.get_parameter("infrared_results_topic").value
        self.preview_width = int(self.get_parameter("preview_width").value)
        self.preview_height = int(self.get_parameter("preview_height").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.stream_fps = float(self.get_parameter("stream_fps").value)
        self.encode_raw_stream = bool(self.get_parameter("encode_raw_stream").value)
        self.encode_infrared_raw_stream = bool(
            self.get_parameter("encode_infrared_raw_stream").value
        )

        # =========================
        # 状态变量
        # =========================
        self.bridge = CvBridge()

        self.latest_jpeg = None
        self.latest_raw_jpeg = None
        self.latest_infrared_jpeg = None
        self.latest_infrared_raw_jpeg = None
        self.frame_id = 0
        self.infrared_frame_id = 0
        self.last_image_time = 0.0
        self.last_raw_image_time = 0.0
        self.last_infrared_image_time = 0.0
        self.last_infrared_raw_image_time = 0.0
        self.frame_times = []
        self.infrared_frame_times = []
        self.infrared_raw_frame_times = []
        self.frame_lock = threading.Lock()

        self.motor_status = "stopped"
        self.motor_direction = "forward"
        self.latest_motor_motion_text = ""

        self.latest_detection_text = ""
        self.latest_infrared_text = ""
        self.latest_detection_time = 0.0
        self.latest_infrared_time = 0.0
        self.logs = []

        # =========================
        # 订阅图像
        # =========================
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.raw_image_sub = self.create_subscription(
            Image,
            self.raw_image_topic,
            self.raw_image_callback,
            qos_profile_sensor_data
        )

        self.infrared_image_sub = self.create_subscription(
            Image,
            self.infrared_image_topic,
            self.infrared_image_callback,
            qos_profile_sensor_data
        )

        self.infrared_raw_image_sub = self.create_subscription(
            Image,
            self.infrared_raw_image_topic,
            self.infrared_raw_image_callback,
            qos_profile_sensor_data
        )

        # 订阅电机状态
        self.status_sub = self.create_subscription(
            String,
            "/motor/status",
            self.status_callback,
            10
        )

        # 订阅电机方向
        self.direction_sub = self.create_subscription(
            String,
            "/motor/direction",
            self.direction_callback,
            10
        )

        self.motion_status_sub = self.create_subscription(
            String,
            "/motor/motion_status",
            self.motion_status_callback,
            10
        )

        # 订阅检测结果
        self.detection_sub = self.create_subscription(
            String,
            "/detection/results",
            self.detection_callback,
            10
        )

        self.infrared_results_sub = self.create_subscription(
            String,
            self.infrared_results_topic,
            self.infrared_results_callback,
            10
        )

        # 发布电机控制命令
        self.control_pub = self.create_publisher(
            String,
            "/motor/control",
            10
        )

        self.get_logger().info("Web控制节点已启动 - 稳定推流 + 直接大模型助手版")
        self.get_logger().info(f"HTML文件: {self.html_path}")
        self.get_logger().info(f"订阅图像话题: {self.image_topic}")
        self.get_logger().info(f"订阅原始图像话题: {self.raw_image_topic}")
        self.get_logger().info(f"订阅红外图像话题: {self.infrared_image_topic}")
        self.get_logger().info(f"订阅红外原始图像话题: {self.infrared_raw_image_topic}")
        self.get_logger().info(
            f"预览尺寸: {self.preview_width}x{self.preview_height}, "
            f"JPEG质量: {self.jpeg_quality}, FPS: {self.stream_fps}"
        )
        self.get_logger().info(
            f"原始流编码: raw={self.encode_raw_stream}, "
            f"infrared_raw={self.encode_infrared_raw_stream}"
        )
        self.get_logger().info("访问 http://<板子IP>:8010 查看界面")

    def _encode_image_msg(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if self.preview_width > 0 and self.preview_height > 0:
                src_h, src_w = cv_image.shape[:2]
                target_size = (self.preview_width, self.preview_height)
                is_upscale = self.preview_width > src_w or self.preview_height > src_h
                cv_image = cv2.resize(
                    cv_image,
                    target_size,
                    interpolation=cv2.INTER_LANCZOS4 if is_upscale else cv2.INTER_AREA
                )
                if is_upscale:
                    blurred = cv2.GaussianBlur(cv_image, (0, 0), 1.0)
                    cv_image = cv2.addWeighted(cv_image, 1.35, blurred, -0.35, 0)

            ok, buffer = cv2.imencode(
                ".jpg",
                cv_image,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )

            if not ok:
                self.get_logger().warn("JPEG 编码失败")
                return None

            return buffer.tobytes()

        except Exception as e:
            self.get_logger().error(f"图像转换/编码失败: {e}")
            return None

    def image_callback(self, msg):
        """
        接收 ROS 图像并提前编码成 JPEG。
        这样 /video_feed 只发送 JPEG bytes，避免网页端反复编码卡顿。
        """
        jpeg = self._encode_image_msg(msg)
        if jpeg is None:
            return

        with self.frame_lock:
            now = time.time()
            self.latest_jpeg = jpeg
            self.frame_id += 1
            self.last_image_time = now
            self.frame_times.append(now)
            cutoff = now - 5.0
            self.frame_times = [x for x in self.frame_times if x >= cutoff]

    def raw_image_callback(self, msg):
        if not self.encode_raw_stream and self.latest_jpeg is not None:
            self.last_raw_image_time = time.time()
            return
        jpeg = self._encode_image_msg(msg)
        if jpeg is None:
            return
        with self.frame_lock:
            self.latest_raw_jpeg = jpeg
            self.last_raw_image_time = time.time()

    def infrared_image_callback(self, msg):
        jpeg = self._encode_image_msg(msg)
        if jpeg is None:
            return
        with self.frame_lock:
            now = time.time()
            self.latest_infrared_jpeg = jpeg
            self.infrared_frame_id += 1
            self.last_infrared_image_time = now
            self.infrared_frame_times.append(now)
            cutoff = now - 5.0
            self.infrared_frame_times = [x for x in self.infrared_frame_times if x >= cutoff]

    def infrared_raw_image_callback(self, msg):
        with inspection_lock:
            inspection_running = inspection_state.get("status") == "running"
        if (
            not inspection_running
            and not self.encode_infrared_raw_stream
            and self.latest_infrared_jpeg is not None
        ):
            self.last_infrared_raw_image_time = time.time()
            return
        jpeg = self._encode_image_msg(msg)
        if jpeg is None:
            return
        with self.frame_lock:
            self.latest_infrared_raw_jpeg = jpeg
            now = time.time()
            self.last_infrared_raw_image_time = now
            self.infrared_raw_frame_times.append(now)
            cutoff = now - 5.0
            self.infrared_raw_frame_times = [
                x for x in self.infrared_raw_frame_times if x >= cutoff
            ]

    def status_callback(self, msg):
        self.motor_status = msg.data

    def direction_callback(self, msg):
        self.motor_direction = msg.data

    def motion_status_callback(self, msg):
        self.latest_motor_motion_text = msg.data

    def detection_callback(self, msg):
        self.latest_detection_text = msg.data
        self.latest_detection_time = time.time()
        summary = parse_detection_summary(msg.data)
        capture_timer = threading.Timer(
            0.12,
            inspection_capture_event,
            args=("vision", summary),
        )
        capture_timer.daemon = True
        capture_timer.start()

    def infrared_results_callback(self, msg):
        self.latest_infrared_text = msg.data
        self.latest_infrared_time = time.time()
        summary = parse_infrared_summary(msg.data)
        capture_timer = threading.Timer(
            0.12,
            inspection_capture_event,
            args=("infrared", summary),
        )
        capture_timer.daemon = True
        capture_timer.start()

    def send_command(self, command):
        msg = String()
        msg.data = command
        self.control_pub.publish(msg)
        self.add_log("motor", f"发送命令: {command}")
        self.get_logger().info(f"发送命令: {command}")

    def format_motor_command(self, command, speed_percent=None):
        if speed_percent is None:
            return command
        try:
            speed = max(10.0, min(100.0, float(speed_percent)))
        except Exception:
            return command
        return f"{command};speed_percent={speed:g}"

    def move_motor_to(self, target_mm, speed_percent=None):
        target_mm = float(target_mm)
        motion = parse_motor_motion_summary(self.latest_motor_motion_text)
        current = motion.get("position_mm")
        if current is not None:
            self.motor_direction = "forward" if target_mm >= float(current) else "reverse"
        command = self.format_motor_command(
            f"move_to_mm:{target_mm:g}",
            speed_percent,
        )
        self.send_command(command)
        self.motor_status = "running"
        return command

    def zero_software_position(self):
        self.send_command("zero_position")
        self.add_log("motor", "软件位置归零")
        self.get_logger().info("软件位置归零")

    def add_log(self, source, message):
        self.logs.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "source": source,
            "message": message
        })
        self.logs = self.logs[:50]

    def actual_fps(self):
        with self.frame_lock:
            times = list(self.frame_times)
        if len(times) < 2:
            return 0.0 if self.latest_jpeg is not None else None
        duration = times[-1] - times[0]
        if duration <= 0:
            return 0.0
        return round((len(times) - 1) / duration, 1)

    def actual_infrared_fps(self):
        with self.frame_lock:
            if self.encode_infrared_raw_stream and self.latest_infrared_raw_jpeg is not None:
                times = list(self.infrared_raw_frame_times)
                has_frame = True
            else:
                times = list(self.infrared_frame_times)
                has_frame = self.latest_infrared_jpeg is not None
        if len(times) < 2:
            return 0.0 if has_frame else None
        duration = times[-1] - times[0]
        if duration <= 0:
            return 0.0
        return round((len(times) - 1) / duration, 1)


# 全局节点实例
node = None


# =========================
# 视频流
# =========================

def make_waiting_jpeg(text="Waiting for camera..."):
    """生成等待画面 JPEG"""
    blank = np.zeros((360, 480, 3), dtype=np.uint8)

    for x in range(0, 480, 40):
        cv2.line(blank, (x, 0), (x, 360), (25, 40, 25), 1)

    for y in range(0, 360, 40):
        cv2.line(blank, (0, y), (480, y), (25, 40, 25), 1)

    cv2.putText(
        blank,
        text,
        (55, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 80),
        2
    )

    ok, buffer = cv2.imencode(
        ".jpg",
        blank,
        [cv2.IMWRITE_JPEG_QUALITY, 55]
    )

    return buffer.tobytes() if ok else b""


def generate_frames():
    """
    生成 MJPEG 视频流。优先输出检测标注图；检测节点未运行时回退到原始摄像头。
    """
    last_sent_frame_id = -1
    last_sent_raw_time = 0.0

    while True:
        try:
            if node and (node.latest_jpeg is not None or node.latest_raw_jpeg is not None):
                with node.frame_lock:
                    if node.latest_jpeg is not None:
                        jpeg = node.latest_jpeg
                        fid = node.frame_id
                        raw_time = None
                    else:
                        jpeg = node.latest_raw_jpeg
                        fid = None
                        raw_time = node.last_raw_image_time

                if fid is not None and fid == last_sent_frame_id:
                    time.sleep(0.02)
                    continue
                if raw_time is not None and raw_time == last_sent_raw_time:
                    time.sleep(0.02)
                    continue

                if fid is not None:
                    last_sent_frame_id = fid
                if raw_time is not None:
                    last_sent_raw_time = raw_time

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                    b"Pragma: no-cache\r\n"
                    b"Expires: 0\r\n\r\n"
                    + jpeg +
                    b"\r\n"
                )

                fps = node.stream_fps if node and node.stream_fps > 0 else 10.0
                time.sleep(max(0.01, 1.0 / fps))

            else:
                jpeg = make_waiting_jpeg()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache, no-store, must-revalidate\r\n\r\n"
                    + jpeg +
                    b"\r\n"
                )

                time.sleep(0.3)

        except GeneratorExit:
            break

        except Exception as e:
            print(f"视频流错误: {e}")
            time.sleep(0.2)


def generate_infrared_frames():
    """
    Generate the infrared MJPEG stream. Prefer annotated infrared frames and
    fall back to raw infrared camera frames while the detector starts.
    """
    last_sent_frame_id = -1
    last_sent_raw_time = 0.0

    while True:
        try:
            if node and (
                node.latest_infrared_jpeg is not None
                or node.latest_infrared_raw_jpeg is not None
            ):
                with node.frame_lock:
                    prefer_raw = (
                        node.encode_infrared_raw_stream
                        and node.latest_infrared_raw_jpeg is not None
                    )
                    if prefer_raw:
                        jpeg = node.latest_infrared_raw_jpeg
                        fid = None
                        raw_time = node.last_infrared_raw_image_time
                    elif node.latest_infrared_jpeg is not None:
                        jpeg = node.latest_infrared_jpeg
                        fid = node.infrared_frame_id
                        raw_time = None
                    else:
                        jpeg = node.latest_infrared_raw_jpeg
                        fid = None
                        raw_time = node.last_infrared_raw_image_time

                if fid is not None and fid == last_sent_frame_id:
                    time.sleep(0.02)
                    continue
                if raw_time is not None and raw_time == last_sent_raw_time:
                    time.sleep(0.02)
                    continue

                if fid is not None:
                    last_sent_frame_id = fid
                if raw_time is not None:
                    last_sent_raw_time = raw_time

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                    b"Pragma: no-cache\r\n"
                    b"Expires: 0\r\n\r\n"
                    + jpeg +
                    b"\r\n"
                )

                fps = node.stream_fps if node and node.stream_fps > 0 else 10.0
                time.sleep(max(0.01, 1.0 / fps))

            else:
                jpeg = make_waiting_jpeg("Waiting for infrared camera...")
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache, no-store, must-revalidate\r\n\r\n"
                    + jpeg +
                    b"\r\n"
                )
                time.sleep(0.3)

        except GeneratorExit:
            break

        except Exception as e:
            print(f"红外视频流错误: {e}")
            time.sleep(0.2)


# =========================
# Flask 路由





@app.route("/")
def index():
    """主页面：加载新版前端 HTML"""
    if not node:
        return "ROS2 Web 节点尚未初始化", 503

    html_path = node.html_path

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
            return render_template_string(html)
    except Exception as e:
        return (
            f"新版前端文件读取失败: {e}<br>"
            f"请检查文件是否存在: {html_path}",
            500
        )


@app.route("/video_feed")
def video_feed():
    """MJPEG 视频流接口"""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/infrared_feed")
def infrared_feed():
    """Infrared MJPEG stream endpoint."""
    return Response(
        generate_infrared_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no"
        }
    )


def request_speed_percent(data):
    if not isinstance(data, dict):
        return None
    raw = (
        data.get("speed_percent")
        if "speed_percent" in data
        else data.get("speed")
    )
    if raw is None:
        return None
    try:
        return max(10.0, min(100.0, float(raw)))
    except Exception:
        return None


@app.route("/forward", methods=["POST"])
def forward():
    """正转"""
    data = request.get_json(silent=True) or {}
    speed_percent = request_speed_percent(data)
    if node:
        node.motor_direction = "forward"
        node.motor_status = "running"
        node.send_command(node.format_motor_command("forward", speed_percent))

    return jsonify({
        "status": "ok",
        "command": "forward",
        "speed_percent": speed_percent,
    })


@app.route("/reverse", methods=["POST"])
def reverse():
    """反转"""
    data = request.get_json(silent=True) or {}
    speed_percent = request_speed_percent(data)
    if node:
        node.motor_direction = "reverse"
        node.motor_status = "running"
        node.send_command(node.format_motor_command("reverse", speed_percent))

    return jsonify({
        "status": "ok",
        "command": "reverse",
        "speed_percent": speed_percent,
    })


@app.route("/stop", methods=["POST"])
def stop():
    """停止电机"""
    if node:
        node.motor_status = "stopped"
        node.send_command("stop")

    return jsonify({
        "status": "ok",
        "command": "stop"
    })


@app.route("/move_to", methods=["POST"])
def move_to():
    """Move the lead-screw nut to an absolute software position in millimeters."""
    data = request.get_json(silent=True) or {}
    speed_percent = request_speed_percent(data)
    target = data.get("target_mm", data.get("position_mm", data.get("target", "")))
    try:
        target_mm = float(target)
    except Exception:
        return jsonify({
            "status": "error",
            "message": "target_mm must be a number",
        }), 400

    motion = parse_motor_motion_summary(node.latest_motor_motion_text if node else "")
    travel_min = float(motion.get("travel_min_mm", 0.0))
    travel_max = float(motion.get("travel_max_mm", 400.0))
    if target_mm < travel_min or target_mm > travel_max:
        return jsonify({
            "status": "error",
            "message": (
                f"target {target_mm:.2f} mm exceeds software travel "
                f"{travel_min:.0f}-{travel_max:.0f} mm"
            ),
            "target_mm": target_mm,
            "travel_min_mm": travel_min,
            "travel_max_mm": travel_max,
        }), 400

    if node:
        command = node.move_motor_to(target_mm, speed_percent=speed_percent)
        return jsonify({
            "status": "ok",
            "command": command,
            "target_mm": target_mm,
            "speed_percent": speed_percent,
        })

    return jsonify({
        "status": "error",
        "message": "ROS2 web node is not ready",
    }), 503


@app.route("/motor/zero_position", methods=["POST"])
def motor_zero_position():
    """Set the software nut position to 0 mm."""
    if node:
        node.zero_software_position()

    return jsonify({
        "status": "ok",
        "command": "zero_position"
    })


@app.route("/status")
def status():
    """旧前端兼容状态接口"""
    if node:
        image_age = None

        if node.last_image_time:
            image_age = round(time.time() - node.last_image_time, 3)

        return jsonify({
            "status": node.motor_status,
            "direction": node.motor_direction,
            "motion": parse_motor_motion_summary(node.latest_motor_motion_text),
            "image_topic": node.image_topic,
            "frame_id": node.frame_id,
            "image_age_sec": image_age,
            "has_image": node.latest_jpeg is not None
        })

    return jsonify({
        "status": "unknown",
        "direction": "unknown",
        "has_image": False
    })


@app.route("/api/status")
def api_status():
    """新版前端：RDK X5 状态接口"""
    image_age = None
    infrared_image_age = None

    if node and node.last_image_time:
        image_age = round(time.time() - node.last_image_time, 3)
    if node and node.last_infrared_image_time:
        infrared_image_age = round(time.time() - node.last_infrared_image_time, 3)

    cpu_percent = get_cpu_percent()
    memory = get_memory_info()
    disk = get_disk_info()
    bpu = get_bpu_info()
    temperature_c = get_temperature_c()
    if temperature_c is None and bpu.get("temperature_c") is not None:
        temperature_c = bpu.get("temperature_c")
    ros_graph = get_ros_graph_snapshot()
    ros_nodes = ros_graph.get("nodes", [])
    ros_node_status = get_ros_node_status(
        ros_nodes,
        graph_source=ros_graph.get("source", "unavailable"),
    )
    detection = parse_detection_summary(node.latest_detection_text if node else "")
    infrared = parse_infrared_summary(node.latest_infrared_text if node else "")
    infrared_temperature = latest_infrared_temperature_sample()
    motor_motion = parse_motor_motion_summary(node.latest_motor_motion_text if node else "")
    with ad_state_lock:
        ad_status_snapshot = dict(latest_ad_status) if latest_ad_status else {}
        ad_result_snapshot = dict(latest_ad_result) if latest_ad_result else {}
        ad_age = round(time.time() - latest_ad_result_time, 3) if latest_ad_result_time else None
    impedance = summarize_impedance_for_fusion(ad_result_snapshot, ad_status_snapshot, ad_age)
    fusion = compute_fusion_summary(
        detection,
        infrared,
        impedance,
        motor_motion=motor_motion,
        infrared_temperature=infrared_temperature,
    )
    fps = node.actual_fps() if node else None
    infrared_fps = node.actual_infrared_fps() if node else None
    fp_summary = false_positive_summary()
    totals = dataset_totals()
    current_model = current_model_summary()
    health_score = 100

    for value, warn_at in (
        (cpu_percent, 85),
        (bpu.get("percent"), 85),
        (memory.get("percent"), 85),
        (disk.get("percent"), 85),
        (temperature_c, 75),
        (bpu.get("temperature_c"), 80),
    ):
        if value is not None and value >= warn_at:
            health_score -= 12

    if node and not node.latest_jpeg:
        health_score -= 10

    for item in ros_node_status:
        if item["state"] == "offline":
            health_score -= 10

    health_score = max(0, min(100, health_score))

    return jsonify({
        "status": "ok",
        "device": "RDK X5",
        "workspace": BASE_DIR,
        "health": {
            "score": health_score,
            "level": "ok" if health_score >= 80 else "warn"
        },
        "system": {
            "cpu_percent": cpu_percent,
            "bpu_percent": bpu.get("percent"),
            "bpu": bpu,
            "memory": memory,
            "disk": disk,
            "temperature_c": temperature_c,
            "ros_nodes": ros_nodes,
            "ros_node_status": ros_node_status,
            "ros_graph": ros_graph,
            "uptime": get_uptime_info(),
            "time": datetime.now().isoformat()
        },
        "motor": {
            "status": node.motor_status if node else "unknown",
            "direction": node.motor_direction if node else "unknown",
            "motion": motor_motion
        },
        "fusion": fusion,
        "video": {
            "image_topic": node.image_topic if node else "unknown",
            "raw_image_topic": node.raw_image_topic if node else "unknown",
            "frame_id": node.frame_id if node else 0,
            "has_image": bool(node and node.latest_jpeg is not None),
            "has_raw_image": bool(node and node.latest_raw_jpeg is not None),
            "image_age_sec": image_age,
            "raw_image_age_sec": round(time.time() - node.last_raw_image_time, 3) if node and node.last_raw_image_time else None,
            "stream_fps": fps,
            "configured_fps": node.stream_fps if node else None
        },
        "detection": detection,
        "infrared": {
            "image_topic": node.infrared_image_topic if node else "unknown",
            "raw_image_topic": node.infrared_raw_image_topic if node else "unknown",
            "results_topic": node.infrared_results_topic if node else "unknown",
            "frame_id": node.infrared_frame_id if node else 0,
            "has_image": bool(node and node.latest_infrared_jpeg is not None),
            "has_raw_image": bool(node and node.latest_infrared_raw_jpeg is not None),
            "image_age_sec": infrared_image_age,
            "raw_image_age_sec": round(time.time() - node.last_infrared_raw_image_time, 3) if node and node.last_infrared_raw_image_time else None,
            "stream_fps": infrared_fps,
            "detection": infrared,
            "temperature": infrared_temperature,
        },
        "datasets": {
            "totals": totals,
            "false_positive": fp_summary,
        },
        "models": {
            "count": len(list_models()),
            "current": current_model,
        },
        "assistant": {
            "mode": "direct_llm",
            "config": OPENCLAW_CONFIG_PATH
        },
        "logs": node.logs[:30] if node else []
    })


@app.route("/api/inspection/status")
def api_inspection_status():
    return jsonify({
        "status": "ok",
        "inspection": inspection_snapshot(),
    })


@app.route("/api/inspection/start", methods=["POST"])
def api_inspection_start():
    data = request.get_json(silent=True) or {}
    ok, message, state = start_inspection_session(data.get("client_time"))
    return jsonify({
        "status": "ok" if ok else "error",
        "message": message,
        "inspection": state,
    }), 200 if ok else 409


@app.route("/api/inspection/stop", methods=["POST"])
def api_inspection_stop():
    ok, message, state = stop_inspection_session()
    return jsonify({
        "status": "ok" if ok else "error",
        "message": message,
        "inspection": state,
    }), 200 if ok else 409


@app.route("/api/inspection/report/<session_id>")
def api_inspection_report(session_id):
    safe_id = secure_filename(session_id)
    if safe_id != session_id:
        return jsonify({"status": "error", "message": "invalid session id"}), 400
    session_dir = os.path.abspath(os.path.join(INSPECTION_DIR, safe_id))
    report_path = ""

    state_path = os.path.join(session_dir, "inspection.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            candidate = os.path.abspath(str(state.get("report_path") or ""))
            if (
                candidate.startswith(session_dir + os.sep)
                and os.path.exists(candidate)
            ):
                report_path = candidate
        except Exception:
            report_path = ""

    if not report_path and os.path.isdir(session_dir):
        candidates = []
        for name in os.listdir(session_dir):
            if not name.lower().endswith(".pdf"):
                continue
            if name.startswith("检测报告_") or name == "industrial_inspection_report.pdf":
                path = os.path.join(session_dir, name)
                candidates.append((os.path.getmtime(path), path))
        if candidates:
            candidates.sort(reverse=True)
            report_path = candidates[0][1]

    if not os.path.exists(report_path):
        return jsonify({"status": "error", "message": "report not found"}), 404
    return send_file(
        report_path,
        as_attachment=True,
        download_name=os.path.basename(report_path),
        mimetype="application/pdf",
    )


@app.route("/api/inspection/image/<session_id>/<path:filename>")
def api_inspection_image(session_id, filename):
    safe_id = secure_filename(session_id)
    rel = str(filename or "").replace("\\", "/")
    parts = [part for part in rel.split("/") if part]
    if (
        safe_id != session_id
        or not parts
        or any(part in (".", "..") for part in parts)
        or rel.startswith("/")
    ):
        return jsonify({"status": "error", "message": "invalid path"}), 400
    if len(parts) == 1:
        parts = ["images", parts[0]]
    session_dir = os.path.abspath(os.path.join(INSPECTION_DIR, safe_id))
    image_path = os.path.abspath(os.path.join(session_dir, *parts))
    try:
        if os.path.commonpath([session_dir, image_path]) != session_dir:
            return jsonify({"status": "error", "message": "invalid path"}), 400
    except ValueError:
        return jsonify({"status": "error", "message": "invalid path"}), 400
    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "image not found"}), 404
    return send_file(image_path, mimetype="image/jpeg")


@app.route("/api/ad5933/status")
def api_ad5933_status():
    """AD5933 module status and saved calibration summary."""
    global latest_ad_status
    data, code = run_ad5933_action("status", timeout=10)
    data.update({
        "status": "ok" if code == 200 else "error",
        "installed": os.path.exists(AD_RUNNER_SCRIPT),
        "module_dir": AD_ANALYZER_DIR,
    })
    if code == 200:
        with ad_state_lock:
            latest_ad_status = data
    return jsonify(data), code


@app.route("/api/ad5933/action", methods=["POST"])
def api_ad5933_action():
    """Run one AD5933 cable analyzer action."""
    global latest_ad_result, latest_ad_result_time
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action", "")).strip()
    data, code = run_ad5933_action(action, payload)
    data["status"] = "ok" if code == 200 else "error"
    if action != "status":
        with ad_state_lock:
            latest_ad_result = data
            latest_ad_result_time = time.time()
    if node:
        node.add_log("ad5933", f"{action}: {data.get('message') or data.get('error') or code}")
    return jsonify(data), code


@app.route("/api/fusion/status")
def api_fusion_status():
    """Return the current visual + impedance + infrared fusion summary."""
    detection = parse_detection_summary(node.latest_detection_text if node else "")
    infrared = parse_infrared_summary(node.latest_infrared_text if node else "")
    with ad_state_lock:
        ad_status_snapshot = dict(latest_ad_status) if latest_ad_status else {}
        ad_result_snapshot = dict(latest_ad_result) if latest_ad_result else {}
        ad_age = round(time.time() - latest_ad_result_time, 3) if latest_ad_result_time else None
    impedance = summarize_impedance_for_fusion(ad_result_snapshot, ad_status_snapshot, ad_age)
    return jsonify({
        "status": "ok",
        "fusion": compute_fusion_summary(
            detection,
            infrared,
            impedance,
            motor_motion=parse_motor_motion_summary(node.latest_motor_motion_text if node else ""),
            infrared_temperature=latest_infrared_temperature_sample(),
        ),
        "detection": detection,
        "infrared": infrared,
        "impedance": impedance,
    })


@app.route("/api/detections")
def api_detections():
    """返回最近一次检测结果"""
    if not node or not node.latest_detection_text:
        return jsonify({
            "status": "empty",
            "detections": []
        })

    try:
        return jsonify({
            "status": "ok",
            "detections": json.loads(node.latest_detection_text)
        })
    except Exception:
        return jsonify({
            "status": "raw",
            "raw": node.latest_detection_text
        })


@app.route("/api/review_pool/file/<path:relpath>")
def api_review_pool_file(relpath):
    safe = safe_join_under(REVIEW_POOL_DIR, clean_upload_parts(relpath))
    if not safe or not os.path.exists(safe):
        return jsonify({"status": "error", "message": "file not found"}), 404
    return send_from_directory(os.path.dirname(safe), os.path.basename(safe))


@app.route("/api/review_pool")
def api_review_pool():
    with review_pool_lock:
        items = read_review_pool()
    datasets = list_datasets()
    dataset_classes = {}
    for ds in datasets:
        name = ds.get("name", "")
        if name:
            dataset_classes[name] = read_dataset_classes(os.path.join(DATASETS_DIR, name))
    default_name = default_dataset_name()
    classes = dataset_classes.get(default_name) or (next(iter(dataset_classes.values()), ["defect"]))
    enriched = []
    for item in items:
        row = dict(item)
        row["image_url"] = image_url_for_path(row.get("image_path", ""))
        label_path = row.get("label_path", "")
        if label_path and os.path.exists(label_path):
            try:
                row["label_text"] = open(label_path, "r", encoding="utf-8").read()
            except Exception:
                row["label_text"] = ""
        enriched.append(row)
    return jsonify({
        "status": "ok",
        "samples": list(reversed(enriched)),
        "summary": review_pool_summary(items),
        "audit": read_loop_audit(limit=40),
        "classes": classes,
        "dataset_classes": dataset_classes,
        "datasets": datasets,
    })


@app.route("/api/review_pool/capture", methods=["POST"])
def api_review_pool_capture():
    data = request.get_json(silent=True) or {}
    jpeg = None
    source = "none"
    if node:
        with node.frame_lock:
            if node.latest_jpeg:
                jpeg = node.latest_jpeg
                source = node.image_topic
            elif node.latest_raw_jpeg:
                jpeg = node.latest_raw_jpeg
                source = node.raw_image_topic

    if not jpeg:
        return jsonify({
            "status": "error",
            "message": "No current camera frame is available for capture.",
        }), 409

    roi = data.get("roi") or {}
    image_bytes = jpeg
    try:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None and roi:
            h, w = img.shape[:2]
            x = max(0.0, min(1.0, float(roi.get("x", 0))))
            y = max(0.0, min(1.0, float(roi.get("y", 0))))
            rw = max(0.01, min(1.0, float(roi.get("w", 1))))
            rh = max(0.01, min(1.0, float(roi.get("h", 1))))
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = max(x1 + 1, min(w, int((x + rw) * w)))
            y2 = max(y1 + 1, min(h, int((y + rh) * h)))
            crop = img[y1:y2, x1:x2]
            ok, encoded = cv2.imencode(".jpg", crop)
            if ok:
                image_bytes = encoded.tobytes()
                source += " roi"
    except Exception:
        pass

    ensure_review_pool_dirs()
    sample_id = datetime.now().strftime("POOL_%Y%m%d_%H%M%S_%f")[:-3]
    image_path = os.path.join(REVIEW_POOL_IMAGES_DIR, sample_id + ".jpg")
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    record = make_sample_record(
        sample_id,
        image_path,
        source,
        roi=roi,
        note=data.get("note", ""),
    )
    with review_pool_lock:
        items = read_review_pool()
        items.append(record)
        write_review_pool(items)
    append_loop_audit("capture_to_review_pool", {"sample_id": sample_id, "roi": roi})
    if node:
        node.add_log("dataset", f"待审核样本池拍照: {sample_id}")
    return jsonify({"status": "ok", "sample": record, "image_url": image_url_for_path(image_path)})


@app.route("/api/review_pool/upload", methods=["POST"])
def api_review_pool_upload():
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        return jsonify({"status": "error", "message": "No sample files were uploaded."}), 400

    ensure_review_pool_dirs()
    labels = labels_by_uploaded_stem(files)
    created = []
    for item in files:
        raw_name = item.filename or ""
        base = os.path.basename(raw_name.replace("\\", "/"))
        safe_name = secure_filename(base)
        stem, ext = os.path.splitext(safe_name)
        ext = ext.lower()
        if ext not in IMAGE_EXTS:
            continue
        sample_id = datetime.now().strftime("POOL_%Y%m%d_%H%M%S_%f")[:-3] + "_" + uuid.uuid4().hex[:4]
        image_path = os.path.join(REVIEW_POOL_IMAGES_DIR, sample_id + ext)
        item.save(image_path)
        label_path = ""
        label_text = normalize_label_text(labels.get(stem, ""))
        if label_text:
            label_path = os.path.join(REVIEW_POOL_LABELS_DIR, sample_id + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(label_text + "\n")
        created.append(make_sample_record(
            sample_id,
            image_path,
            "uploaded:" + raw_name,
            label_path=label_path,
            original_name=raw_name,
        ))

    if not created:
        return jsonify({"status": "error", "message": "No image files were found in uploaded samples."}), 400

    with review_pool_lock:
        items = read_review_pool()
        items.extend(created)
        write_review_pool(items)
    append_loop_audit("upload_to_review_pool", {"count": len(created)})
    if node:
        node.add_log("dataset", f"待审核样本上传: {len(created)}")
    return jsonify({"status": "ok", "count": len(created), "samples": created})


@app.route("/api/review_pool/annotate", methods=["POST"])
def api_review_pool_annotate():
    data = request.get_json(silent=True) or {}
    sample_id = data.get("sample_id", "")
    label_text = normalize_label_text(data.get("label_text", ""))
    defect_class = str(data.get("defect_class") or "").strip()
    note = str(data.get("note") or "").strip()
    if not sample_id:
        return jsonify({"status": "error", "message": "No sample selected."}), 400
    if not label_text:
        return jsonify({"status": "error", "message": "YOLO26 label text is empty or invalid."}), 400

    with review_pool_lock:
        items = read_review_pool()
        target = None
        for item in items:
            if item.get("sample_id") == sample_id:
                target = item
                break
        if not target:
            return jsonify({"status": "error", "message": "Sample not found."}), 404
        label_path = os.path.join(REVIEW_POOL_LABELS_DIR, sample_id + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(label_text + "\n")
        target.update({
            "label_path": label_path,
            "annotated": True,
            "status": "annotated",
            "defect_class": defect_class,
            "note": note,
            "annotated_at": iso_now(),
        })
        write_review_pool(items)

    append_loop_audit("annotate_review_sample", {"sample_id": sample_id, "defect_class": defect_class})
    if node:
        node.add_log("dataset", f"待审核样本标注: {sample_id}")
    return jsonify({"status": "ok", "sample": target})


@app.route("/api/review_pool/import", methods=["POST"])
def api_review_pool_import():
    data = request.get_json(silent=True) or {}
    sample_ids = data.get("sample_ids") or []
    dataset_name = data.get("dataset") or data.get("dataset_name") or default_dataset_name()
    split_policy = str(data.get("split_policy") or "ratio_8_2")
    defect_class = str(data.get("defect_class") or "").strip()
    if not sample_ids:
        return jsonify({"status": "error", "message": "No review samples selected."}), 400

    dataset_path, error = validate_dataset_path(dataset_name)
    if error:
        return jsonify({"status": "error", "message": error}), 400

    imported = []
    skipped = []
    with review_pool_lock:
        items = read_review_pool()
        by_id = {x.get("sample_id"): x for x in items}
        selected = [by_id.get(x) for x in sample_ids if by_id.get(x)]
        missing_ids = [x for x in sample_ids if x not in by_id]
        annotated = [x for x in selected if x.get("annotated") and os.path.exists(x.get("label_path", ""))]
        for sample_id in missing_ids:
            skipped.append({"sample_id": sample_id, "reason": "sample not found"})
        for item in selected:
            if item not in annotated:
                skipped.append({"sample_id": item.get("sample_id"), "reason": "not annotated"})

        if not selected:
            return jsonify({
                "status": "error",
                "message": "Selected review samples were not found.",
                "skipped": skipped,
            }), 404
        if not annotated:
            return jsonify({
                "status": "error",
                "message": "No selected review samples have YOLO26 labels; annotate them before importing.",
                "skipped": skipped,
            }), 400

        class_id, names = ensure_dataset_class(dataset_path, defect_class)

        for idx, item in enumerate(annotated):
            if split_policy == "force_train":
                split = "train"
            elif split_policy in ("force_valid", "force_val"):
                split = "valid"
            elif split_policy == "force_test":
                split = "test"
            else:
                split = "train" if idx < int(len(annotated) * 0.8 + 0.5) else "valid"
            images_dir, labels_dir = ensure_yolo_split_dirs(dataset_path, split)
            ext = os.path.splitext(item.get("image_path", ""))[1].lower() or ".jpg"
            dst_image = os.path.join(images_dir, item["sample_id"] + ext)
            dst_label = os.path.join(labels_dir, item["sample_id"] + ".txt")
            shutil.copy2(item["image_path"], dst_image)
            try:
                raw_label_text = open(item["label_path"], "r", encoding="utf-8").read()
            except Exception:
                raw_label_text = ""
            dataset_label_text = rewrite_label_class(raw_label_text, class_id)
            with open(dst_label, "w", encoding="utf-8") as f:
                f.write(dataset_label_text + ("\n" if dataset_label_text else ""))
            item.update({
                "status": "imported",
                "dataset": safe_dataset_name(dataset_name),
                "split": split,
                "dataset_image_path": dst_image,
                "dataset_label_path": dst_label,
                "defect_class": defect_class or item.get("defect_class", ""),
                "class_id": class_id,
                "imported_at": iso_now(),
            })
            imported.append({
                "sample_id": item["sample_id"],
                "split": split,
                "dataset_image_path": dst_image,
                "dataset_label_path": dst_label,
            })
        write_review_pool(items)

    append_loop_audit("import_review_samples", {
        "dataset": dataset_name,
        "split_policy": split_policy,
        "defect_class": defect_class,
        "class_names": names,
        "imported": len(imported),
        "skipped": skipped,
    })
    if node:
        node.add_log("dataset", f"待审核样本导入: dataset={dataset_name}, imported={len(imported)}")
    return jsonify({
        "status": "ok",
        "imported": imported,
        "skipped": skipped,
        "dataset": dataset_summary(dataset_path),
    })


@app.route("/api/review_pool/export", methods=["POST"])
def api_review_pool_export():
    data = request.get_json(silent=True) or {}
    sample_ids = data.get("sample_ids") or []
    dataset_name = data.get("dataset") or data.get("dataset_name") or "review_pool_export"
    defect_class = str(data.get("defect_class") or "").strip()

    with review_pool_lock:
        items = read_review_pool()

    by_id = {x.get("sample_id"): x for x in items}
    if sample_ids:
        selected = [by_id.get(x) for x in sample_ids if by_id.get(x)]
    else:
        selected = [x for x in items if x.get("annotated")]

    exported = []
    skipped = []
    class_ids = set()
    export_class_id = 0 if defect_class else None
    for item in selected:
        image_path = item.get("image_path", "")
        label_path = item.get("label_path", "")
        if not item.get("annotated") or not os.path.exists(image_path) or not os.path.exists(label_path):
            skipped.append({"sample_id": item.get("sample_id", ""), "reason": "missing image or yolo label"})
            continue
        try:
            label_text = open(label_path, "r", encoding="utf-8").read()
        except Exception:
            label_text = ""
        if export_class_id is not None:
            label_text = rewrite_label_class(label_text, export_class_id)
        class_ids.update(class_ids_from_label_text(label_text))
        exported.append((item, image_path, label_path, label_text))

    if not exported:
        return jsonify({
            "status": "error",
            "message": "No annotated review samples are available for export.",
            "skipped": skipped,
        }), 400

    max_cls = max(class_ids) if class_ids else 0
    names = ["defect" for _ in range(max_cls + 1)]
    if defect_class:
        for idx in range(len(names)):
            names[idx] = defect_class if idx == 0 else f"{defect_class}_{idx}"
    for item, _, _, _ in exported:
        cls = str(item.get("defect_class") or "").strip()
        if cls:
            names[0] = cls

    mem = io.BytesIO()
    manifest = {
        "created_at": iso_now(),
        "source": "review_sample_pool",
        "dataset": dataset_name,
        "format": "yolo26",
        "exported_count": len(exported),
        "class_rewrite": {"enabled": export_class_id is not None, "class_id": export_class_id},
        "skipped": skipped,
        "samples": [],
    }
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        data_yaml = [
            "path: .",
            "train: images",
            "val: images",
            "test: images",
            f"nc: {len(names)}",
            "names:",
        ]
        data_yaml += [f"  {idx}: {name}" for idx, name in enumerate(names)]
        zf.writestr("data.yaml", "\n".join(data_yaml) + "\n")
        for item, image_path, _, label_text in exported:
            sample_id = item.get("sample_id", os.path.splitext(os.path.basename(image_path))[0])
            ext = os.path.splitext(image_path)[1].lower() or ".jpg"
            image_arc = f"images/{sample_id}{ext}"
            label_arc = f"labels/{sample_id}.txt"
            zf.write(image_path, image_arc)
            zf.writestr(label_arc, normalize_label_text(label_text) + "\n")
            manifest["samples"].append({
                "sample_id": sample_id,
                "image": image_arc,
                "label": label_arc,
                "defect_class": item.get("defect_class", ""),
                "source": item.get("source", ""),
            })
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    mem.seek(0)
    append_loop_audit("export_review_samples", {
        "dataset": dataset_name,
        "exported": len(exported),
        "skipped": skipped,
    })
    if node:
        node.add_log("dataset", f"待审核样本导出: exported={len(exported)}")
    filename = safe_dataset_name(dataset_name) + "_yolo26_review_export.zip"
    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )






@app.route("/api/dataset/false_positive", methods=["POST"])
def api_false_positive():
    """
    误检入库接口：
    - 保存当前视频帧 JPEG 到 false_positive_samples 留痕目录
    - 写入所选 YOLO 数据集 split/images 和 split/labels
    - 写入 metadata.jsonl
    """
    if request.content_type and request.content_type.startswith("multipart/"):
        data = request.form.to_dict()
    else:
        data = request.get_json(silent=True) or {}

    dataset_name = data.get("dataset") or data.get("dataset_name") or ""
    if not dataset_name:
        dataset_name = default_dataset_name()

    dataset_path, error = validate_dataset_path(dataset_name)
    if error:
        return jsonify({
            "status": "error",
            "message": error,
        }), 400

    split = normalize_dataset_split(data.get("split", "train"))
    images_dir, labels_dir = ensure_yolo_split_dirs(dataset_path, split)

    save_dir = FP_SAMPLE_DIR
    os.makedirs(save_dir, exist_ok=True)

    sample_id = datetime.now().strftime("FP_%Y%m%d_%H%M%S_%f")[:-3]

    image_path = None
    dataset_label_path = os.path.join(labels_dir, sample_id + ".txt")

    source = "none"
    jpeg = None
    image_ext = ".jpg"
    uploaded_image = request.files.get("image") or request.files.get("file")

    if uploaded_image and uploaded_image.filename:
        raw_name = secure_filename(uploaded_image.filename)
        ext = os.path.splitext(raw_name)[1].lower()
        if ext not in IMAGE_EXTS:
            return jsonify({
                "status": "error",
                "message": "Selected false-positive source must be an image file.",
            }), 400
        jpeg = uploaded_image.read()
        if not jpeg:
            return jsonify({
                "status": "error",
                "message": "Selected image file is empty.",
            }), 400
        image_ext = ext
        source = "uploaded_file:" + raw_name
    elif node:
        with node.frame_lock:
            if node.latest_raw_jpeg:
                jpeg = node.latest_raw_jpeg
                source = node.raw_image_topic
            elif node.latest_jpeg:
                jpeg = node.latest_jpeg
                source = node.image_topic

    dataset_image_path = os.path.join(images_dir, sample_id + image_ext)

    if jpeg:
        image_path = os.path.join(save_dir, sample_id + image_ext)
        with open(image_path, "wb") as f:
            f.write(jpeg)
        with open(dataset_image_path, "wb") as f:
            f.write(jpeg)
    else:
        return jsonify({
            "status": "error",
            "message": "No current camera frame is available. Select a false-positive image file or wait for camera frames.",
        }), 409

    label_text = str(data.get("label_text") or "").strip()
    mode_text = str(data.get("mode") or "")
    if "空标签" in mode_text or "负样本" in mode_text or "empty" in mode_text.lower() or "negative" in mode_text.lower() or not label_text:
        label_text = ""
    with open(dataset_label_path, "w", encoding="utf-8") as f:
        f.write(label_text + ("\n" if label_text else ""))

    meta_path = os.path.join(save_dir, "metadata.jsonl")

    meta = {
        "sample_id": sample_id,
        "time": datetime.now().isoformat(),
        "image_path": image_path,
        "dataset": safe_dataset_name(dataset_name),
        "dataset_path": dataset_path,
        "split": split,
        "dataset_image_path": dataset_image_path,
        "dataset_label_path": dataset_label_path,
        "image_source": source,
        "request": data,
    }

    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    if node:
        node.add_log("dataset", f"误检样本入库: {sample_id}, dataset={dataset_name}, split={split}")

    return jsonify({
        "status": "ok",
        "sample_id": sample_id,
        "image_path": image_path,
        "dataset": dataset_summary(dataset_path),
        "dataset_image_path": dataset_image_path,
        "dataset_label_path": dataset_label_path,
        "image_source": source,
    })


@app.route("/api/dataset/report")
def api_dataset_report():
    datasets = list_datasets()
    fp = false_positive_summary(limit=20)
    models = list_models()
    current = current_model_summary()
    totals = dataset_totals()
    report = {
        "generated_at": iso_now(),
        "workspace": BASE_DIR,
        "datasets": datasets,
        "totals": totals,
        "false_positive": fp,
        "models": {
            "count": len(models),
            "current": current,
        },
    }
    lines = [
        f"数据集统计报表（{report['generated_at']}）",
        f"数据集数量：{totals['count']}",
        f"图像总量：{totals['images']}，训练：{totals['train_images']}，验证：{totals['val_images']}，测试：{totals['test_images']}",
        f"标签文件：{totals['labels']}",
        f"误检样本：{fp['count']}，今日新增：{fp['today']}",
        f"模型数量：{len(models)}，当前模型：{current.get('name') or '未应用'}",
    ]
    for item in datasets:
        lines.append(
            f"- {item['name']}: images={item['images']}, labels={item['labels']}, "
            f"best.pt={'yes' if item['has_best_pt'] else 'no'}, yolo26n.pt={'yes' if item['has_yolo26n'] else 'no'}"
        )
    report["text"] = "\n".join(lines)
    return jsonify({
        "status": "ok",
        "report": report,
    })


@app.route("/api/dataset/false_positive/analyze", methods=["POST"])
def api_false_positive_analyze():
    data = request.get_json(silent=True) or {}
    detection = parse_detection_summary(node.latest_detection_text if node else "")
    context = {
        "device": "RDK X5",
        "workspace": BASE_DIR,
        "selected_dataset": data.get("dataset") or data.get("dataset_name") or "",
        "split": normalize_dataset_split(data.get("split", "train")),
        "current_prediction": data.get("pred", ""),
        "reason": data.get("reason", ""),
        "mode": data.get("mode", ""),
        "detection": detection,
        "false_positive_summary": false_positive_summary(limit=8),
        "dataset_totals": dataset_totals(),
        "current_model": current_model_summary(),
    }
    prompt = (
        "请分析当前电缆缺陷检测误检样本，给出："
        "1. 可能误检原因；2. 是否建议作为 hard negative；"
        "3. 推荐加入的数据集分区；4. 下一轮训练建议。"
        "回答必须简洁，适合展示在网页右侧分析卡片。"
    )
    reply = call_direct_llm(prompt, context)
    if (
        "timed out" in reply
        or "连接模型接口失败" in reply
        or "模型接口 HTTP 错误" in reply
        or "Internal server error" in reply
    ):
        reason = data.get("reason") or "未填写误检原因"
        dataset = data.get("dataset") or data.get("dataset_name") or default_dataset_name() or "未选择数据集"
        reply = (
            f"本地闭环分析：当前样本可作为 hard negative 负样本入库。"
            f"原因记录为：{reason}。建议写入 {dataset}/{normalize_dataset_split(data.get('split', 'train'))}，"
            "生成空标签，下一轮训练时重点覆盖反光、金属边缘和背景纹理场景。"
        )
    if node:
        node.add_log("assistant", "AI 分析误检样本")
    return jsonify({
        "status": "ok",
        "analysis": reply,
        "context": context,
    })


@app.route("/api/pipeline/train", methods=["POST"])
def api_train():
    """一键训练接口，占位"""
    if node:
        node.add_log("pipeline", "收到一键训练请求")

    return jsonify({
        "status": "accepted",
        "message": "训练任务已接收。当前为占位接口，后续可接入训练脚本。"
    })


@app.route("/api/pipeline/deploy", methods=["POST"])
def api_deploy():
    """一键部署接口，占位"""
    if node:
        node.add_log("pipeline", "收到一键部署请求")

    return jsonify({
        "status": "accepted",
        "message": "部署任务已接收。当前为占位接口，后续可接入部署脚本。"
    })


@app.route("/api/claw/command", methods=["POST"])
def api_claw_command():
    """
    Right-side AI Agent endpoint.
    It collects live board context first, executes only whitelisted local actions,
    then asks the configured LLM. If the LLM is unavailable, it still returns a
    useful local engineering answer.
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "status": "error",
            "reply": "没有收到指令内容。",
            "action_result": ""
        })

    client_context = {
        "page": data.get("page", ""),
        "history": data.get("history", [])[-8:] if isinstance(data.get("history"), list) else [],
        "client_time": data.get("client_time", ""),
    }
    context = assistant_collect_context(client_context)
    intent = classify_assistant_intent(text)
    action_result = execute_assistant_intent(intent, context)
    context["inspection"] = inspection_snapshot()
    context["intent"] = intent
    context["local_action_result"] = action_result

    local_reply = assistant_local_reply(text, context, intent, action_result)
    if assistant_use_fast_skill(intent):
        reply = local_reply
        mode = "local_skill"
    else:
        llm_reply = call_direct_llm(text, context)
        if assistant_llm_failed(llm_reply):
            reply = local_reply
            mode = "local_fallback"
        else:
            reply = llm_reply
            mode = "direct_llm"
            if action_result and action_result not in reply:
                reply = action_result + "\n" + reply

    reply = clean_assistant_reply(reply) or local_reply

    if node:
        node.add_log("assistant", f"{intent.get('name')}: {text[:60]}")

    return jsonify({
        "status": "ok",
        "reply": reply,
        "action_result": action_result,
        "intent": intent.get("name", "general"),
        "skill": intent.get("skill", "general_chat"),
        "skills": assistant_skill_catalog(),
        "suggestions": assistant_suggestions(intent, context),
        "inspection": inspection_snapshot(),
        "mode": mode
    })


@app.route("/api/datasets")
def api_datasets():
    return jsonify({
        "status": "ok",
        "datasets": list_datasets()
    })


@app.route("/api/datasets/<dataset_name>/samples")
def api_dataset_samples(dataset_name):
    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 24))
    except (TypeError, ValueError):
        page, page_size = 1, 24
    result, error = list_dataset_samples(
        dataset_name,
        split_filter=request.args.get("split", "all"),
        annotation_filter=request.args.get("annotation", "all"),
        query=request.args.get("q", ""),
        page=page,
        page_size=page_size,
    )
    if error:
        return jsonify({"status": "error", "message": error}), 404
    return jsonify({"status": "ok", **result})


@app.route("/api/datasets/<dataset_name>/preview/<split>/<path:filename>")
def api_dataset_preview(dataset_name, split, filename):
    dataset_path, error = validate_dataset_path(dataset_name)
    split = str(split or "").strip().lower()
    if error or split not in ("train", "valid", "test"):
        return jsonify({
            "status": "error",
            "message": error or "Invalid dataset split.",
        }), 404

    filename = os.path.basename(filename)
    image_path = safe_join_under(dataset_path, [split, "images", filename])
    if (
        not image_path
        or os.path.splitext(image_path)[1].lower() not in IMAGE_EXTS
        or not os.path.isfile(image_path)
    ):
        return jsonify({
            "status": "error",
            "message": "Dataset image not found.",
        }), 404

    image = cv2.imread(image_path)
    if image is None:
        return jsonify({
            "status": "error",
            "message": "Dataset image could not be decoded.",
        }), 415

    stem = os.path.splitext(filename)[0]
    label_path = os.path.join(dataset_path, split, "labels", stem + ".txt")
    class_names = read_dataset_classes(dataset_path)
    annotations = read_yolo_annotations(label_path, class_names)
    height, width = image.shape[:2]
    colors = (
        (32, 201, 122),
        (255, 136, 22),
        (95, 103, 255),
        (84, 180, 255),
        (77, 77, 255),
    )

    for item in annotations:
        x_center = item["x_center"] * width
        y_center = item["y_center"] * height
        box_width = item["width"] * width
        box_height = item["height"] * height
        x1 = max(0, int(round(x_center - box_width / 2)))
        y1 = max(0, int(round(y_center - box_height / 2)))
        x2 = min(width - 1, int(round(x_center + box_width / 2)))
        y2 = min(height - 1, int(round(y_center + box_height / 2)))
        color = colors[item["class_id"] % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label = item["class_name"]
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            2,
        )
        label_top = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(
            image,
            (x1, label_top),
            (min(width - 1, x1 + text_width + 12), y1),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 6, max(text_height + 2, y1 - baseline - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (4, 15, 25),
            2,
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({
            "status": "error",
            "message": "Dataset preview could not be encoded.",
        }), 500
    return send_file(
        io.BytesIO(encoded.tobytes()),
        mimetype="image/jpeg",
        download_name=stem + "_annotated.jpg",
        max_age=30,
    )


@app.route("/api/datasets/<dataset_name>", methods=["DELETE"])
def api_dataset_delete(dataset_name):
    dataset_path, error = validate_dataset_path(dataset_name)
    if error:
        return jsonify({"status": "error", "message": error}), 404

    safe_name = safe_dataset_name(dataset_name)
    with pipeline_lock:
        busy = bool(pipeline_state.get("running"))
        busy_dataset = safe_dataset_name(pipeline_state.get("dataset", ""))
    if busy and busy_dataset == safe_name:
        return jsonify({
            "status": "error",
            "message": "当前数据集正在训练/转换任务中，任务结束后再删除。",
        }), 409

    try:
        shutil.rmtree(dataset_path)
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": f"删除数据集失败: {exc}",
        }), 500

    if node:
        node.add_log("dataset", f"dataset deleted: {safe_name}")
    return jsonify({
        "status": "ok",
        "deleted": safe_name,
    })


@app.route("/api/datasets/upload", methods=["POST"])
def api_dataset_upload():
    dataset_name = safe_dataset_name(request.form.get("name", ""))
    files = request.files.getlist("files")

    if not files:
        return jsonify({
            "status": "error",
            "message": "No dataset files were uploaded."
        }), 400

    target = os.path.join(DATASETS_DIR, dataset_name)
    if os.path.exists(target):
        dataset_name = dataset_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        target = os.path.join(DATASETS_DIR, dataset_name)

    os.makedirs(target, exist_ok=True)
    saved = 0
    errors = []

    for item in files:
        raw_name = item.filename or ""
        parts = upload_destination_parts(clean_upload_parts(raw_name, dataset_name), dataset_name)
        if not parts:
            continue
        dest = safe_join_under(target, parts)
        if not dest:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.splitext(raw_name)[1].lower() in ARCHIVE_EXTS:
            zip_path = dest
            item.save(zip_path)
            saved += 1
            try:
                extract_uploaded_archive(zip_path, target, dataset_name)
            except Exception as exc:
                errors.append(f"{raw_name}: {exc}")
                if node:
                    node.add_log("dataset", f"zip extract failed: {raw_name}, {exc}")
            finally:
                try:
                    os.remove(zip_path)
                except OSError:
                    pass
            continue
        item.save(dest)
        saved += 1

    normalize_uploaded_yolo_dataset(target)
    summary = dataset_summary(target)
    if int(summary.get("images") or 0) <= 0 and errors:
        shutil.rmtree(target, ignore_errors=True)
        return jsonify({
            "status": "error",
            "message": "；".join(errors),
        }), 400
    if node:
        node.add_log("dataset", f"dataset uploaded: {dataset_name}, files={saved}")

    return jsonify({
        "status": "ok",
        "dataset": summary,
        "saved_files": saved,
        "warnings": errors,
    })


@app.route("/api/models")
def api_models():
    return jsonify({
        "status": "ok",
        "models": list_models()
    })


@app.route("/api/models/upload_pt", methods=["POST"])
def api_models_upload_pt():
    item = request.files.get("pt") or request.files.get("file")
    if not item or not item.filename:
        return jsonify({"status": "error", "message": "No .pt model file uploaded."}), 400
    raw_name = secure_filename(item.filename)
    if not raw_name.lower().endswith(".pt"):
        return jsonify({"status": "error", "message": "Only .pt files are accepted."}), 400

    target_dir = os.path.join(MODELS_DIR, "pt_uploads")
    os.makedirs(target_dir, exist_ok=True)
    stem, ext = os.path.splitext(raw_name)
    model_id = stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join(target_dir, model_id + ext)
    item.save(target)

    metadata = {
        "model_id": model_id,
        "model_path": target,
        "uploaded_at": iso_now(),
        "training_mode": "uploaded_pt",
        "dataset": request.form.get("dataset", ""),
        "defect_note": request.form.get("defect_note", ""),
        "template": "yolo26n",
        "distillation": request.form.get("distillation", "dinov3"),
    }
    meta = write_model_metadata(target, metadata)
    append_model_history(
        "pt_uploaded",
        target,
        meta,
        "Uploaded PT model; it can be converted to ONNX/BIN through the pipeline.",
    )
    append_loop_audit("pt_uploaded", {"model_path": target, "defect_note": metadata["defect_note"]})
    if node:
        node.add_log("pipeline", f"PT uploaded: {target}")
    return jsonify({"status": "ok", "model_path": target, "metadata": meta})


@app.route("/api/models/upload_bin", methods=["POST"])
def api_models_upload_bin():
    item = request.files.get("model") or request.files.get("bin") or request.files.get("file")
    if not item or not item.filename:
        return jsonify({"status": "error", "message": "No .bin model file uploaded."}), 400
    raw_name = secure_filename(item.filename)
    if not raw_name.lower().endswith(".bin"):
        return jsonify({"status": "error", "message": "Only .bin files are accepted."}), 400

    target_dir = os.path.join(MODELS_DIR, "uploads")
    os.makedirs(target_dir, exist_ok=True)
    stem, ext = os.path.splitext(raw_name)
    model_id = stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join(target_dir, model_id + ext)
    item.save(target)

    metadata = {
        "model_id": model_id,
        "model_name": os.path.basename(target),
        "model_path": target,
        "uploaded_at": iso_now(),
        "training_mode": "manual_bin_upload",
        "dataset": request.form.get("dataset", ""),
        "source_model_path": request.form.get("source_model_path", ""),
    }
    metadata = reconcile_model_metadata(target, metadata)
    meta = write_model_metadata(target, metadata)
    append_model_history(
        "trained_upload",
        target,
        meta,
        "Manual BPU BIN model uploaded from web UI.",
    )
    if node:
        node.add_log("pipeline", f"BIN uploaded: {target}")
    return jsonify({"status": "ok", "model_path": target, "metadata": meta})


@app.route("/api/models/delete", methods=["POST"])
def api_models_delete():
    data = request.get_json(silent=True) or {}
    model_path, error = validate_model_path(select_model_path(data))
    if error:
        return jsonify({"status": "error", "message": error}), 400

    abs_path = os.path.abspath(model_path)
    protected = {
        os.path.abspath(APPLIED_MODEL_BIN),
        os.path.abspath(load_json_file(CURRENT_MODEL_JSON, {}).get("applied_model_path", "") or APPLIED_MODEL_BIN),
    }
    if abs_path in protected:
        return jsonify({
            "status": "error",
            "message": "当前正在应用的模型不能删除；请先部署其它模型。",
        }), 400

    meta = read_model_metadata(model_path)
    deleted = []
    for path in (model_path, model_metadata_path(model_path)):
        if path and os.path.exists(path):
            os.remove(path)
            deleted.append(path)

    append_model_history(
        "model_deleted",
        model_path,
        meta,
        "Model file deleted from web UI.",
    )
    if node:
        node.add_log("pipeline", f"model deleted: {model_path}")
    return jsonify({"status": "ok", "deleted": deleted, "model_path": model_path})


@app.route("/api/models/register", methods=["POST"])
def api_models_register():
    data = request.get_json(silent=True) or {}
    model_path, error = validate_model_path(data.get("model_path", ""))
    if error:
        return jsonify({
            "status": "error",
            "message": error
        }), 400

    metadata = data.get("metadata") or {}
    metadata.update({
        "board_model_path": model_path,
        "registered_at": iso_now(),
    })
    metadata = reconcile_model_metadata(model_path, metadata)
    meta = write_model_metadata(model_path, metadata)
    append_model_history(
        "trained_upload",
        model_path,
        meta,
        "Model trained or converted locally and uploaded to RDK X5.",
    )

    if node:
        node.add_log("pipeline", f"model registered: {model_path}")

    return jsonify({
        "status": "ok",
        "model": meta
    })


@app.route("/api/model_history")
def api_model_history():
    try:
        limit = int(request.args.get("limit", "100"))
    except ValueError:
        limit = 100
    return jsonify({
        "status": "ok",
        "history": list_model_history(limit=max(1, min(limit, 500)))
    })


@app.route("/api/pipeline/status")
def api_pipeline_status():
    with pipeline_lock:
        state = dict(pipeline_state)
    state["log_tail"] = read_tail(state.get("log_path", ""))
    worker = local_yolo26_worker_status(timeout=0.5)
    if worker and isinstance(worker.get("pipeline"), dict):
        worker_state = dict(worker["pipeline"])
        if worker_state.get("status") not in ("", "idle", None) or worker_state.get("running"):
            state.update(worker_state)
            state["local_worker_url"] = worker.get("worker_url", "")
            state["execution_host"] = "windows_local_worker"
    return jsonify({
        "status": "ok",
        "pipeline": state,
    })


@app.route("/api/pipeline/preflight", methods=["GET", "POST"])
def api_pipeline_preflight():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
    else:
        data = request.args
    dataset_name = data.get("dataset") or data.get("dataset_name") or default_dataset_name()
    weights = str(data.get("weights") or data.get("pt_model_path") or "").strip()
    skip_train = bool(data.get("skip_train") or data.get("direct_pt"))
    distill_teacher = str(data.get("distill_teacher") or data.get("teacher") or "").strip()
    preflight = pipeline_preflight(dataset_name, weights=weights, skip_train=skip_train, distill_teacher=distill_teacher)
    worker = local_yolo26_worker_status(timeout=0.5)
    if worker:
        preflight["ok"] = True
        preflight["missing"] = []
        preflight["actions"] = []
        preflight["local_worker"] = {
            "ok": True,
            "url": worker.get("worker_url", ""),
            "status": (worker.get("pipeline") or {}).get("status", "idle"),
        }
        for stage in preflight.get("stages", []):
            if stage.get("stage") in ("export_onnx", "quantize_bin"):
                stage["ok"] = True
                stage["detail"] = "Windows local YOLO26 worker"
    return jsonify({
        "status": "ok",
        "preflight": preflight,
    })


def api_train_real():
    data = request.get_json(silent=True) or {}
    dataset_name = data.get("dataset") or data.get("dataset_name") or ""
    epochs = int(data.get("epochs", 80))
    imgsz = int(data.get("imgsz", 640))
    batch = int(data.get("batch", 8))
    weights = str(data.get("weights") or data.get("pt_model_path") or "").strip()
    skip_train = bool(data.get("skip_train") or data.get("direct_pt"))
    distill_raw = data.get("distill", True)
    distill_teacher = str(data.get("distill_teacher") or data.get("teacher") or "").strip()
    if isinstance(distill_raw, str):
        lower_distill = distill_raw.lower().strip()
        distill = lower_distill not in ("0", "false", "no", "off")
        if distill and lower_distill not in ("1", "true", "yes", "on", "dinov3"):
            distill_teacher = distill_raw
    else:
        distill = bool(distill_raw)
    defect_note = str(data.get("defect_note") or "").strip()

    datasets = list_datasets()
    if not dataset_name and datasets:
        dataset_name = datasets[0]["name"]

    dataset_path = os.path.join(DATASETS_DIR, dataset_name)
    if not dataset_name or not os.path.isdir(dataset_path):
        return jsonify({
            "status": "error",
            "message": "No valid dataset selected."
        }), 400
    if weights:
        abs_weights = os.path.abspath(weights)
        models_root = os.path.abspath(MODELS_DIR)
        try:
            inside_models = os.path.commonpath([models_root, abs_weights]) == models_root
        except ValueError:
            inside_models = False
        if not inside_models or not os.path.exists(abs_weights) or not abs_weights.endswith(".pt"):
            return jsonify({
                "status": "error",
                "message": "Uploaded PT must exist inside the models directory.",
            }), 400
        weights = abs_weights

    worker_payload = {
        "dataset": dataset_name,
        "dataset_name": dataset_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "weights": weights,
        "pt_model_path": weights,
        "skip_train": skip_train,
        "direct_pt": skip_train,
        "distill": distill,
        "distill_teacher": distill_teacher,
        "defect_note": defect_note,
    }
    try:
        worker_response, worker_code = local_yolo26_worker_train(worker_payload)
        if worker_response:
            with pipeline_lock:
                pipeline_state.update({
                    "running": True,
                    "stage": "local_worker",
                    "status": "running" if worker_code < 400 else "failed",
                    "message": worker_response.get("message", "Local YOLO26 worker started."),
                    "dataset": dataset_name,
                    "model": "",
                    "training_mode": "windows_local_pt_onnx_openexplore_bin",
                    "log_path": "",
                    "started_at": datetime.now().isoformat(),
                    "finished_at": None,
                    "local_worker_url": worker_response.get("worker_url", ""),
                })
            return jsonify(worker_response), worker_code
    except Exception as exc:
        if node:
            node.add_log("pipeline", f"local worker unavailable: {exc}")

    preflight = pipeline_preflight(dataset_name, weights=weights, skip_train=skip_train, distill_teacher=distill_teacher)
    if not preflight.get("ok"):
        return jsonify({
            "status": "error",
            "message": "Training pipeline preflight failed: " + "; ".join(preflight.get("missing", [])),
            "preflight": preflight,
        }), 400

    with pipeline_lock:
        if pipeline_state.get("running"):
            return jsonify({
                "status": "busy",
                "pipeline": dict(pipeline_state)
            }), 409

    if node:
        node.add_log("pipeline", f"start training pipeline: {dataset_name}")

    thread = threading.Thread(
        target=pipeline_worker,
        args=(dataset_name, epochs, imgsz, batch, weights, skip_train, distill, defect_note, distill_teacher),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "status": "accepted",
        "message": "Training pipeline started.",
        "dataset": dataset_name
    })


def api_deploy_real():
    return api_apply_model()


@app.route("/api/pipeline/apply_model", methods=["POST"])
def api_apply_model():
    data = request.get_json(silent=True) or {}
    model_path, error = validate_model_path(select_model_path(data))
    if error:
        return jsonify({
            "status": "error",
            "message": error
        }), 400

    staged_path, meta = stage_model_file(
        model_path,
        "Applied deployment. Detection node is restarted with the selected model.",
    )

    os.makedirs(DEPLOY_DIR, exist_ok=True)
    tmp_current = APPLIED_MODEL_BIN + ".tmp"
    shutil.copy2(staged_path, tmp_current)
    os.replace(tmp_current, APPLIED_MODEL_BIN)
    applied_at = iso_now()
    applied_meta = copy_model_metadata(staged_path, APPLIED_MODEL_BIN, {
        "applied_model_path": APPLIED_MODEL_BIN,
        "applied_at": applied_at,
    })

    stopped_pids = stop_detection_node()
    started_pids = start_detection_node(APPLIED_MODEL_BIN)

    applied_meta.update({
        "applied_model_path": APPLIED_MODEL_BIN,
        "applied_at": applied_at,
        "stopped_detection_pids": stopped_pids,
        "started_detection_pids": started_pids,
    })
    write_model_metadata(APPLIED_MODEL_BIN, applied_meta)
    write_current_model_meta(applied_meta)
    append_model_history(
        "model_applied",
        APPLIED_MODEL_BIN,
        applied_meta,
        "Model applied to detection_node.",
    )

    if node:
        node.add_log(
            "pipeline",
            f"model applied: {APPLIED_MODEL_BIN}, detection_pids={started_pids}",
        )

    if not started_pids:
        return jsonify({
            "status": "error",
            "message": f"Model copied, but detection_node did not start. Check {DETECTION_LOG}.",
            "model_path": APPLIED_MODEL_BIN,
            "log_path": DETECTION_LOG,
        }), 500

    return jsonify({
        "status": "ok",
        "message": "Model applied. Only detection_node was restarted; camera and motor nodes were not restarted.",
        "model_path": APPLIED_MODEL_BIN,
        "staged_model_path": staged_path,
        "stopped_detection_pids": stopped_pids,
        "started_detection_pids": started_pids,
        "log_path": DETECTION_LOG,
    })


app.view_functions["api_train"] = api_train_real
app.view_functions["api_deploy"] = api_deploy_real


def run_flask():
    """运行 Flask 服务器"""
    app.run(
        host="0.0.0.0",
        port=8010,
        threaded=True,
        debug=False,
        use_reloader=False
    )


def main(args=None):
    global node

    rclpy.init(args=args)
    node = WebControlNode()

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
