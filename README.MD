# Rixing Bawanli: RDK X5 Cable Defect Inspection System

`cable1` is a ROS2-based cable inspection project for the Horizon RDK X5 board. It combines visible-light camera capture, DINOv3 distillation training, YOLO26 coarse localization on the Horizon BPU, OpenCV fine localization, infrared hotspot diagnostics, stepper motor control, a browser-based operations console, AD5933 impedance analysis, dataset/model lifecycle tools, and an OpenClaw-compatible assistant endpoint into one edge-AI inspection workflow.

The project is designed for cable surface defect inspection, RDK X5 edge inference demos, teaching, and multimodal inspection experiments that combine vision, infrared, and impedance signals.

## Core Features

- Real-time camera capture with ROS2 image publishing and MJPEG streaming.
- Cascaded visual detection: DINOv3 acts as a training-time teacher for YOLO26/YOLO26n distillation; the deployed model runs YOLO26 coarse localization on the RDK X5 BPU; OpenCV ROI analysis then performs fine localization.
- Cable-band extraction filters background clutter, and OpenCV blackhat, gradient, and local-contrast features refine the YOLO coarse boxes.
- Infrared camera ingestion with pseudo-color display and relative hotspot diagnostics.
- GPIO stepper motor control with continuous movement, distance moves, move-to-position, speed percentage, button input, and software travel limits.
- Web console for video streams, motor control, runtime health, datasets, review samples, model upload/deployment, and inspection reports.
- AD5933 impedance analysis for low-resistance, open-circuit, moisture, and cable-location assisted checks.
- Dataset upload, false-positive capture, review sample pools, YOLO training, ONNX export, and RDK X5 BPU BIN conversion.
- OpenClaw-compatible assistant endpoint for operator support and inspection troubleshooting. The current backend keeps `/api/claw/command`, reads OpenClaw-style configuration, and calls the model API directly instead of routing through OpenClaw Gateway.

## Hardware And Runtime

- Board: Horizon RDK X5
- OS/runtime: Ubuntu with ROS2 Humble
- Inference: `hobot_dnn` / Horizon BPU
- Main camera: publishes to `/camera/image_raw`; device can be set with `/dev/v4l/by-id/...`
- Infrared camera: publishes to `/infrared/image_raw`
- Motor GPIO: Hobot GPIO in BOARD mode, defaults `DIR=31`, `STEP=32`, `EN=36`, `BUTTON=37`
- Web service: `http://<board-ip>:8010`

## Project Layout

```text
cable1/
├── src/
│   ├── camera_pkg/          # Camera capture node
│   ├── detection_pkg/       # YOLO26/BPU coarse localization, OpenCV refinement, infrared detection
│   ├── image_saver_pkg/     # Periodic and defect-triggered image saver
│   └── motor_control_pkg/   # Motor node and Flask web backend
├── ad5933_cable_analyzer/   # AD5933 impedance and cable diagnosis module
├── config/                  # Current model, motor position, and runtime params
├── scripts/                 # Training-to-BIN pipeline and status scripts
├── tools/                   # DINOv3 distillation, YOLO26 training/export, and mapper tools
├── third_party/tessdata/    # Third-party OCR/vision data
├── agentic_aiops_clean_sidebar.html  # Web frontend
├── start_system.sh          # Board-side startup script
└── WEB_GUIDE.md             # Web console guide
```

> Use `start_system.sh` for normal board startup, or run the ROS2 console scripts individually when debugging.

## Quick Start

On the RDK X5 board:

```bash
cd /home/sunrise/cable1
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
./start_system.sh
```

Then open the web console from a browser on the same network:

```text
http://<board-ip>:8010
```

The startup script launches the main camera supervisor, optional infrared camera, infrared detector, BPU defect detector, motor controller, optional image saver, and web control node. Logs are written under `/tmp`, including `/tmp/camera.log`, `/tmp/infrared_detection.log`, `/tmp/detection.log`, `/tmp/motor.log`, and `/tmp/web.log`.

## Running Individual Nodes

For focused debugging:

```bash
source /opt/ros/humble/setup.bash
source /home/sunrise/cable1/install/setup.bash

ros2 run camera_pkg camera_node
ros2 run detection_pkg detection_node
ros2 run detection_pkg infrared_detection_node
ros2 run motor_control_pkg motor_control_node
ros2 run motor_control_pkg web_control_node
ros2 run image_saver_pkg image_saver_node
```

Important topics:

- `/camera/image_raw`: main camera image
- `/detection/results`: visual defect JSON
- `/detection/annotated_image`: annotated visual image
- `/infrared/results`: infrared diagnostic JSON
- `/infrared/annotated_image`: pseudo-color infrared image
- `/motor/control`: motor command topic
- `/motor/motion_status`: motor position, speed, limits, and movement state

## Models And Data

The deployed model is expected at:

```text
/home/sunrise/cable1/models/deployed/current.bin
```

`config/current_model.json` records the active model name, classes, source path, hash, and deployment metadata. Large generated assets are intentionally excluded from the repository. Put them on the board as needed:

- `models/`: uploaded, staged, and deployed models
- `datasets/`: YOLO datasets
- `runs/`: training and conversion outputs
- `inspection_reports/`: generated inspection reports
- `false_positive_samples/`, `review_sample_pool/`: false-positive and review samples

Training and conversion entry point:

```bash
python3 scripts/train_to_bin_pipeline.py \
  --dataset /home/sunrise/cable1/datasets/<dataset_name> \
  --output-dir /home/sunrise/cable1/runs/<run_name>
```

The pipeline trains or reuses a `.pt` model. When distillation is enabled, DINOv3 is used only as a training-time teacher; the saved `best.pt` remains a normal YOLO26/YOLO26n detector. The pipeline then exports a BPU-friendly ONNX model and converts it to an RDK X5 `.bin` model through `hb_mapper` or the OpenExplorer Docker toolchain. The board-side runtime chain is `YOLO26/BPU coarse localization -> OpenCV ROI fine localization -> result publishing/report fusion`.

## Web Console

`web_control_node` runs a Flask service on `0.0.0.0:8010`. Key surfaces include:

- `/video_feed`: visual inspection MJPEG stream
- `/infrared_feed`: infrared MJPEG stream
- `/forward`, `/reverse`, `/stop`, `/move_to`: motor control
- `/api/status`: system, ROS graph, BPU, CPU, memory, disk, and temperature status
- `/api/inspection/*`: inspection sessions and reports
- `/api/ad5933/*`: impedance analyzer actions and status
- `/api/datasets`, `/api/models`, `/api/pipeline/*`: dataset, model, and deployment lifecycle
- `/api/claw/command`: OpenClaw-compatible assistant endpoint that reads OpenClaw-style configuration and calls the model API directly

The frontend file is `agentic_aiops_clean_sidebar.html` in the repository root.

## Dependencies

Core board dependencies:

- ROS2 Humble: `rclpy`, `sensor_msgs`, `std_msgs`, `cv_bridge`
- Python: OpenCV, NumPy, Flask, Werkzeug
- Horizon runtime: `hobot_dnn`
- GPIO: `Hobot.GPIO`
- AD5933: `smbus2`
- Training/conversion: Ultralytics, PyTorch, ONNX Runtime, `hb_mapper` or the OpenExplorer Docker image

## Notes

This repository keeps `README_cn.md` for Chinese documentation and `README.MD` for English project indexing. If Windows reports a case-collision between `README.md` and `README.MD`, keep `README.MD` as the English README and use `README_cn.md` for the Chinese version.
