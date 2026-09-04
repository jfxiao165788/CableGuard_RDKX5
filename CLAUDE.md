# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ROS2-based cable defect detection system for RDK X5 board. The system uses a Horizon BPU model to detect cable defects in real-time, controls a stepper motor via GPIO, and automatically stops when a defect is detected in the center of the camera view.

## Hardware Requirements

- **Board**: Horizon RDK X5 (ARM64)
- **Camera**: /dev/video0 or /dev/video1
- **Motor**: Stepper motor connected via GPIO pins (BOARD mode: DIR=31, STEP=32, EN=36)
- **Model**: Horizon BPU model (.bin format) for YOLO-based defect detection

## System Architecture

### ROS2 Nodes

1. **camera_node** (camera_pkg)
   - Captures images from camera
   - Publishes to `/camera/image_raw` and `/camera/image_raw/compressed`

2. **detection_node** (detection_pkg)
   - Subscribes to camera images
   - Runs Horizon BPU model inference using `hobot_dnn`
   - Publishes detection results to `/detection/results`
   - Publishes annotated images to `/detection/annotated_image`
   - Publishes defect center status to `/detection/defect_in_center`
   - Publishes defect detected flag to `/detection/defect_detected`

3. **motor_control_node** (motor_control_pkg)
   - Controls stepper motor via GPIO using `Hobot.GPIO` library
   - Subscribes to `/detection/defect_in_center`
   - Automatically stops motor when defect is in center
   - Publishes motor status to `/motor/status`

4. **image_saver_node** (image_saver_pkg)
   - Saves images periodically (every 0.5s by default)
   - Saves immediately when defect is detected
   - Saves to `/home/sunrise/saved_images` by default

### Topic Communication Flow

```
camera_node → /camera/image_raw → detection_node
detection_node → /detection/annotated_image → image_saver_node
detection_node → /detection/defect_in_center → motor_control_node
detection_node → /detection/defect_detected → image_saver_node
```

## Building and Running

### On RDK X5 Board

1. **Source ROS2 environment**:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Build the workspace**:
   ```bash
   cd cable_defect_detection
   colcon build
   source install/setup.bash
   ```

3. **Copy model file to board**:
   ```bash
   # Place model at: /home/sunrise/yolo26cable_bayese_640x640_nv12.bin
   ```

4. **Launch all nodes**:
   ```bash
   ros2 launch launch/cable_detection.launch.py
   ```

5. **View camera stream on remote PC**:
   ```bash
   # On your PC (same network):
   ros2 run rqt_image_view rqt_image_view /detection/annotated_image
   ```

### Individual Node Testing

```bash
# Test camera only
ros2 run camera_pkg camera_node

# Test detection only (requires camera running)
ros2 run detection_pkg detection_node

# Test motor control only
ros2 run motor_control_pkg motor_control_node

# Test image saver only
ros2 run image_saver_pkg image_saver_node
```

## Key Configuration Parameters

Edit `launch/cable_detection.launch.py` to modify:

- **Model path**: `model_path` parameter in detection_node
- **Camera device**: `camera_id` (0 or 1)
- **Center threshold**: `center_threshold` (pixels from image center)
- **Motor speed**: `step_delay` (seconds between pulses)
- **Save interval**: `save_interval` (seconds)
- **GPIO pins**: `pin_dir`, `pin_step`, `pin_en`

## Dependencies

### Python packages required on RDK X5:
- `hobot_dnn` (Horizon BPU inference library)
- `Hobot.GPIO` (GPIO control library)
- `opencv-python`
- `numpy`

### ROS2 packages:
- `rclpy`
- `sensor_msgs`
- `std_msgs`
- `cv_bridge`

## Model Information

- **Format**: Horizon BPU binary (.bin)
- **Input**: 640x640 NV12 format
- **Classes**: ['burn', 'pr']
- **Architecture**: YOLO-based with strides [8, 16, 32]

The detection node handles NV12 conversion automatically from BGR images.

## Remote Viewing

To view the camera stream from your PC:

1. Ensure PC and RDK X5 are on same network
2. On PC, source ROS2 and set ROS_DOMAIN_ID if needed
3. Use `rqt_image_view` or `rviz2` to view topics:
   - `/camera/image_raw/compressed` - Raw camera feed
   - `/detection/annotated_image` - Annotated with detections

## Troubleshooting

- **Camera not opening**: Check `/dev/video*` permissions and device availability
- **Model loading fails**: Verify model path and BPU availability
- **GPIO errors**: Ensure running with proper permissions (may need sudo)
- **No images on PC**: Check network connectivity and ROS_DOMAIN_ID settings
