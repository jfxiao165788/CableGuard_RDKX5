#!/bin/bash
# 启动电缆缺陷检测系统的所有节点

echo "启动电缆缺陷检测系统..."

# Source ROS2环境
source /opt/ros/humble/setup.bash
source /home/sunrise/cable1/install/setup.bash
export CABLE_WORKSPACE_DIR=/home/sunrise/cable1

# 停止旧进程
if [ -f /tmp/cable1_camera_supervisor.pid ]; then
  kill "$(cat /tmp/cable1_camera_supervisor.pid)" 2>/dev/null || true
  rm -f /tmp/cable1_camera_supervisor.pid
fi
if [ -f /tmp/cable1_infrared_camera_supervisor.pid ]; then
  kill "$(cat /tmp/cable1_infrared_camera_supervisor.pid)" 2>/dev/null || true
  rm -f /tmp/cable1_infrared_camera_supervisor.pid
fi
pkill -f 'camera_node|detection_node|infrared_detection_node|motor_control_node|web_control_node|image_saver_node'
sleep 2
# Force-clean any node that ignored SIGTERM so cameras and Web ports are released.
pkill -9 -f 'camera_node|detection_node|infrared_detection_node|motor_control_node|web_control_node|image_saver_node' 2>/dev/null || true
sleep 1

# 启动所有节点
echo "启动摄像头节点..."
: > /tmp/camera.log
(
  while true; do
    echo "[$(date '+%F %T')] 启动主摄像头" >> /tmp/camera.log
    ros2 run camera_pkg camera_node --ros-args \
      -p camera_device:=/dev/v4l/by-id/usb-HJ_USB_2.0_Camera_HJ_USB_2.0_Camera_SN0001-video-index0 \
      -p width:=640 \
      -p height:=480 \
      -p fps:=15 \
      -p pixel_format:=MJPG \
      -p crop_top:=80 \
      -p reconnect_after_failures:=3 \
      -p watchdog_timeout:=8.0 >> /tmp/camera.log 2>&1
    code=$?
    echo "[$(date '+%F %T')] 摄像头节点退出(code=$code)，2秒后重启" >> /tmp/camera.log
    sleep 2
  done
) &
echo $! > /tmp/cable1_camera_supervisor.pid

sleep 2

if [ "${ENABLE_INFRARED:-1}" = "1" ]; then
  echo "启动红外摄像头与检测节点..."
  : > /tmp/infrared_camera.log
  (
    while true; do
      echo "[$(date '+%F %T')] 启动红外摄像头" >> /tmp/infrared_camera.log
      ros2 run camera_pkg camera_node --ros-args \
        -r __node:=infrared_camera_node \
        -p camera_device:=/dev/v4l/by-id/usb-HIKVISION_HikCamera_12345678-video-index0 \
        -p width:=640 \
        -p height:=360 \
        -p fps:=30 \
        -p pixel_format:=MJPG \
        -p image_topic:=/infrared/image_raw \
        -p compressed_topic:=/infrared/image_raw/compressed \
        -p frame_id:=infrared_frame \
        -p reconnect_after_failures:=3 \
        -p watchdog_timeout:=8.0 >> /tmp/infrared_camera.log 2>&1
      code=$?
      echo "[$(date '+%F %T')] 红外摄像头节点退出(code=$code)，2秒后重启" >> /tmp/infrared_camera.log
      sleep 2
    done
  ) &
  echo $! > /tmp/cable1_infrared_camera_supervisor.pid
  ros2 run detection_pkg infrared_detection_node --ros-args \
    -p intensity_threshold:=225 \
    -p temperature_threshold_c:=70.0 \
    -p fallback_to_intensity_alarm:=false \
    -p relative_heat_alarm:=false \
    -p relative_heat_confirmations:=2 \
    -p relative_heat_roi_top_fraction:=0.16 \
    -p relative_heat_roi_bottom_fraction:=0.90 \
    -p relative_heat_percentile:=98.0 \
    -p relative_heat_min_score:=205.0 \
    -p relative_heat_min_contrast:=30.0 \
    -p relative_heat_min_area:=80 \
    -p relative_heat_max_area_ratio:=0.18 \
    -p relative_heat_local_sigma:=19.0 \
    -p min_area:=200 \
    -p display_width:=640 \
    -p display_height:=360 \
    -p display_low_percentile:=3.0 \
    -p display_high_percentile:=97.0 \
    -p display_smoothing:=0.35 \
    -p sharpen_amount:=0.0 > /tmp/infrared_detection.log 2>&1 &
else
  echo "跳过红外摄像头: ENABLE_INFRARED=0"
fi

sleep 1

echo "启动检测节点..."
ros2 run detection_pkg detection_node --ros-args \
  -p model_path:=/home/sunrise/cable1/models/deployed/current.bin \
  -p class_names:=defect \
  -p conf_thresh:=0.7 \
  -p inference_interval:=3 > /tmp/detection.log 2>&1 &

sleep 2

echo "启动电机控制节点..."
ros2 run motor_control_pkg motor_control_node --ros-args \
  -p lead_mm_per_rev:=10.0 \
  -p steps_per_rev:=400 \
  -p travel_min_mm:=0.0 \
  -p travel_max_mm:=450.0 \
  -p auto_stop:=false \
  -p position_state_path:=/home/sunrise/cable1/config/motor_position.json \
  -p step_delay:=0.0003 > /tmp/motor.log 2>&1 &

sleep 1

if [ "${ENABLE_IMAGE_SAVER:-0}" = "1" ]; then
  echo "启动图像保存节点..."
  ros2 run image_saver_pkg image_saver_node > /tmp/image_saver.log 2>&1 &
else
  echo "跳过图像保存节点: ENABLE_IMAGE_SAVER=0"
fi

sleep 1

echo "启动Web控制节点..."
ros2 run motor_control_pkg web_control_node --ros-args \
  -p preview_width:=0 \
  -p preview_height:=0 \
  -p jpeg_quality:=95 \
  -p stream_fps:=30.0 \
  -p encode_raw_stream:=false \
  -p encode_infrared_raw_stream:=true > /tmp/web.log 2>&1 &

sleep 3

echo ""
echo "========================================="
echo "系统启动完成！"
echo "========================================="
echo ""
echo "Web界面: http://192.168.128.10:8010"
echo ""
echo "查看日志:"
echo "  摄像头: tail -f /tmp/camera.log"
echo "  红外摄像头: tail -f /tmp/infrared_camera.log"
echo "  红外检测: tail -f /tmp/infrared_detection.log"
echo "  检测: tail -f /tmp/detection.log"
echo "  电机: tail -f /tmp/motor.log"
echo "  Web: tail -f /tmp/web.log"
echo ""
