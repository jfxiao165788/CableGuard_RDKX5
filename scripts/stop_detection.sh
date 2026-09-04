#!/bin/bash
# 停止电缆缺陷检测系统

echo "🛑 停止电缆缺陷检测系统..."

# 停止所有相关进程
pkill -f 'cable_detection.launch.py'
pkill -f 'camera_node'
pkill -f 'detection_node'
pkill -f 'motor_control_node'
pkill -f 'web_control_node'
pkill -f 'image_saver_node'

sleep 2

# 检查是否还有残留进程
if pgrep -f "camera_node|detection_node|motor_control_node|web_control_node" > /dev/null; then
    echo "⚠️  强制终止残留进程..."
    pkill -9 -f 'camera_node|detection_node|motor_control_node|web_control_node|image_saver_node'
    sleep 1
fi

echo "✅ 系统已停止"
