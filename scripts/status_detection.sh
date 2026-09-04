#!/bin/bash
# 检查电缆缺陷检测系统状态

echo "📊 电缆缺陷检测系统状态"
echo "================================"

# 检查各个节点是否运行
check_process() {
    local name=$1
    local pattern=$2
    if pgrep -f "$pattern" > /dev/null; then
        echo "✅ $name: 运行中"
        return 0
    else
        echo "❌ $name: 未运行"
        return 1
    fi
}

check_process "摄像头节点" "camera_node"
check_process "检测节点" "detection_node"
check_process "电机控制节点" "motor_control_node"
check_process "Web控制节点" "web_control_node"

echo ""
echo "================================"

# 检查Web服务
if curl -s http://localhost:8010 > /dev/null 2>&1; then
    echo "✅ Web界面: http://192.168.128.10:8010"
else
    echo "❌ Web界面: 无法访问"
fi

echo ""

# 显示最近的日志
if [ -f /tmp/cable1_detection.log ]; then
    echo "最近日志 (最后10行):"
    echo "--------------------------------"
    tail -10 /tmp/cable1_detection.log
fi
