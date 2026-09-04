# Web界面使用说明

## 新功能

现在Web界面已经升级，包含以下功能：

### 1. 实时视频流
- 显示摄像头实时画面
- 自动显示检测框和标注
- 流畅的30fps视频流

### 2. 电机控制
- **正转按钮** (⬆️): 电机正向旋转
- **反转按钮** (⬇️): 电机反向旋转  
- **停止按钮** (⏹️): 停止电机

### 3. 状态显示
- **电机状态**: 显示运行中/已停止
- **运行方向**: 显示正转/反转

## 使用步骤

### 1. 在RDK X5板子上重新编译并启动

```bash
cd /path/to/cable_defect_detection
./rebuild_and_start.sh
```

或者手动操作：

```bash
# Source环境
source /opt/ros/humble/setup.bash

# 重新编译
colcon build --packages-select motor_control_pkg

# Source工作空间
source install/setup.bash

# 启动系统
ros2 launch launch/cable_detection.launch.py
```

### 2. 打开Web界面

在浏览器中访问：
```
http://192.168.128.10:5000
```

支持的浏览器：
- Chrome / Edge (推荐)
- Firefox
- Safari
- 手机浏览器

### 3. 控制电机

- 点击 **⬆️ 正转** 按钮：电机开始正向旋转
- 点击 **⬇️ 反转** 按钮：电机开始反向旋转
- 点击 **⏹️ 停止** 按钮：停止电机

**注意**：
- 切换方向时，电机会自动停止并重新启动
- 当检测到缺陷在中央时，电机会自动停止
- 需要手动点击按钮重新启动

## 界面说明

### 状态栏
```
┌─────────────────────────────────────┐
│  电机状态: 运行中  |  运行方向: 正转  │
└─────────────────────────────────────┘
```

- **电机状态**
  - 🟢 运行中 (绿色)
  - 🔴 已停止 (红色)

- **运行方向**
  - 🔵 正转 (蓝色)
  - 🟠 反转 (橙色)

### 视频区域
显示实时摄像头画面，包含：
- 检测框（绿色=burn，红色=pr）
- 置信度标签
- 缺陷中心点
- 图像中心十字线

### 控制按钮
```
┌──────────┬──────────┐
│ ⬆️ 正转   │ ⬇️ 反转   │
├──────────┴──────────┤
│      ⏹️ 停止        │
└─────────────────────┘
```

## 故障排查

### 问题1: 看不到视频流

**检查摄像头节点**：
```bash
ros2 topic list | grep camera
ros2 topic hz /detection/annotated_image
```

**检查日志**：
```bash
ros2 node info /web_control_node
```

### 问题2: 按钮点击无反应

**检查话题**：
```bash
ros2 topic echo /motor/control
```

**检查电机控制节点**：
```bash
ros2 node list | grep motor_control
```

### 问题3: 无法访问Web界面

**检查端口**：
```bash
netstat -tuln | grep 5000
```

**检查防火墙**：
```bash
sudo ufw allow 5000
```

**检查网络连接**：
```bash
ping 192.168.128.10
```

## 技术细节

### 视频流
- 格式: MJPEG (Motion JPEG)
- 质量: 90%
- 帧率: ~30fps
- 延迟: <100ms

### 通信协议
- HTTP REST API
- ROS2 话题通信
- 实时状态更新 (500ms间隔)

### ROS2话题

**订阅**：
- `/detection/annotated_image` - 带标注的图像
- `/motor/status` - 电机状态
- `/motor/direction` - 电机方向

**发布**：
- `/motor/control` - 控制命令 (forward/reverse/stop)

## 移动端优化

界面已针对移动设备优化：
- 响应式布局
- 触摸友好的大按钮
- 自适应视频尺寸

在手机上访问同样的地址即可使用。
