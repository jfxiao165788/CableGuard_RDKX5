# 日行八万里：RDK X5 电缆缺陷智能巡检系统

`cable1` 是一个运行在地平线 RDK X5 上的 ROS2 电缆缺陷巡检项目。系统把可见光摄像头、DINOv3 蒸馏训练、YOLO26/BPU 粗定位、OpenCV 精定位、红外热异常检测、步进电机运动控制、Web 控制台、AD5933 阻抗分析、数据集管理、模型训练/转换/部署和 OpenClaw 兼容助手接口集成到同一套边缘 AI 工作流中。

项目适合用于电缆表面缺陷巡检、教学演示、RDK X5 边缘推理验证，以及视觉、红外、阻抗多模态融合检测实验。

## 核心能力

- 实时采集主摄像头画面，发布 ROS2 图像话题并通过 Web 端推流。
- 视觉检测采用级联式架构：训练阶段使用 DINOv3 作为 teacher 蒸馏 YOLO26/YOLO26n student；部署阶段在 RDK X5 BPU 上运行 YOLO26 做缺陷粗定位；运行时再用 OpenCV ROI 特征做精定位。
- 在视觉检测前提取电缆暗色带，过滤背景干扰，并对 YOLO 粗框做 blackhat、梯度、局部对比度等 OpenCV 精修。
- 支持红外摄像头接入，输出热异常诊断结果和伪彩色红外画面。
- 通过 GPIO 控制步进电机，支持连续运动、定距移动、移动到指定位置、速度百分比和软件行程限位。
- Web 控制台提供视频流、红外流、电机控制、状态面板、数据集管理、样本复核、模型上传/部署和巡检报告入口。
- 集成 AD5933 阻抗检测模块，可进行低阻、开路、潮湿等辅助诊断。
- 支持 YOLO 数据集上传、误报样本收集、复核样本池、训练流水线、ONNX 导出和 RDK X5 BPU BIN 转换。
- 提供 OpenClaw 兼容的大模型助手接口，用于运维问答和巡检辅助；当前实现保留 `/api/claw/command` 兼容接口，并直接读取 OpenClaw 配置调用模型 API。

## 硬件与运行环境

- 开发板：地平线 RDK X5
- 系统：Ubuntu + ROS2 Humble
- 推理：`hobot_dnn` / Horizon BPU
- 主摄像头：默认从 `/camera/image_raw` 发布，可在启动参数中指定 `/dev/v4l/by-id/...`
- 红外摄像头：默认发布 `/infrared/image_raw`
- 电机控制：Hobot GPIO，BOARD 编号默认 `DIR=31`、`STEP=32`、`EN=36`、`BUTTON=37`
- Web 服务：`http://<board-ip>:8010`

## 项目结构

```text
cable1/
├── src/
│   ├── camera_pkg/          # 摄像头采集节点
│   ├── detection_pkg/       # YOLO26/BPU 粗定位、OpenCV 精定位和红外检测节点
│   ├── image_saver_pkg/     # 定时/缺陷触发图像保存节点
│   └── motor_control_pkg/   # 电机控制节点和 Flask Web 后端
├── ad5933_cable_analyzer/   # AD5933 阻抗/通断/开路定位分析模块
├── config/                  # 当前模型、运动位置和运行参数
├── scripts/                 # 训练到 BIN 转换流水线、状态脚本
├── tools/                   # DINOv3 蒸馏、YOLO26 训练/导出和 mapper 转换工具
├── third_party/tessdata/    # OCR/视觉相关第三方数据
├── agentic_aiops_clean_sidebar.html  # Web 前端页面
├── start_system.sh          # 板端一键启动脚本
└── WEB_GUIDE.md             # Web 控制台说明
```

> 使用 `start_system.sh` 或单独运行各个 ROS2 console script。

## 快速启动

在 RDK X5 板端执行：

```bash
cd /home/sunrise/cable1
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
./start_system.sh
```

启动后，在同一网络内访问：

```text
http://<board-ip>:8010
```

脚本会依次启动主摄像头、可选红外摄像头、红外检测、BPU 缺陷检测、电机控制、可选图像保存和 Web 控制节点。日志默认写入 `/tmp/camera.log`、`/tmp/infrared_detection.log`、`/tmp/detection.log`、`/tmp/motor.log`、`/tmp/web.log`。

## 单节点运行

调试时可以单独启动节点：

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

常用话题：

- `/camera/image_raw`：主摄像头图像
- `/detection/results`：视觉缺陷检测 JSON
- `/detection/annotated_image`：带标注的视觉画面
- `/infrared/results`：红外诊断 JSON
- `/infrared/annotated_image`：伪彩色红外画面
- `/motor/control`：电机控制命令
- `/motor/motion_status`：电机位置、速度、限位和运动状态

## 模型与数据

运行时默认使用：

```text
/home/sunrise/cable1/models/deployed/current.bin
```

`config/current_model.json` 会记录当前部署模型、类别、来源路径、哈希和部署时间。模型权重、BPU BIN、数据集、训练输出、巡检报告和采集图片通常体积较大，默认不提交到仓库，建议按以下目录放置在板端：

- `models/`：上传、暂存和部署后的模型
- `datasets/`：YOLO 数据集
- `runs/`：训练和转换输出
- `inspection_reports/`：巡检报告
- `false_positive_samples/`、`review_sample_pool/`：误报与复核样本

训练/转换入口：

```bash
python3 scripts/train_to_bin_pipeline.py \
  --dataset /home/sunrise/cable1/datasets/<dataset_name> \
  --output-dir /home/sunrise/cable1/runs/<run_name>
```

流水线会优先训练或复用 `.pt`。启用蒸馏时，DINOv3 只在训练阶段作为 teacher 提供特征监督，生成的 `best.pt` 仍是普通 YOLO26/YOLO26n 检测模型；随后导出 BPU 友好的 ONNX，并通过 `hb_mapper` 或 Docker 中的 OpenExplorer 工具链生成 RDK X5 可用的 `.bin`。板端运行时链路为 `YOLO26/BPU 粗定位 -> OpenCV ROI 精定位 -> 结果发布/报告融合`。

## Web 控制台

`web_control_node` 内置 Flask 服务，默认监听 `0.0.0.0:8010`。主要能力包括：

- `/video_feed`：视觉检测画面 MJPEG 流
- `/infrared_feed`：红外画面 MJPEG 流
- `/forward`、`/reverse`、`/stop`、`/move_to`：电机运动控制
- `/api/status`：系统、ROS 节点、BPU、CPU、内存、磁盘和温度状态
- `/api/inspection/*`：巡检会话与报告
- `/api/ad5933/*`：阻抗分析动作和状态
- `/api/datasets`、`/api/models`、`/api/pipeline/*`：数据集、模型和训练部署管理
- `/api/claw/command`：OpenClaw 兼容助手接口，读取 OpenClaw 风格配置后直连模型 API

前端页面为仓库根目录的 `agentic_aiops_clean_sidebar.html`。

## 依赖

板端核心依赖：

- ROS2 Humble：`rclpy`、`sensor_msgs`、`std_msgs`、`cv_bridge`
- Python：OpenCV、NumPy、Flask、Werkzeug
- 地平线运行库：`hobot_dnn`
- GPIO：`Hobot.GPIO`
- AD5933：`smbus2`
- 训练/转换环境：Ultralytics、PyTorch、ONNX Runtime、`hb_mapper` 或 OpenExplorer Docker 镜像

## 说明

这个仓库包含 `README_cn.md` 和 `README.MD`，便于中文展示和英文平台索引。若在 Windows 上克隆时看到 `README.md` 与 `README.MD` 的大小写冲突，建议仓库只保留 `README.MD` 作为英文 README，中文内容使用 `README_cn.md`。
