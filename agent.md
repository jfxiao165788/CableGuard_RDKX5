# RDK X5 Cable Defect Detection Agent Notes

最后更新：2026-05-02

## 项目位置

当前主要工作目录：

```text
/home/sunrise/cable1
```

原始备份目录：

```text
/home/sunrise/cable_defect_detection
```

后续操作默认只针对 `/home/sunrise/cable1`。不要改动原始目录，除非用户明确要求。

## 项目目标

这是一个基于 ROS2 的 RDK X5 电缆缺陷检测项目，包含：

- 摄像头采集
- YOLO26 电缆缺陷检测
- BPU `.bin` 模型推理
- 电机控制
- Web 控制台
- 右侧智能助手
- 数据集上传、训练、ONNX 导出、BIN 转换和模型部署流程

Web 品牌名已改为：

```text
日行八万里
```

## 当前运行状态

当前 Web 后端：

```text
http://192.168.128.10:8010
```

运行中的关键进程：

```text
camera_node
motor_control_node
web_control_node
```

旧版进程 `web_control_enhanced` 已清理：

```text
8000 端口：不应再监听
motor-web.service：disabled / inactive
web_control_enhanced：不应再运行
```

如果它又出现，优先检查：

```bash
systemctl --no-pager cat motor-web.service
systemctl --no-pager is-enabled motor-web.service
systemctl --no-pager is-active motor-web.service
```

## 重要注意事项

摄像头和电机功能已经调试好，默认不要动：

- 不要重启 `camera_node`
- 不要重启 `motor_control_node`
- 不要随意发布 `/motor/control`
- 不要替换正在运行的检测模型，除非用户明确确认

测试部署模型时，先放到：

```text
/home/sunrise/cable1/models
```

部署接口当前只做“暂存部署”，不会自动重启检测节点。

## ROS2 工作空间结构

主要目录：

```text
/home/sunrise/cable1/src/camera_pkg
/home/sunrise/cable1/src/detection_pkg
/home/sunrise/cable1/src/motor_control_pkg
/home/sunrise/cable1/launch
/home/sunrise/cable1/models
/home/sunrise/cable1/datasets
/home/sunrise/cable1/scripts
/home/sunrise/cable1/tools
```

主要节点：

```text
camera_pkg/camera_node
detection_pkg/detection_node
motor_control_pkg/motor_control_node
motor_control_pkg/web_control_node
```

Web 后端源码：

```text
/home/sunrise/cable1/src/motor_control_pkg/motor_control_pkg/web_control_node.py
```

Web 前端页面：

```text
/home/sunrise/cable1/agentic_aiops_clean_sidebar.html
```

## Web 后端接口

核心接口：

```text
GET  /
GET  /video_feed
GET  /status
POST /forward
POST /reverse
POST /stop
GET  /api/status
GET  /api/detections
POST /api/claw/command
```

数据集和训练相关接口：

```text
GET  /api/datasets
POST /api/datasets/upload
GET  /api/models
POST /api/pipeline/train
POST /api/pipeline/deploy
GET  /api/pipeline/status
```

右侧智能助手已确认可以连接模型 API。后端已做回复清洗，避免返回大量 Markdown 星号，例如 `**`、`* `、`#`。

## 数据集

当前已上传数据集：

```text
/home/sunrise/cable1/datasets/game.v1i.yolo26
```

本地原始数据集位置：

```text
C:\Users\hlj23\Desktop\cable\game.v1i.yolo26
```

数据集格式是 YOLO26/YOLO 检测格式，包含：

```text
data.yaml
train/images
train/labels
valid/images
valid/labels
runs/detect/train/weights/best.pt
yolo26n.pt
```

当前数据集统计：

```text
train images: 120
valid images: 28
total images: 148
labels: 150
```

## 模型与转换流程

用户指定的正式流程：

1. 本地使用 `yolo26n.pt` 训练得到 `best.pt`
2. 本地 Anaconda 环境 `yolo26` 使用官方脚本导出 ONNX
3. 本地 Docker OpenExplore 镜像执行 ONNX -> BIN
4. 将生成的 `.bin` 上传到 RDK X5 并部署

本地 Anaconda 环境：

```text
D:\anaconda\envs\yolo26
```

官方导出脚本：

```text
C:\Users\hlj23\Downloads\rdk_model_zoo-rdk_x5\samples\vision\yolo26\conversion\onnx_export\export_yolo26_detect_bpu.py
```

官方 mapper 脚本：

```text
C:\Users\hlj23\Downloads\rdk_model_zoo-rdk_x5\samples\vision\yolo26\conversion\mapper.py
```

板端脚本副本：

```text
/home/sunrise/cable1/tools/export_yolo26_detect_bpu.py
/home/sunrise/cable1/tools/mapper.py
```

当前已生成并上传的 BIN：

```text
/home/sunrise/cable1/models/best_yolo26_bpu_bayese_640x640_nv12.bin
```

## 当前转换环境状态

本地 Docker 已有 OpenExplore 镜像：

```text
openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```

正式流程不要求 RDK X5 安装 `hb_mapper` 或 Docker。RDK X5 只负责接收最终 `.bin`、运行 ROS2 检测和 Web 控制。

RDK X5 板卡当前缺少转换工具是可接受状态：

```text
hb_mapper: not found
docker: not found
onnxruntime: not found
```

## 已做过的精度验证

使用 `best.pt` 跳过训练，验证后半段流程时得到：

```text
PT mAP50: 0.960
BIN mAP50: 0.968
```

说明量化后的 BIN 在当前 28 张验证图上没有明显精度损失。正式流程使用本机 Docker 完成 ONNX -> BIN。

## 本地训练 Worker

本地训练 worker 文件：

```text
C:\Users\hlj23\Documents\Codex\2026-05-02\rdk-x5\local_training_worker.py
```

监听地址：

```text
http://127.0.0.1:8765
```

作用：

- 从 RDK X5 拉取所选数据集
- 在本地 `yolo26` 环境训练或复用 `best.pt`
- 本地导出 ONNX
- 本地 Docker 执行 ONNX -> BIN
- 上传最终 `.bin` 到 RDK X5 的 `/home/sunrise/cable1/models`

RDK X5 不需要安装转换工具链。

## 常用命令

查看 Web 后端状态：

```bash
ps -eo pid,ppid,cmd | grep -E 'web_control_node|web_control_enhanced' | grep -v grep
ss -ltnp | grep -E ':8000|:8010'
```

启动 `cable1` Web 后端：

```bash
cd /home/sunrise/cable1
source /opt/ros/humble/setup.bash
source install/setup.bash
nohup ros2 run motor_control_pkg web_control_node > /tmp/cable1_web.log 2>&1 < /dev/null &
```

重新编译 Web 包：

```bash
cd /home/sunrise/cable1
source /opt/ros/humble/setup.bash
colcon build --packages-select motor_control_pkg
```

测试龙虾助手 API：

```bash
python3 - <<'PY'
import json, urllib.request
payload=json.dumps({'text':'请简短说明当前RDK X5状态'}).encode('utf-8')
req=urllib.request.Request(
    'http://localhost:8010/api/claw/command',
    data=payload,
    headers={'Content-Type':'application/json'},
    method='POST'
)
with urllib.request.urlopen(req, timeout=90) as r:
    print(r.read().decode('utf-8'))
PY
```

## 交接建议

后续最优先事项：

1. 重新跑网页“一键训练出 bin”完整流程，确认本地 Docker worker 状态能实时显示
2. 在不影响摄像头和电机的前提下，增加模型热切换或受控重启检测节点的部署确认流程
3. 继续完善网页训练平台体验，包括任务日志、模型版本、精度报告和数据集版本管理

## 2026-05-02 模型应用与留痕更新

本次新增“应用新模型”能力：

- 页面训练区域新增按钮：`应用新模型`
- 新接口：`POST /api/pipeline/apply_model`
- 应用模型时只重启 `detection_node`
- 不重启 `camera_node`
- 不重启 `motor_control_node`
- 当前生效模型固定为：

```text
/home/sunrise/cable1/models/deployed/current.bin
```

模型版本与操作留痕：

- 本地 worker 新生成的 bin 会使用带时间戳和 run id 的文件名，避免覆盖历史模型
- 每个模型旁边保存同名元数据文件：

```text
/home/sunrise/cable1/models/<model>.bin.json
```

- 板端追加式历史文件：

```text
/home/sunrise/cable1/config/model_history.jsonl
```

- 当前应用模型记录：

```text
/home/sunrise/cable1/config/current_model.json
```

新增接口：

```text
GET  /api/model_history
POST /api/models/register
POST /api/pipeline/apply_model
```

已验证：

- `POST /api/pipeline/apply_model` 可将模型应用到检测节点
- `detection_node` 已使用 `/home/sunrise/cable1/models/deployed/current.bin` 成功启动
- 摄像头节点和电机节点未被重启
- `/api/models` 会返回模型训练时间、数据集、是否暂存、是否应用
- `/api/model_history` 会返回训练上传、暂存部署、应用模型等历史记录

## 2026-05-02 历史模型按钮

训练与部署页面新增按钮：

```text
历史模型
```

点击后打开历史模型弹窗，读取：

```text
GET /api/model_history
GET /api/models
```

弹窗展示字段：

- 模型名称
- 模型 ID
- 模型状态：训练上传 / 暂存部署 / 已应用 / 历史文件
- 数据集
- 训练时间
- 模型大小
- 训练模式
- epochs / imgsz / batch
- 上传时间
- 暂存时间
- 应用时间
- 模型路径
- 来源路径

弹窗内支持按模型名、数据集、状态、路径搜索。

历史模型弹窗追加能力：

- 每条历史模型记录右侧新增 `应用此模型` 按钮
- 当前已经应用的模型显示 `当前已应用`，按钮禁用
- 点击 `应用此模型` 会调用：

```text
POST /api/pipeline/apply_model
```

- 该操作会把所选历史模型应用为当前视觉检测模型
- 只重启 `detection_node`
- 不重启摄像头节点
- 不重启电机节点
- 应用后自动刷新模型列表和历史记录

本地 worker 更新：

- 文件：

```text
C:\Users\hlj23\Documents\Codex\2026-05-02\rdk-x5\local_training_worker.py
```

- 后续一键训练上传的模型会自动携带：
  - dataset
  - epochs
  - imgsz
  - batch
  - training_mode
  - trained_at
  - onnx_exported_at
  - bin_generated_at
  - uploaded_at
  - local_run_id
  - board_model_path

## 2026-05-02 实时状态界面更新

本次把网页状态页和关键卡片改为真实实时数据，不再使用随机数或演示值。

后端增强：

- /api/status 新增真实 BPU 信息，来源优先使用 hrut_somstatus
  - system.bpu.percent
  - system.bpu.temperature_c
  - system.bpu.cur_freq_mhz
  - system.bpu.source
- /api/status 新增真实 uptime、ROS2 节点状态表、数据集总量、误检样本统计、当前模型摘要、最近后端日志。
- 视频 FPS 改为根据 /detection/annotated_image 实际帧到达时间计算，不再使用配置值。
- ROS2 节点状态会标记 online / offline / duplicate。

前端更新：

- 资源占用、健康分、温度、FPS、网络延迟、ROS2 节点表全部轮询真实 API。
- 数据集页从 /api/datasets 和 /api/status.datasets 生成真实统计。
- 误检样本来自 /home/sunrise/cable1/false_positive_samples/metadata.jsonl。
- 模型生命周期读取 /api/models 和 /api/status.models.current。
- 检测事件来自 /api/status.detection，无检测结果时显示未上报，不再展示假 bbox/置信度。
- 页面源码已移除 Math.random 和演示数值回退；接口离线时显示未连接/未上报。

部署验证：

- 已执行 colcon build --packages-select motor_control_pkg。
- 只重启了 web_control_node，没有重启 camera_node、motor_control_node、detection_node。
- 当前 Web 地址仍为 http://192.168.128.10:8010。
- 验证时 /api/status 返回真实值：CPU 约 40%，BPU ratio 约 12%，温度约 79°C，内存约 86%，视频约 17 FPS，数据集图像 148 张，误检样本 1 条。

注意：当前真实健康状态为 warn，原因包括温度/内存偏高，以及 ROS graph 中存在重复的 motor_control_node。这是真实状态展示，不是页面误报。

## 2026-05-02 状态页与误检闭环增强

状态页顶部健康卡已增强：

- 健康分环形图继续实时读取 /api/status.health.score
- 新增风险原因卡片，实时分析：温度、BPU 温度、CPU/BPU/内存/磁盘占用、视频帧、ROS2 节点离线或重复
- 新增更新时间、BPU 数据来源、workspace
- 新增运行时间、当前模型、BPU 温度、视频帧状态
- 新增运行摘要：设备、模型数量、数据集图像、误检样本

误检闭环页面增强：

- 上传/创建训练集 按钮接入真实 /api/datasets/upload
- 误检入库弹窗新增 目标数据集 选择
- POST /api/dataset/false_positive 现在会：
  - 保存误检留痕到 /home/sunrise/cable1/false_positive_samples
  - 优先使用 /camera/image_raw 原始帧
  - 写入所选数据集的 <split>/images/<sample>.jpg
  - 写入所选数据集的 <split>/labels/<sample>.txt
  - 空标签模式会生成空 label，作为 hard negative 负样本
- 新增真实统计报表接口：GET /api/dataset/report
- 新增 AI 误检分析接口：POST /api/dataset/false_positive/analyze
  - 使用现有模型 API 配置
  - 结合当前检测结果、误检记录、数据集统计、当前模型生成分析
- 闭环任务区域扩大，新增目标数据集、最近误检、训练状态、当前模型，并接入 AI 分析、入库、训练、部署按钮

验证：

- colcon build --packages-select motor_control_pkg 成功，仅有 setuptools tests_require 警告
- 只重启了 web_control_node
- 已验证 /api/dataset/report 返回真实统计：图像 148、标签 150、误检样本 1
- 已验证不存在的数据集会返回错误，不会默认写入第一个数据集
- 已验证 AI 误检分析接口可返回真实模型分析
- 验证过程中曾误触发一次测试入库样本 FP_20260502_230708_893，已按明确路径逐个删除图片、标签和留痕 metadata 行，统计已恢复

## 2026-05-02 ROS2 节点状态防抖更新

状态页 ROS2 节点表已改为稳定判定，避免 ROS2 discovery 或 os2 node list 偶发超时造成离线误报：

- /api/status.system.ros_graph 新增 source/stale/age_sec/error
- os2 node list 成功时更新缓存
- 查询失败或超时时使用最近一次成功结果，source=cached
- 没有缓存时使用进程表兜底，source=process
- 节点短时间缺失会进入 grace 窗口，source=grace
- 只有超过 grace 且进程也不存在时才标为 offline
- 同名多实例不再作为健康风险，只在表格中显示 online xN
- 前端说明改为 稳定判定，缓存/兜底结果会显示稳定保持

当前验证：os2 node list 偶发 5 秒超时时，页面仍显示 camera/detection/motor/web 为 online，source=cached，stale=true，不再抖动成离线。
