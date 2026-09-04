#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mapper_yolo26_raw_images_to_bpu.py

用途：
    在 D-Robotics / Horizon / 地平线 OpenExplore Docker 里，
    将 YOLO26 的 float ONNX 转成 RDK X5 可运行的 BPU .bin。

适用情况：
    你的训练数据集是普通图片，不是 ROI + 掩膜 + 滑动窗口后的 tile 图。
    因此校准图片也应该使用普通图片，最好使用训练/验证集中同分布的原图。

重要：
    1. 必须在 OpenExplore Docker 里运行，因为需要 hb_mapper。
    2. 校准图片不要只放缺陷图，也要放正常图。
    3. 默认使用 letterbox 预处理，尽量对齐你 ROS detection_node 里的 preprocess_image：
        等比例缩放 + 灰色补边 127 + RGB NCHW + scale 1/255。
    4. 如果你的真实推理代码是直接 resize，不是 letterbox，可以加：
        --cal-preprocess resize

运行示例：
    python mapper_yolo26_raw_images_to_bpu.py \
      --onnx /workspace/deploy/yolo26_bpu.onnx \
      --cal-images /workspace/deploy/cal_raw_images \
      --output-dir /workspace/deploy/bpu_output \
      --cal-sample-num 200 \
      --quantized int8

如果 int8 精度损失明显，可以试 int16：
    python mapper_yolo26_raw_images_to_bpu.py \
      --onnx /workspace/deploy/yolo26_bpu.onnx \
      --cal-images /workspace/deploy/cal_raw_images \
      --output-dir /workspace/deploy/bpu_output_int16 \
      --cal-sample-num 200 \
      --quantized int16
"""

import argparse
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("YOLO26_RawImage_Mapper")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 ONNX -> RDK X5 BPU bin")

    parser.add_argument("--onnx", type=str, required=True, help="输入 float ONNX 模型")
    parser.add_argument("--cal-images", type=str, required=True, help="校准图片目录，使用普通原图")
    parser.add_argument("--output-dir", type=str, default="./bpu_output", help="输出目录")
    parser.add_argument("--ws", type=str, default="./.mapper_workspace", help="临时工作目录")

    parser.add_argument("--quantized", type=str, default="int8", choices=["int8", "int16"], help="量化精度")
    parser.add_argument("--jobs", type=int, default=16, help="hb_mapper jobs")
    parser.add_argument("--optimize-level", type=str, default="O3", choices=["O0", "O1", "O2", "O3"], help="编译优化等级")

    parser.add_argument("--cal-sample-num", type=int, default=200, help="校准图片数量，建议 100~300")
    parser.add_argument("--no-sample", action="store_true", help="不用抽样，使用所有校准图片")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")

    parser.add_argument("--cal-preprocess", type=str, default="letterbox", choices=["letterbox", "resize"], help="校准预处理方式")
    parser.add_argument("--pad-value", type=int, default=127, help="letterbox 补边值")

    parser.add_argument("--march", type=str, default="bayes-e", help="RDK X5 使用 bayes-e")
    parser.add_argument("--input-type-rt", type=str, default="nv12", help="runtime 输入类型")
    parser.add_argument("--input-type-train", type=str, default="rgb", help="calibration 输入类型")
    parser.add_argument("--calibration-type", type=str, default="default", help="hb_mapper calibration_type")
    parser.add_argument("--save-cache", action="store_true", help="保留临时工作目录")

    return parser.parse_args()


def require_hb_mapper():
    try:
        ret = subprocess.run(["hb_mapper", "--version"], capture_output=True, text=True, check=True)
        logger.info("hb_mapper OK")
        if ret.stdout.strip():
            logger.info(ret.stdout.strip())
    except Exception as e:
        raise RuntimeError("找不到 hb_mapper。请在 OpenExplore Docker 里运行本脚本。") from e


def require_onnxruntime():
    try:
        import onnxruntime as ort
        return ort
    except Exception as e:
        raise RuntimeError("当前环境缺少 onnxruntime。请在 Docker 内安装 onnxruntime，或使用带 onnxruntime 的 OpenExplore 镜像。") from e


def analyze_onnx(onnx_path: Path):
    ort = require_onnxruntime()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()

    if len(inputs) != 1:
        raise RuntimeError(f"ONNX 输入数量为 {len(inputs)}，期望 1。")

    inp = inputs[0]
    shape = list(inp.shape)
    input_type = inp.type

    if input_type != "tensor(float)":
        raise RuntimeError(f"ONNX 输入类型为 {input_type}，期望 tensor(float)。")

    if len(shape) != 4:
        raise RuntimeError(f"ONNX 输入 shape 为 {shape}，期望 NCHW 4 维。")

    if not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise RuntimeError(f"ONNX 输入 H/W 不是固定值：{shape}。请确保 dynamic=False。")

    n, c, h, w = shape
    if c != 3:
        raise RuntimeError(f"ONNX 输入通道数为 {c}，期望 3。")

    logger.info(f"ONNX input name : {inp.name}")
    logger.info(f"ONNX input shape: {shape}")
    logger.info("ONNX outputs:")
    for out in sess.get_outputs():
        logger.info(f"  {out.name}: shape={out.shape}, type={out.type}")

    return inp.name, w, h


def list_images(cal_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in cal_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def letterbox_bgr(img: np.ndarray, dst_w: int, dst_h: int, pad_value: int = 127) -> np.ndarray:
    src_h, src_w = img.shape[:2]

    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_left = (dst_w - new_w) // 2
    pad_top = (dst_h - new_h) // 2
    pad_right = dst_w - new_w - pad_left
    pad_bottom = dst_h - new_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )

    return padded


def preprocess_image_to_rgb_nchw_float32(img_path: Path, width: int, height: int, mode: str, pad_value: int) -> np.ndarray:
    """
    输出 RGB NCHW float32，数值范围 0~255。
    scale=1/255 交给 hb_mapper 的 config 做。
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图片: {img_path}")

    if mode == "letterbox":
        img = letterbox_bgr(img, width, height, pad_value=pad_value)
    else:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1))
    nchw = np.expand_dims(chw, axis=0).astype(np.float32)

    return nchw


def prepare_calibration_data(img_paths: List[Path], cal_data_dir: Path, width: int, height: int, args):
    cal_data_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for idx, img_path in enumerate(img_paths):
        try:
            tensor = preprocess_image_to_rgb_nchw_float32(
                img_path=img_path,
                width=width,
                height=height,
                mode=args.cal_preprocess,
                pad_value=args.pad_value,
            )

            # 文件名不要太长，避免 Docker / mapper 下路径问题
            out_path = cal_data_dir / f"{idx:06d}.rgbchw"
            tensor.tofile(str(out_path))
            ok += 1
        except Exception as e:
            logger.warning(f"跳过校准图片 {img_path}: {e}")

    if ok == 0:
        raise RuntimeError("没有成功生成校准数据。")

    logger.info(f"Calibration blobs generated: {ok}")


def write_mapper_config(config_path: Path, onnx_path: Path, input_name: str, width: int, height: int,
                        output_prefix: str, bpu_output_dir: Path, cal_data_dir: Path, args):
    int16_opt = ",set_all_nodes_int16" if args.quantized == "int16" else ""

    yaml_text = f"""model_parameters:
  onnx_model: '{onnx_path}'
  march: '{args.march}'
  layer_out_dump: False
  working_dir: '{bpu_output_dir}'
  output_model_file_prefix: '{output_prefix}'

input_parameters:
  input_name: '{input_name}'
  input_type_rt: '{args.input_type_rt}'
  input_type_train: '{args.input_type_train}'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451

calibration_parameters:
  cal_data_dir: '{cal_data_dir}'
  cal_data_type: 'float32'
  calibration_type: '{args.calibration_type}'
  optimization: set_Softmax_input_int8,set_Softmax_output_int8{int16_opt}

compiler_parameters:
  jobs: {args.jobs}
  compile_mode: 'latency'
  debug: true
  optimize_level: '{args.optimize_level}'
"""
    config_path.write_text(yaml_text, encoding="utf-8")
    logger.info(f"Mapper config saved: {config_path}")


def run_cmd(cmd: List[str], cwd: Path):
    logger.info("Running: " + " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(cwd))
    if ret.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(cmd)}")


def main():
    args = parse_args()

    onnx_path = Path(args.onnx).resolve()
    cal_dir = Path(args.cal_images).resolve()
    output_dir = Path(args.output_dir).resolve()
    ws = Path(args.ws).resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 不存在: {onnx_path}")
    if not cal_dir.exists():
        raise FileNotFoundError(f"校准图片目录不存在: {cal_dir}")

    require_hb_mapper()

    input_name, width, height = analyze_onnx(onnx_path)

    img_paths = list_images(cal_dir)
    if not img_paths:
        raise RuntimeError(f"校准目录中没有图片: {cal_dir}")

    if not args.no_sample and len(img_paths) > args.cal_sample_num:
        random.seed(args.seed)
        img_paths = random.sample(img_paths, args.cal_sample_num)

    logger.info(f"Calibration image count: {len(img_paths)}")
    logger.info(f"Calibration preprocess: {args.cal_preprocess}")

    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    cal_data_dir = ws / "calibration_data"
    bpu_output_dir = ws / "bpu_model_output"

    model_base = onnx_path.stem
    march_name = args.march.replace("-", "")
    output_prefix = f"{model_base}_{march_name}_{width}x{height}_{args.input_type_rt}_{args.quantized}"

    prepare_calibration_data(
        img_paths=img_paths,
        cal_data_dir=cal_data_dir,
        width=width,
        height=height,
        args=args,
    )

    config_path = ws / "config.yaml"

    write_mapper_config(
        config_path=config_path,
        onnx_path=onnx_path,
        input_name=input_name,
        width=width,
        height=height,
        output_prefix=output_prefix,
        bpu_output_dir=bpu_output_dir,
        cal_data_dir=cal_data_dir,
        args=args,
    )

    # 保存一份 config，便于以后复现
    shutil.copy2(config_path, output_dir / "config.yaml")

    run_cmd(["hb_mapper", "makertbin", "--config", "config.yaml", "--model-type", "onnx"], cwd=ws)

    bin_src = bpu_output_dir / f"{output_prefix}.bin"

    if not bin_src.exists():
        candidates = list(bpu_output_dir.glob("*.bin"))
        if len(candidates) == 1:
            bin_src = candidates[0]
        else:
            raise RuntimeError(f"没有找到输出 bin 文件，目录：{bpu_output_dir}")

    bin_dst = output_dir / bin_src.name
    if bin_dst.exists():
        bin_dst.unlink()
    shutil.move(str(bin_src), str(bin_dst))

    # 复制常见日志
    for log_name in ["hb_mapper_makertbin.log", "hb_mapper_checker.log"]:
        log_path = ws / log_name
        if log_path.exists():
            shutil.copy2(log_path, output_dir / log_name)

    logger.info("=" * 80)
    logger.info("转换完成")
    logger.info(f"BPU bin: {bin_dst}")
    logger.info("=" * 80)

    if not args.save_cache:
        shutil.rmtree(ws)
        logger.info("临时工作目录已清理")


if __name__ == "__main__":
    main()
