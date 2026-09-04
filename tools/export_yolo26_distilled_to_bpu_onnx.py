#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_yolo26_distilled_to_bpu_onnx.py

用途：
    将 YOLO26 / YOLO26n 的 .pt 权重导出为适合 RDK X5 BPU mapper 转换的 ONNX。

适用：
    1. 普通 YOLO26 训练得到的 best.pt
    2. DINOv3 蒸馏训练得到的 best.pt

说明：
    DINOv3 只在训练阶段作为 teacher。
    只要你的训练脚本保存时没有把 teacher 保存进模型，最终 best.pt 仍然是普通 YOLO26 检测模型。
    因此 PT -> ONNX 不需要导出 DINOv3。

导出输出：
    Detect 头输出 3 个尺度的原始输出，顺序为：
        [box_s8, cls_s8, box_s16, cls_s16, box_s32, cls_s32]
    输出 layout 为 NHWC，适合后续 RDK X5 BPU 后处理。

运行示例：
    python export_yolo26_distilled_to_bpu_onnx.py ^
      --weights runs_dinov3_distill/yolo26n_dinov3_from/weights/best.pt ^
      --output deploy/yolo26_surface_bpu.onnx ^
      --imgsz 640
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect


def bpu_detect_forward(self, x):
    """
    将 YOLO26 Detect head 改为 BPU 友好的 raw 输出。

    返回：
        [box_0, cls_0, box_1, cls_1, box_2, cls_2]

    每个输出：
        NCHW -> NHWC
    """
    outputs = []

    # YOLO26 / end2end 版本可能存在 one2one 分支。
    # 这里优先用 one2one，与很多 YOLO26 BPU 导出脚本保持一致。
    if hasattr(self, "one2one_cv2") and hasattr(self, "one2one_cv3"):
        box_layers = self.one2one_cv2
        cls_layers = self.one2one_cv3
        branch = "one2one"
    else:
        box_layers = self.cv2
        cls_layers = self.cv3
        branch = "normal"

    if not hasattr(self, "_bpu_branch_printed"):
        print(f"[Export] Detect branch: {branch}")
        self._bpu_branch_printed = True

    for i in range(self.nl):
        box = box_layers[i](x[i]).permute(0, 2, 3, 1)
        cls = cls_layers[i](x[i]).permute(0, 2, 3, 1)
        outputs.append(box)
        outputs.append(cls)

    return outputs


def print_model_info(model: YOLO):
    print("=" * 80)
    print("[Export] Model information")

    try:
        print(f"[Export] names: {model.names}")
        print(f"[Export] nc: {len(model.names)}")
    except Exception as e:
        print(f"[Export] Cannot read model names: {e}")

    try:
        detect = model.model.model[-1]
        print(f"[Export] Detect module: {type(detect).__name__}")
        print(f"[Export] Detect nc: {getattr(detect, 'nc', None)}")
        print(f"[Export] Detect nl: {getattr(detect, 'nl', None)}")
        print(f"[Export] Detect stride: {getattr(detect, 'stride', None)}")
    except Exception as e:
        print(f"[Export] Cannot read Detect info: {e}")

    print("=" * 80)


def verify_onnx(onnx_path: str):
    print("=" * 80)
    print("[Export] ONNX check")

    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        print("[Export] onnx.checker: OK")
    except Exception as e:
        print(f"[Export] onnx checker skipped/failed: {e}")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        print("[Export] Inputs:")
        for inp in sess.get_inputs():
            print(f"  name={inp.name}, shape={inp.shape}, type={inp.type}")

        print("[Export] Outputs:")
        for out in sess.get_outputs():
            print(f"  name={out.name}, shape={out.shape}, type={out.type}")
    except Exception as e:
        print(f"[Export] onnxruntime check skipped/failed: {e}")

    print("=" * 80)


def export_onnx(weights: str, output: str, imgsz: int, opset: int, simplify: bool, check: bool):
    weights = str(Path(weights).resolve())
    output = str(Path(output).resolve())

    if not os.path.exists(weights):
        raise FileNotFoundError(f"权重文件不存在: {weights}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print("=" * 80)
    print("[Export] YOLO26 PT -> BPU-friendly ONNX")
    print(f"[Export] weights: {weights}")
    print(f"[Export] output : {output}")
    print(f"[Export] imgsz  : {imgsz}")
    print(f"[Export] opset  : {opset}")
    print("=" * 80)

    torch.set_grad_enabled(False)

    model = YOLO(weights)
    print_model_info(model)

    # Monkey patch Detect.forward
    Detect.forward = bpu_detect_forward

    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        opset=opset,
        simplify=simplify,
        verbose=False,
    )

    if not exported_path:
        raise RuntimeError("导出失败：Ultralytics 没有返回 ONNX 路径。")

    exported_path = str(Path(exported_path).resolve())

    if os.path.abspath(exported_path) != os.path.abspath(output):
        if os.path.exists(output):
            os.remove(output)
        shutil.move(exported_path, output)

    print(f"[Export] ONNX saved: {output}")

    if check:
        verify_onnx(output)

    print("[Export] Done.")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO26 distilled PT to RDK X5 BPU ONNX")
    parser.add_argument("--weights", type=str, required=True, help="YOLO26 best.pt 路径")
    parser.add_argument("--output", type=str, default="yolo26_bpu.onnx", help="输出 ONNX 路径")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸，建议 640")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset，RDK X5 常用 11")
    parser.add_argument("--no-simplify", action="store_true", help="关闭 ONNX simplify")
    parser.add_argument("--no-check", action="store_true", help="不检查 ONNX 输入输出")

    args = parser.parse_args()

    export_onnx(
        weights=args.weights,
        output=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        check=not args.no_check,
    )


if __name__ == "__main__":
    main()
