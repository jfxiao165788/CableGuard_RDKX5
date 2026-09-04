#!/usr/bin/env python3
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from datetime import datetime


BASE_DIR = os.getenv(
    "CABLE_WORKSPACE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
TOOLS_DIR = os.path.join(BASE_DIR, "tools")


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def run(cmd, cwd=None):
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def find_existing_best_pt(dataset):
    candidates = [
        os.path.join(dataset, "runs", "detect", "train", "weights", "best.pt"),
        os.path.join(dataset, "best.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def first_existing_tool(*names):
    for name in names:
        path = os.path.join(TOOLS_DIR, name)
        if os.path.exists(path):
            return path
    return os.path.join(TOOLS_DIR, names[0])


def module_available(name):
    return importlib.util.find_spec(name) is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", default="80")
    parser.add_argument("--imgsz", default="640")
    parser.add_argument("--batch", default="8")
    parser.add_argument("--weights", default="")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--distill", default="")
    parser.add_argument("--defect-note", default="")
    args = parser.parse_args()

    dataset = os.path.abspath(args.dataset)
    output_dir = os.path.abspath(args.output_dir)
    data_yaml = os.path.join(dataset, "data.yaml")

    if not os.path.exists(data_yaml):
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    os.makedirs(output_dir, exist_ok=True)
    pt_path = find_existing_best_pt(dataset)

    if args.skip_train:
        if not args.weights or not os.path.exists(args.weights):
            raise SystemExit("--skip-train requires an uploaded .pt path via --weights")
        pt_path = os.path.abspath(args.weights)
        log(f"Using uploaded PT directly: {pt_path}")
    elif pt_path:
        log(f"Reusing existing trained PT: {pt_path}")
    else:
        weights = args.weights or os.path.join(dataset, "yolo26n.pt")
        if not os.path.exists(weights):
            raise SystemExit("No existing best.pt and no yolo26n.pt weights found.")

        log("Training YOLO26 model...")
        train_project = os.path.join(output_dir, "train")
        yolo_cmd = shutil.which("yolo")
        if yolo_cmd:
            cmd = [
                yolo_cmd, "detect", "train",
                f"model={weights}",
                f"data={data_yaml}",
                f"epochs={args.epochs}",
                f"imgsz={args.imgsz}",
                f"batch={args.batch}",
                f"project={train_project}",
                "name=yolo26_train",
                "exist_ok=True",
            ]
        else:
            cmd = [
                "python3", "-c",
                (
                    "from ultralytics import YOLO; "
                    "import sys; "
                    "model=YOLO(sys.argv[1]); "
                    "model.train(data=sys.argv[2], epochs=int(sys.argv[3]), imgsz=int(sys.argv[4]), "
                    "batch=int(sys.argv[5]), project=sys.argv[6], name='yolo26_train', exist_ok=True)"
                ),
                weights, data_yaml, str(args.epochs), str(args.imgsz), str(args.batch), train_project,
            ]
        run(cmd)
        pt_path = os.path.join(train_project, "yolo26_train", "weights", "best.pt")

    if not os.path.exists(pt_path):
        raise SystemExit(f"best.pt not found after training: {pt_path}")

    export_runtime_ok = module_available("ultralytics") and module_available("torch")
    mapper_runtime_ok = module_available("onnxruntime") and (shutil.which("hb_mapper") or shutil.which("docker"))
    if not export_runtime_ok or not mapper_runtime_ok:
        missing = []
        if not export_runtime_ok:
            missing.append("ultralytics/torch")
        if not module_available("onnxruntime"):
            missing.append("onnxruntime")
        if not shutil.which("hb_mapper") and not shutil.which("docker"):
            missing.append("hb_mapper/docker")
        raise SystemExit("Real PT->ONNX->BIN pipeline unavailable on this host: missing " + ", ".join(missing))

    if args.distill:
        distill_script = os.path.join(TOOLS_DIR, "dinov3_distill_yolo26.py")
        log(f"DINOv3 distillation requested: {args.distill}")
        if os.path.exists(distill_script):
            distilled_pt = os.path.join(output_dir, "dinov3_distilled_best.pt")
            log("Running DINOv3 distillation...")
            run([
                "python3",
                distill_script,
                "--teacher", args.distill,
                "--student", pt_path,
                "--data", data_yaml,
                "--output", distilled_pt,
                "--epochs", str(args.epochs),
                "--imgsz", str(args.imgsz),
                "--batch", str(args.batch),
            ])
            if os.path.exists(distilled_pt):
                pt_path = distilled_pt
        else:
            log("DINOv3 distillation script is not configured; continuing with the current YOLO26 PT.")

    if args.defect_note:
        log("Supplemented defect note: " + args.defect_note)

    onnx_path = os.path.join(output_dir, "best_yolo26_bpu.onnx")
    export_script = first_existing_tool(
        "export_yolo26_distilled_to_bpu_onnx.py",
        "export_yolo26_detect_bpu.py",
    )
    if not os.path.exists(export_script):
        raise SystemExit(f"Export script not found: {export_script}")

    log("Exporting PT to BPU-friendly ONNX...")
    run([
        "python3",
        export_script,
        "--weights", pt_path,
        "--output", onnx_path,
        "--imgsz", str(args.imgsz),
    ])

    mapper_script = first_existing_tool(
        "mapper_yolo26_raw_images_to_bpu.py",
        "mapper.py",
    )
    cal_images = os.path.join(dataset, "valid", "images")
    if not os.path.exists(mapper_script):
        raise SystemExit(f"Mapper script not found: {mapper_script}")
    if not os.path.isdir(cal_images):
        raise SystemExit(f"Calibration image directory not found: {cal_images}")

    log("Converting ONNX to BPU BIN with mapper...")
    if shutil.which("hb_mapper"):
        run([
            "python3", mapper_script,
            "--onnx", onnx_path,
            "--cal-images", cal_images,
            "--output-dir", output_dir,
            "--ws", os.path.join(output_dir, "mapper_ws"),
        ])
    elif shutil.which("docker"):
        image = "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8"
        run([
            "docker", "run", "--rm",
            "-v", f"{output_dir}:/data",
            "-v", f"{TOOLS_DIR}:/tools",
            "-v", f"{dataset}:/dataset:ro",
            image,
            "/bin/bash", "-lc",
            (
                f"python3 /tools/{os.path.basename(mapper_script)} "
                "--onnx /data/best_yolo26_bpu.onnx "
                "--cal-images /dataset/valid/images "
                "--output-dir /data "
                "--ws /data/mapper_ws"
            ),
        ])
    else:
        raise SystemExit("Neither hb_mapper nor docker is available in this runtime.")

    bins = [x for x in os.listdir(output_dir) if x.endswith(".bin")]
    if not bins:
        raise SystemExit("Pipeline finished but no .bin model was produced.")

    log("BIN model ready: " + os.path.join(output_dir, bins[0]))


if __name__ == "__main__":
    main()
