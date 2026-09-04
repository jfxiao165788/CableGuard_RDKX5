#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def run_configured_backend(args):
    cmd_template = os.getenv("DINOV3_DISTILL_CMD", "").strip()
    if not cmd_template:
        return False

    env = os.environ.copy()
    env.update({
        "DINOV3_TEACHER": args.teacher,
        "YOLO26_STUDENT_PT": args.student,
        "YOLO26_DATA_YAML": args.data,
        "YOLO26_DISTILLED_PT": args.output,
    })
    log("Running configured DINOv3 distillation backend from DINOV3_DISTILL_CMD")
    proc = subprocess.run(cmd_template, shell=True, env=env, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return os.path.exists(args.output)


def has_dino_hf_files(path):
    return (
        path
        and os.path.isdir(path)
        and os.path.exists(os.path.join(path, "config.json"))
        and os.path.exists(os.path.join(path, "model.safetensors"))
    )


def run_local_distill_script(args):
    script = os.getenv(
        "DINOV3_DISTILL_SCRIPT",
        os.path.join(SCRIPT_DIR, "train_dinov3_yolo26_distill.py"),
    )
    if not os.path.exists(script):
        return False
    if not has_dino_hf_files(args.teacher):
        return False

    project = os.path.join(os.path.dirname(os.path.abspath(args.output)), "dinov3_train")
    cmd = [
        sys.executable,
        script,
        "--data", args.data,
        "--student", args.student,
        "--teacher", args.teacher,
        "--output", args.output,
        "--project", project,
        "--name", "yolo26n_dinov3_distill",
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
    ]
    log("Running local DINOv3 distillation script: " + " ".join(cmd))
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return os.path.exists(args.output)


def write_manifest(args, mode):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "teacher": args.teacher,
        "student": args.student,
        "data": args.data,
        "output": args.output,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "note": (
            "DINOv3 distillation adapter executed. Provide a teacher directory "
            "with config.json and model.safetensors, set DINOV3_DISTILL_SCRIPT, "
            "or set DINOV3_DISTILL_CMD to run a site-specific backend."
        ),
    }
    with open(args.output + ".distill.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="dinov3")
    parser.add_argument("--student", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", default=os.getenv("DINOV3_DISTILL_EPOCHS", "30"))
    parser.add_argument("--imgsz", default=os.getenv("DINOV3_DISTILL_IMGSZ", "640"))
    parser.add_argument("--batch", default=os.getenv("DINOV3_DISTILL_BATCH", "4"))
    args = parser.parse_args()

    if not os.path.exists(args.student):
        raise SystemExit(f"student PT not found: {args.student}")
    if not os.path.exists(args.data):
        raise SystemExit(f"data.yaml not found: {args.data}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if run_configured_backend(args):
        log("DINOv3 distillation backend produced: " + args.output)
        write_manifest(args, "configured_backend")
        return
    if run_local_distill_script(args):
        log("Local DINOv3 distillation script produced: " + args.output)
        write_manifest(args, "local_script")
        return

    shutil.copy2(args.student, args.output)
    log("DINOv3 backend is not configured; copied YOLO26 student PT as distillation output.")
    log("Set DINOV3_DISTILL_CMD to enable the real DINOv3 teacher-student step.")
    write_manifest(args, "adapter_copy")


if __name__ == "__main__":
    main()
