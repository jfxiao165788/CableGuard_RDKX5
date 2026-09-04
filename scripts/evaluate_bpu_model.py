#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn


CLASSES = ["burn", "puncture"]
STRIDES = [8, 16, 32]
INPUT_SIZE = 640


def image_files(path):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.lower().endswith(exts)
    )


def label_path_for(image_path, labels_dir):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, stem + ".txt")


def load_labels(label_path, width, height):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = [float(x) for x in parts[1:5]]
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height
            boxes.append({
                "class_id": cls_id,
                "bbox": [x1, y1, x2, y2],
                "matched": False,
            })
    return boxes


def preprocess(frame):
    height, width = frame.shape[:2]
    scale = min(INPUT_SIZE / height, INPUT_SIZE / width)
    new_w, new_h = int(width * scale), int(height * scale)
    x_shift = (INPUT_SIZE - new_w) // 2
    y_shift = (INPUT_SIZE - new_h) // 2

    resized = cv2.resize(frame, (new_w, new_h))
    padded = cv2.copyMakeBorder(
        resized,
        y_shift,
        INPUT_SIZE - new_h - y_shift,
        x_shift,
        INPUT_SIZE - new_w - x_shift,
        cv2.BORDER_CONSTANT,
        value=127,
    )

    yuv = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420).flatten()
    nv12 = np.empty((INPUT_SIZE * INPUT_SIZE * 3 // 2,), dtype=np.uint8)
    y_size = INPUT_SIZE * INPUT_SIZE
    nv12[:y_size] = yuv[:y_size]
    nv12[y_size::2] = yuv[y_size:y_size + y_size // 4]
    nv12[y_size + 1::2] = yuv[y_size + y_size // 4:]
    return nv12, scale, x_shift, y_shift, width, height


def make_grids():
    grids = {}
    for stride in STRIDES:
        gh, gw = INPUT_SIZE // stride, INPUT_SIZE // stride
        grid = np.stack(np.indices((gh, gw))[::-1], axis=-1)
        grids[stride] = grid.reshape(-1, 2).astype(np.float32) + 0.5
    return grids


def postprocess(outputs, scale, x_shift, y_shift, width, height, conf_thresh, nms_thresh, grids):
    dets = []
    conf_raw = -np.log(1 / conf_thresh - 1)
    num_cls = len(CLASSES)

    for i, stride in enumerate(STRIDES):
        h_grid, w_grid = INPUT_SIZE // stride, INPUT_SIZE // stride
        box_data = outputs[i * 2].buffer.reshape(h_grid, w_grid, 4)
        cls_data = outputs[i * 2 + 1].buffer.reshape(h_grid, w_grid, num_cls)

        max_scores = np.max(cls_data, axis=2)
        mask = max_scores >= conf_raw
        if not np.any(mask):
            continue

        grid = grids[stride][mask.flatten()]
        v_box = box_data.reshape(-1, 4)[mask.flatten()]
        v_score = 1 / (1 + np.exp(-max_scores[mask]))
        v_id = np.argmax(cls_data.reshape(-1, num_cls)[mask.flatten()], axis=1)
        xyxy = np.hstack([(grid - v_box[:, :2]), (grid + v_box[:, 2:])]) * stride
        dets.extend(np.hstack([xyxy, v_score[:, None], v_id[:, None]]))

    results = []
    if not dets:
        return results

    dets = np.array(dets)
    xywh = dets[:, :4].copy()
    xywh[:, 2:] -= xywh[:, :2]
    indices = cv2.dnn.NMSBoxes(
        xywh.tolist(), dets[:, 4].tolist(), conf_thresh, nms_thresh
    )

    if len(indices) <= 0:
        return results

    for idx in indices.flatten():
        d = dets[idx]
        x1, y1, x2, y2 = (d[:4] - [x_shift, y_shift, x_shift, y_shift]) / scale
        results.append({
            "class_id": int(d[5]),
            "confidence": float(d[4]),
            "bbox": [
                float(np.clip(x1, 0, width)),
                float(np.clip(y1, 0, height)),
                float(np.clip(x2, 0, width)),
                float(np.clip(y2, 0, height)),
            ],
        })
    return results


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ap_from_pr(tp, fp, gt_count):
    if gt_count == 0:
        return None
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / max(gt_count, 1)
    precision = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(model_path, dataset_dir, conf_thresh, nms_thresh):
    images_dir = os.path.join(dataset_dir, "valid", "images")
    labels_dir = os.path.join(dataset_dir, "valid", "labels")
    model = dnn.load(model_path)[0]
    grids = make_grids()

    gt_by_image = {}
    preds_by_class = defaultdict(list)
    gt_count_by_class = defaultdict(int)

    for img_path in image_files(images_dir):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        labels = load_labels(label_path_for(img_path, labels_dir), w, h)
        image_id = os.path.basename(img_path)
        gt_by_image[image_id] = labels
        for label in labels:
            gt_count_by_class[label["class_id"]] += 1

        nv12, scale, x_shift, y_shift, width, height = preprocess(frame)
        outputs = model.forward(nv12)
        preds = postprocess(
            outputs, scale, x_shift, y_shift, width, height,
            conf_thresh, nms_thresh, grids
        )
        for pred in preds:
            preds_by_class[pred["class_id"]].append((image_id, pred))

    per_class = []
    total_tp = 0
    total_fp = 0
    total_gt = sum(gt_count_by_class.values())

    for cls_id, cls_name in enumerate(CLASSES):
        preds = sorted(
            preds_by_class.get(cls_id, []),
            key=lambda item: item[1]["confidence"],
            reverse=True,
        )
        tp_flags = []
        fp_flags = []
        matched = defaultdict(set)

        for image_id, pred in preds:
            candidates = [
                (idx, gt)
                for idx, gt in enumerate(gt_by_image.get(image_id, []))
                if gt["class_id"] == cls_id and idx not in matched[image_id]
            ]
            best_iou = 0.0
            best_idx = None
            for idx, gt in candidates:
                val = iou(pred["bbox"], gt["bbox"])
                if val > best_iou:
                    best_iou = val
                    best_idx = idx
            if best_iou >= 0.5 and best_idx is not None:
                matched[image_id].add(best_idx)
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        gt_count = gt_count_by_class.get(cls_id, 0)
        tp = int(sum(tp_flags))
        fp = int(sum(fp_flags))
        total_tp += tp
        total_fp += fp
        ap50 = ap_from_pr(np.array(tp_flags), np.array(fp_flags), gt_count)
        per_class.append({
            "class": cls_name,
            "gt": gt_count,
            "pred": len(preds),
            "tp50": tp,
            "fp50": fp,
            "precision50": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall50": tp / gt_count if gt_count else 0.0,
            "ap50": ap50,
        })

    valid_aps = [x["ap50"] for x in per_class if x["ap50"] is not None]
    return {
        "model": model_path,
        "dataset": dataset_dir,
        "images": len(gt_by_image),
        "gt": total_gt,
        "precision50": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0,
        "recall50": total_tp / total_gt if total_gt else 0.0,
        "map50": float(np.mean(valid_aps)) if valid_aps else 0.0,
        "classes": per_class,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = evaluate(args.model, args.dataset, args.conf, args.nms)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
