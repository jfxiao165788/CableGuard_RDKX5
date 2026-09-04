#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Tools -- OpenAI function calling compatible interface.

Loads tool definitions from ai_tools.json and dispatches calls.
"""

import json
import os

_CUR = os.path.dirname(os.path.abspath(__file__))
_SCHEMA_PATH = os.path.join(_CUR, "ai_tools.json")

with open(_SCHEMA_PATH, "r") as _f:
    OPENAI_TOOLS = json.load(_f)


def _ok(data=None, message=""):
    r = {"status": "ok"}
    if data is not None:
        r["data"] = data
    if message:
        r["message"] = message
    return r


def _err(message, code=400):
    return {"status": "error", "message": message, "code": code}


def _parse_detection(text):
    if not text:
        return {"has_detection": False, "detections": []}
    try:
        dets = json.loads(text)
        return {"has_detection": len(dets) > 0, "count": len(dets), "detections": dets}
    except Exception:
        return {"has_detection": False, "raw": str(text)[:200]}


def _parse_infrared(text):
    if not text:
        return {"has_anomaly": False}
    try:
        data = json.loads(text)
        return {"has_anomaly": True, "results": data}
    except Exception:
        return {"has_anomaly": True, "raw": str(text)[:200]}


def execute_tool(name, arguments, ctx):
    """Execute a tool call. ctx provides access to web_control_node runtime."""
    try:
        if name == "get_system_status":
            return _ok(ctx["get_status"]())

        elif name == "get_detection_status":
            n = ctx["node"]
            return _ok({
                "detection": _parse_detection(n.latest_detection_text if n else ""),
                "infrared": _parse_infrared(n.latest_infrared_text if n else ""),
                "has_camera_frame": bool(n and n.latest_jpeg),
            })

        elif name == "get_motor_status":
            n = ctx["node"]
            motion = ctx.get("parse_motor_motion", lambda t: {})(n.latest_motor_motion_text if n else "")
            return _ok({
                "state": n.motor_status if n else "unknown",
                "direction": n.motor_direction if n else "unknown",
                "motion": motion,
            })

        elif name == "get_fusion_status":
            return _ok(ctx["get_fusion_status"]())

        elif name == "get_logs":
            source = str(arguments.get("source") or "detection")
            lines = min(int(arguments.get("lines") or 50), 200)
            return _ok({"source": source, "lines": lines, "content": ctx["get_logs"](source, lines)})

        elif name == "motor_forward":
            return _ok(message=ctx["motor_action"]("forward"))
        elif name == "motor_reverse":
            return _ok(message=ctx["motor_action"]("reverse"))
        elif name == "motor_stop":
            return _ok(message=ctx["motor_action"]("stop"))
        elif name == "motor_move_to":
            pos = float(arguments.get("position_mm", 0))
            return _ok(message=ctx["motor_action"]("move_to", position_mm=pos))
        elif name == "motor_set_zero":
            return _ok(message=ctx["motor_action"]("zero_position"))

        elif name == "capture_false_positive":
            reason = str(arguments.get("reason") or "")
            if not reason or reason.startswith("请填写误检原因"):
                return _err("必须提供有效的误检原因(reason)")
            r = ctx["capture_fp"]({
                "dataset": str(arguments.get("dataset") or ""),
                "split": str(arguments.get("split") or "train"),
                "reason": reason,
                "mode": str(arguments.get("mode") or "生成空标签，作为负样本"),
                "pred": "未上报",
                "type": "误检 / False Positive",
            })
            if r.get("status") == "error":
                return _err(r.get("message", "FP入库失败"))
            return _ok(r)

        elif name == "list_false_positives":
            limit = int(arguments.get("limit") or 20)
            fp = ctx["list_fp"](limit)
            return _ok({"count": len(fp), "samples": fp})

        elif name == "get_datasets":
            return _ok({"count": len(ctx["list_datasets"]()), "datasets": ctx["list_datasets"]()})

        elif name == "get_models":
            return _ok({"count": len(ctx["list_models"]()), "models": ctx["list_models"](), "current": ctx.get("current_model", {})})

        elif name == "start_training":
            r = ctx["start_training"](
                str(arguments.get("dataset") or ""),
                int(arguments.get("epochs") or 80),
                int(arguments.get("imgsz") or 640),
                int(arguments.get("batch") or 8),
            )
            return _ok(r)

        elif name == "get_pipeline_status":
            with ctx["pipeline_lock"]:
                return _ok(dict(ctx["pipeline_state"]))

        elif name == "deploy_model":
            mn = str(arguments.get("model_name") or "")
            if not mn:
                return _err("必须提供model_name参数")
            r = ctx["deploy_model"](mn)
            if r.get("status") == "error":
                return _err(r.get("message", "部署失败"))
            return _ok(r)

        elif name == "apply_model":
            mn = str(arguments.get("model_name") or "")
            if not mn:
                return _err("必须提供model_name参数")
            r = ctx["apply_model"](mn)
            if r.get("status") == "error":
                return _err(r.get("message", "应用失败"))
            return _ok(r)

        else:
            return _err(f"未知工具: {name}。可用工具: GET /api/tools")

    except Exception as exc:
        return _err(f"工具执行异常: {exc}")
