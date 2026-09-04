#!/usr/bin/env python3
import argparse
import json
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path

from ad5933_driver import AD5933Driver
from cable_analyzer import CableAnalyzer
from calibration_manager import CalibrationManager
from config import CAP_CAL_FILE, CAP_FREQ_HZ, LOW_Z_CAL_FILE, LOW_Z_FREQ_HZ

MOISTURE_BASELINE_FILE = Path(CAP_CAL_FILE).parent / "moisture_baseline.json"
MOISTURE_FREQ_HZ_LIST = [5_000, 10_000, 30_000, 50_000]
DEFAULT_LENGTH_TOLERANCE_M = 0.05


def _to_jsonable(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _ok(action, data=None, message="ok", cable_length_m=None, length_tolerance_m=None):
    payload = _to_jsonable(data or {})
    if isinstance(payload, dict):
        payload = _with_readable_output(
            action,
            payload,
            cable_length_m=cable_length_m,
            length_tolerance_m=length_tolerance_m,
        )
    return {"ok": True, "action": action, "message": message, "data": payload}


def _format_num(value, digits=2, suffix=""):
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return f"{value}{suffix}"


def _format_distance(value_m):
    if value_m is None:
        return "本次数据无法定位"
    return f"{float(value_m):.3f} m（约 {float(value_m) * 100:.1f} cm）"


def _status_ok(status):
    return status not in {
        "need_low_cal",
        "need_cap_cal",
        "need_baseline",
        "unstable",
        "invalid_cap",
        "error",
    }


def _first_line(action, data, cable_length_m=None, length_tolerance_m=None):
    status = data.get("status")
    distance_m = data.get("distance_m")
    tolerance = length_tolerance_m or DEFAULT_LENGTH_TOLERANCE_M

    if action == "analyze" and data.get("moisture_status") in ("moisture_detected", "severe_moisture"):
        return "综合检测发现受潮风险"

    if action == "analyze" and data.get("moisture_status") == "dry" and status in ("normal", "dry"):
        return "综合检测未发现明显异常"

    if status == "normal":
        return "未检测到通断/断路故障"

    if status == "open_location" and distance_m is not None:
        return f"检测到稳定电缆电容，等效开路位置{_format_distance(distance_m)}"

    if status == "capacitance_high" and distance_m is not None:
        return f"电容高于正常基准，等效长度{_format_distance(distance_m)}"

    if status == "normal_baseline_saved":
        return "正常电缆基准已保存"

    if status == "open_fault" and distance_m is not None:
        if cable_length_m is not None and abs(float(distance_m) - float(cable_length_m)) <= tolerance:
            return "未检测到故障"
        return f"可能出现断路故障，距离{_format_distance(distance_m)}"

    if status in ("moisture_detected", "severe_moisture"):
        return f"可能出现受潮故障，距离{_format_distance(distance_m)}"

    if status == "connected":
        return f"可能出现短路故障，距离{_format_distance(distance_m)}"

    if status == "dry":
        return "未检测到受潮故障"

    if status == "not_low_z":
        return "未检测到低阻短路"

    if status in ("need_low_cal", "need_cap_cal", "need_baseline"):
        return "检测失败：请先完成对应校准"

    if status in ("unstable", "invalid_cap"):
        return "检测失败：本次数据不稳定或无效"

    if action in ("calibrate_low", "calibrate_cap", "calibrate_stray", "calibrate_normal", "calibrate_moisture_baseline", "set_profile"):
        return "校准完成"

    return "检测完成"


def _data_lines(data):
    lines = []
    if data.get("freq_hz") is not None:
        lines.append(f"频率：{data.get('freq_hz')} Hz")
    if data.get("impedance_ohm") is not None:
        lines.append(f"阻抗：{_format_num(data.get('impedance_ohm'), 2, ' Ω')}")
    if data.get("phase_deg") is not None:
        lines.append(f"相位：{_format_num(data.get('phase_deg'), 2, '°')}")
    if data.get("capacitance_pf") is not None:
        lines.append(f"电容：{_format_num(data.get('capacitance_pf'), 2, ' pF')}")
    if data.get("cap_method") is not None:
        lines.append(f"电容算法：{data.get('cap_method')}")
    if data.get("cap_cv") is not None:
        lines.append(f"电容样本CV：{_format_num(data.get('cap_cv'), 3)}")
    if data.get("distance_m") is not None:
        lines.append(f"距离：{_format_distance(data.get('distance_m'))}")
    if data.get("temperature_c") is not None:
        lines.append(f"芯片温度：{_format_num(data.get('temperature_c'), 2, ' °C')}")
    if data.get("valid_count") is not None and data.get("total_count") is not None:
        lines.append(f"有效数据：{data.get('valid_count')}/{data.get('total_count')} 组")
    if data.get("stray_cap_pf") is not None:
        lines.append(f"夹具杂散电容：{_format_num(data.get('stray_cap_pf'), 2, ' pF')}")
    if data.get("cable_pf_per_m") is not None:
        lines.append(f"单位长度电容：{_format_num(data.get('cable_pf_per_m'), 2, ' pF/m')}")
    if data.get("normal_net_cap_pf") is not None:
        lines.append(f"正常净电容：{_format_num(data.get('normal_net_cap_pf'), 2, ' pF')}")
    if data.get("normal_distance_m") is not None:
        lines.append(f"正常等效长度：{_format_distance(data.get('normal_distance_m'))}")
    if data.get("length_tolerance_m") is not None:
        lines.append(f"长度容差：±{_format_num(data.get('length_tolerance_m'), 3, ' m')}")
    if data.get("re") is not None and data.get("im") is not None:
        lines.append(f"原始实部/虚部：{data.get('re')} / {data.get('im')}")
    if data.get("mag") is not None:
        lines.append(f"原始幅值：{_format_num(data.get('mag'), 2)}")
    if data.get("phase_raw_deg") is not None:
        lines.append(f"原始相位：{_format_num(data.get('phase_raw_deg'), 2, '°')}")
    if data.get("moisture_freq_hz") is not None:
        lines.append(f"受潮最敏感频率：{int(data.get('moisture_freq_hz'))} Hz")
    if data.get("moisture_change_ratio") is not None:
        lines.append(f"损耗因子最大变化：{float(data.get('moisture_change_ratio')) * 100:.1f}%")
    return lines


def _moisture_detail_lines(data):
    details = data.get("details")
    if not isinstance(details, dict):
        return []
    changes = details.get("changes")
    if not isinstance(changes, dict):
        return []
    lines = ["受潮对比："]
    for freq_text, item in sorted(changes.items(), key=lambda kv: int(kv[0])):
        try:
            freq_khz = int(freq_text) / 1000
            base = float(item.get("baseline_loss_factor"))
            current = float(item.get("current_loss_factor"))
            ratio = float(item.get("change_ratio")) * 100
        except Exception:
            continue
        lines.append(f"{freq_khz:.0f} kHz：干燥基线 {base:.4f}，当前 {current:.4f}，变化 {ratio:+.1f}%")
    return lines


def _with_readable_output(action, data, cable_length_m=None, length_tolerance_m=None):
    first = _first_line(action, data, cable_length_m, length_tolerance_m)
    ok_text = "检测成功" if _status_ok(data.get("status")) else "检测失败"
    if action.startswith("calibrate") or action == "set_profile":
        ok_text = "检测成功" if data else "检测失败"

    lines = [
        first,
        f"检测状态：{ok_text}",
        "检测数据：",
    ]
    data_lines = _data_lines(data)
    lines.extend(data_lines or ["暂无可显示的数值数据"])
    lines.append(f"检测结果：{data.get('message') or first}")
    lines.extend(_moisture_detail_lines(data))

    data["first_line"] = first
    data["readable_status"] = ok_text
    data["friendly_text"] = "\n".join(lines)
    return data


def _load_moisture_baseline():
    if not MOISTURE_BASELINE_FILE.exists():
        return None
    return json.loads(MOISTURE_BASELINE_FILE.read_text(encoding="utf-8"))


def _analyze_with_moisture(analyzer):
    electrical = analyzer.analyze_cable()
    data = _to_jsonable(electrical)
    data["electrical_result"] = _to_jsonable(electrical)

    try:
        moisture = analyzer.detect_moisture()
        moisture_data = _to_jsonable(moisture)
    except Exception as exc:
        moisture_data = {
            "status": "error",
            "message": f"受潮检测执行失败：{exc}",
        }

    moisture_status = moisture_data.get("status")
    moisture_message = moisture_data.get("message") or ""
    data["moisture_result"] = moisture_data
    data["moisture_status"] = moisture_status
    data["moisture_message"] = moisture_message

    electrical_message = data.get("message") or ""
    if moisture_status in ("moisture_detected", "severe_moisture"):
        if data.get("status") in ("normal", "open_location", "not_low_z", "dry"):
            data["status"] = moisture_status
        data["message"] = (
            f"通断/断路检测：{electrical_message}\n\n"
            f"受潮检测：{moisture_message}"
        ).strip()
        if moisture_data.get("details"):
            data["details"] = moisture_data.get("details")
    elif moisture_status == "dry":
        data["message"] = (
            f"通断/断路检测：{electrical_message}\n\n"
            f"受潮检测：{moisture_message}"
        ).strip()
        if moisture_data.get("details"):
            data["details"] = moisture_data.get("details")
    elif moisture_status == "need_baseline":
        data["message"] = (
            f"通断/断路检测：{electrical_message}\n\n"
            f"受潮检测未执行：{moisture_message}"
        ).strip()
    elif moisture_status:
        data["message"] = (
            f"通断/断路检测：{electrical_message}\n\n"
            f"受潮检测状态：{moisture_message or moisture_status}"
        ).strip()

    return data


def _run(action, args):
    if action == "status":
        cal_mgr = CalibrationManager(LOW_Z_CAL_FILE, CAP_CAL_FILE)
        low_cal = cal_mgr.load_low_z()
        cap_cal = cal_mgr.load_cap()
        return _ok(
            action,
            {
                "low_calibrated": low_cal is not None,
                "cap_calibrated": cap_cal is not None,
                "normal_baseline_saved": bool(cap_cal and cap_cal.normal_distance_m > 0),
                "moisture_baseline_saved": MOISTURE_BASELINE_FILE.exists(),
                "moisture_baseline": _load_moisture_baseline(),
                "low_cal": low_cal,
                "cap_cal": cap_cal,
                "low_freq_hz": LOW_Z_FREQ_HZ,
                "cap_freq_hz": CAP_FREQ_HZ,
                "moisture_freq_hz_list": MOISTURE_FREQ_HZ_LIST,
            },
            cable_length_m=args.cable_length_m,
            length_tolerance_m=args.length_tolerance_m,
        )

    driver = AD5933Driver()
    analyzer = CableAnalyzer(driver)
    try:
        if action == "raw":
            freq = args.freq or LOW_Z_FREQ_HZ
            return _ok(action, driver.measure_raw_at_freq(freq), cable_length_m=args.cable_length_m, length_tolerance_m=args.length_tolerance_m)

        if action == "calibrate_low":
            return _ok(action, analyzer.calibrate_low_z(args.resistance), "低阻校准已保存", args.cable_length_m, args.length_tolerance_m)

        if action == "calibrate_cap":
            return _ok(action, analyzer.calibrate_cap_mode_with_resistor(args.resistance), "电容校准已保存", args.cable_length_m, args.length_tolerance_m)

        if action == "calibrate_stray":
            return _ok(action, {"stray_cap_pf": analyzer.calibrate_stray_cap()}, "夹具校准已保存", args.cable_length_m, args.length_tolerance_m)

        if action == "calibrate_normal":
            result = analyzer.calibrate_normal_cable(length_tolerance_m=args.length_tolerance_m)
            data = _to_jsonable(result)
            if analyzer.cap_cal is not None:
                data.update({
                    "normal_net_cap_pf": analyzer.cap_cal.normal_net_cap_pf,
                    "normal_distance_m": analyzer.cap_cal.normal_distance_m,
                    "length_tolerance_m": analyzer.cap_cal.length_tolerance_m,
                })
            return _ok(action, data, "正常电缆基准已保存", args.cable_length_m, args.length_tolerance_m)

        if action == "set_profile":
            analyzer.set_cable_profile(args.pf_per_m, args.ref_temp, args.temp_coeff)
            return _ok(action, {"cable_pf_per_m": args.pf_per_m, "ref_temp_c": args.ref_temp, "temp_coeff_per_c": args.temp_coeff}, "电缆参数已保存", args.cable_length_m, args.length_tolerance_m)

        if action == "analyze":
            return _ok(action, _analyze_with_moisture(analyzer), cable_length_m=args.cable_length_m, length_tolerance_m=args.length_tolerance_m)

        if action == "check_low":
            return _ok(action, analyzer.check_low_impedance(), cable_length_m=args.cable_length_m, length_tolerance_m=args.length_tolerance_m)

        if action == "locate_open":
            return _ok(action, analyzer.locate_open_fault(), cable_length_m=args.cable_length_m, length_tolerance_m=args.length_tolerance_m)

        if action == "cap_diagnose":
            if analyzer.cap_cal is None:
                raise RuntimeError("请先执行电容模式校准")
            samples = analyzer._measure_samples(analyzer.cap_cal.freq_hz, args.count)
            rows = []
            for sample in samples:
                z = analyzer._raw_to_impedance_ohm(sample, analyzer.cap_cal.gain_factor)
                ph = analyzer._correct_phase(sample, analyzer.cap_cal.system_phase_deg) if z else None
                rows.append({"sample": sample, "impedance_ohm": z, "phase_deg": ph})
            cap_pf = analyzer.measure_capacitance_pf(count=args.count, reject_if_low_valid=False)
            cap_quality = getattr(analyzer, "_last_cap_quality", {})
            return _ok(
                action,
                {
                    "freq_hz": analyzer.cap_cal.freq_hz,
                    "stray_cap_pf": analyzer.cap_cal.stray_cap_pf,
                    "cable_pf_per_m": analyzer.cap_cal.cable_pf_per_m,
                    "capacitance_pf": cap_pf,
                    "cap_method": cap_quality.get("method"),
                    "cap_cv": cap_quality.get("cv"),
                    "valid_count": cap_quality.get("valid_count"),
                    "total_count": cap_quality.get("total_count"),
                    "samples": rows,
                },
                cable_length_m=args.cable_length_m,
                length_tolerance_m=args.length_tolerance_m,
            )

        if action == "calibrate_moisture_baseline":
            return _ok(action, analyzer.calibrate_moisture_baseline(), "干燥基线已保存", args.cable_length_m, args.length_tolerance_m)

        if action == "detect_moisture":
            return _ok(action, analyzer.detect_moisture(), cable_length_m=args.cable_length_m, length_tolerance_m=args.length_tolerance_m)

        if action == "moisture_diagnose":
            return _ok(
                action,
                {str(freq): {"freq_hz": freq, "loss_factor": loss} for freq, loss in analyzer.measure_multi_frequency_loss_factors().items()},
                cable_length_m=args.cable_length_m,
                length_tolerance_m=args.length_tolerance_m,
            )

        raise ValueError(f"unknown action: {action}")
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description="AD5933 cable analyzer web runner")
    parser.add_argument("action")
    parser.add_argument("--freq", type=int)
    parser.add_argument("--resistance", type=float)
    parser.add_argument("--pf-per-m", type=float, default=100.0)
    parser.add_argument("--ref-temp", type=float, default=25.0)
    parser.add_argument("--temp-coeff", type=float, default=0.0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--cable-length-m", type=float)
    parser.add_argument("--length-tolerance-m", type=float, default=DEFAULT_LENGTH_TOLERANCE_M)
    args = parser.parse_args()

    try:
        result = _run(args.action, args)
    except Exception as exc:
        result = {
            "ok": False,
            "action": args.action,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(result, ensure_ascii=False), file=sys.stdout)
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
