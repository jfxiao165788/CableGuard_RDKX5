import math
import time
from typing import List, Optional

from ad5933_driver import AD5933Driver
from calibration_manager import CalibrationManager
from config import (
    CAP_CAL_FILE,
    CAP_FALLBACK_MAX_CV,
    CAP_FREQ_HZ,
    CAP_MIN_VALID_RATIO,
    CAP_PHASE_MAX_DEG,
    CAP_PHASE_MIN_DEG,
    CAP_SAMPLE_COUNT,
    LOW_Z_CAL_FILE,
    LOW_Z_FREQ_HZ,
    LOW_Z_SAMPLE_COUNT,
    LOW_Z_THRESHOLD_OHM,
    LOW_Z_PHASE_THRESHOLD_DEG,
    MIN_VALID_RATIO,
    NORMAL_LENGTH_TOLERANCE_M,
    CapCal,
    LowZCal,
)
from measurement_models import AnalysisResult, RawMeasurement
from utils_filter import coefficient_of_variation, mad_filter, med, trimmed_mean


class CableAnalyzer:
    def __init__(self, driver: AD5933Driver):
        self.driver = driver
        self.cal_mgr = CalibrationManager(LOW_Z_CAL_FILE, CAP_CAL_FILE)
        self.low_cal = self.cal_mgr.load_low_z()
        self.cap_cal = self.cal_mgr.load_cap()

    @staticmethod
    def _wrap_phase_deg(phase_deg: float) -> float:
        return ((phase_deg + 180.0) % 360.0) - 180.0

    def _correct_phase(self, sample: RawMeasurement, system_phase_deg: float) -> float:
        return self._wrap_phase_deg(sample.phase_raw_deg - system_phase_deg)

    @staticmethod
    def _raw_to_impedance_ohm(sample: RawMeasurement, gain_factor: float) -> Optional[float]:
        if sample.mag <= 0 or gain_factor <= 0:
            return None
        return 1.0 / (gain_factor * sample.mag)

    def _measure_samples(self, freq_hz: int, count: int, delay_s: float = 0.05) -> List[RawMeasurement]:
        out = []
        for _ in range(count):
            out.append(self.driver.measure_raw_at_freq(freq_hz))
            time.sleep(delay_s)
        return out

    def calibrate_low_z(self, known_resistance_ohm: float, count: int = 15) -> LowZCal:
        if known_resistance_ohm <= 0:
            raise ValueError('标准电阻必须大于0')

        samples = self._measure_samples(LOW_Z_FREQ_HZ, count)
        mags = [s.mag for s in samples if s.mag > 0]
        if len(mags) < max(5, count // 2):
            raise RuntimeError('低阻校准失败：有效测量太少')

        avg_mag = trimmed_mean(mags, 0.2)
        gain_factor = 1.0 / (known_resistance_ohm * avg_mag)
        system_phase_deg = trimmed_mean([s.phase_raw_deg for s in samples], 0.2)

        cal = LowZCal(
            freq_hz=LOW_Z_FREQ_HZ,
            gain_factor=gain_factor,
            system_phase_deg=system_phase_deg,
            z_threshold_ohm=LOW_Z_THRESHOLD_OHM,
            phase_threshold_deg=LOW_Z_PHASE_THRESHOLD_DEG,
        )
        self.cal_mgr.save_low_z(cal)
        self.low_cal = cal
        return cal

    def calibrate_cap_mode_with_resistor(self, known_resistance_ohm: float, count: int = 20) -> CapCal:
        if known_resistance_ohm <= 0:
            raise ValueError('标准电阻必须大于0')

        samples = self._measure_samples(CAP_FREQ_HZ, count)
        mags = [s.mag for s in samples if s.mag > 0]
        if len(mags) < max(8, count // 2):
            raise RuntimeError('电容模式增益校准失败：有效测量太少')

        avg_mag = trimmed_mean(mags, 0.2)
        gain_factor = 1.0 / (known_resistance_ohm * avg_mag)
        system_phase_deg = trimmed_mean([s.phase_raw_deg for s in samples], 0.2)

        old = self.cap_cal
        cal = CapCal(
            freq_hz=CAP_FREQ_HZ,
            gain_factor=gain_factor,
            system_phase_deg=system_phase_deg,
            stray_cap_pf=old.stray_cap_pf if old else 0.0,
            cable_pf_per_m=old.cable_pf_per_m if old else 100.0,
            normal_net_cap_pf=old.normal_net_cap_pf if old else 0.0,
            normal_distance_m=old.normal_distance_m if old else 0.0,
            length_tolerance_m=old.length_tolerance_m if old else NORMAL_LENGTH_TOLERANCE_M,
            ref_temp_c=old.ref_temp_c if old else 25.0,
            temp_coeff_per_c=old.temp_coeff_per_c if old else 0.0,
        )
        self.cal_mgr.save_cap(cal)
        self.cap_cal = cal
        return cal

    def calibrate_stray_cap(self, count: int = 30) -> float:
        if self.cap_cal is None:
            raise RuntimeError('请先完成电容模式的标准电阻校准')
        # Follow the integrated moisture reference: stray capacitance is just
        # a capacitance measurement with the fixture connected and no cable.
        cap_pf = self.measure_capacitance_pf(count=count, reject_if_low_valid=False)
        if cap_pf is None:
            raise RuntimeError('杂散电容校准失败：有效测量太少或数据不稳定')
        self.cap_cal.stray_cap_pf = cap_pf
        self.cal_mgr.save_cap(self.cap_cal)
        return cap_pf

    def _measure_stray_capacitance_pf(self, count: int = 30) -> Optional[float]:
        samples = self._measure_samples(self.cap_cal.freq_hz, count)
        cap_vals = []
        for s in samples:
            z = self._raw_to_impedance_ohm(s, self.cap_cal.gain_factor)
            if z is None:
                continue
            ph = self._correct_phase(s, self.cap_cal.system_phase_deg)
            # Empty fixture stray capacitance should look capacitive. Near-zero
            # phase samples are open-circuit noise and must not be saved.
            if not (-120.0 < ph < -60.0):
                continue
            xc = abs(z * math.sin(math.radians(ph)))
            if xc <= 0:
                continue
            cap_vals.append(1e12 / (2 * math.pi * self.cap_cal.freq_hz * xc))

        min_valid = max(8, int(count * MIN_VALID_RATIO))
        if len(cap_vals) < min_valid:
            return None
        cap_vals = mad_filter(cap_vals)
        if len(cap_vals) < min_valid:
            return None
        return trimmed_mean(cap_vals, 0.2)

    def set_cable_profile(self, cable_pf_per_m: float, ref_temp_c: float = 25.0, temp_coeff_per_c: float = 0.0) -> None:
        if self.cap_cal is None:
            raise RuntimeError('请先完成电容模式校准')
        self.cap_cal.cable_pf_per_m = cable_pf_per_m
        self.cap_cal.ref_temp_c = ref_temp_c
        self.cap_cal.temp_coeff_per_c = temp_coeff_per_c
        self.cal_mgr.save_cap(self.cap_cal)

    def calibrate_normal_cable(self, count: int = CAP_SAMPLE_COUNT, length_tolerance_m: float = NORMAL_LENGTH_TOLERANCE_M) -> AnalysisResult:
        if self.cap_cal is None:
            return AnalysisResult(status='need_cap_cal', message='请先执行电容模式校准')
        if self.cap_cal.cable_pf_per_m <= 0:
            return AnalysisResult(status='invalid_profile', message='电缆单位电容必须大于0，请先设置电缆单位电容')
        if length_tolerance_m <= 0:
            length_tolerance_m = NORMAL_LENGTH_TOLERANCE_M

        result = self.locate_open_fault(count=count)
        if result.status != 'open_fault':
            return result

        net_cap_pf = result.capacitance_pf - self.cap_cal.stray_cap_pf
        self.cap_cal.normal_net_cap_pf = net_cap_pf
        self.cap_cal.normal_distance_m = result.distance_m
        self.cap_cal.length_tolerance_m = length_tolerance_m
        self.cal_mgr.save_cap(self.cap_cal)
        return AnalysisResult(
            status='normal_baseline_saved',
            message=f'正常电缆基准已保存：等效长度 {result.distance_m:.3f} m，容差 ±{length_tolerance_m:.3f} m',
            capacitance_pf=result.capacitance_pf,
            distance_m=result.distance_m,
            temperature_c=result.temperature_c,
        )

    def check_low_impedance(self) -> AnalysisResult:
        if self.low_cal is None:
            return AnalysisResult(status='need_low_cal', message='请先执行低阻模式校准')

        samples = self._measure_samples(self.low_cal.freq_hz, LOW_Z_SAMPLE_COUNT)
        z_vals = []
        ph_vals = []
        for s in samples:
            z = self._raw_to_impedance_ohm(s, self.low_cal.gain_factor)
            if z is None:
                continue
            ph = self._correct_phase(s, self.low_cal.system_phase_deg)
            z_vals.append(z)
            ph_vals.append(ph)

        if len(z_vals) < max(5, int(LOW_Z_SAMPLE_COUNT * MIN_VALID_RATIO)):
            return AnalysisResult(status='unstable', message='低阻模式有效数据不足', valid_count=len(z_vals), total_count=LOW_Z_SAMPLE_COUNT)

        z_vals = mad_filter(z_vals)
        ph_vals = mad_filter(ph_vals)
        z_med = med(z_vals)
        ph_med = med(ph_vals)
        z_cv = coefficient_of_variation(z_vals)

        if z_med < self.low_cal.z_threshold_ohm and abs(ph_med) < self.low_cal.phase_threshold_deg:
            return AnalysisResult(status='connected', message='检测到低阻导通', impedance_ohm=z_med, phase_deg=ph_med, valid_count=len(z_vals), total_count=LOW_Z_SAMPLE_COUNT)

        if z_cv > 0.25 and z_med < self.low_cal.z_threshold_ohm * 3.0:
            return AnalysisResult(status='unstable', message=f'低阻模式疑似接触不稳，CV={z_cv:.2f}', impedance_ohm=z_med, phase_deg=ph_med, valid_count=len(z_vals), total_count=LOW_Z_SAMPLE_COUNT)

        return AnalysisResult(status='not_low_z', message='未检测到低阻导通，进入开路容性判断', impedance_ohm=z_med, phase_deg=ph_med, valid_count=len(z_vals), total_count=LOW_Z_SAMPLE_COUNT)

    def measure_capacitance_pf(self, count: int = CAP_SAMPLE_COUNT, reject_if_low_valid: bool = True) -> Optional[float]:
        if self.cap_cal is None:
            raise RuntimeError('请先执行电容模式校准')

        samples = self._measure_samples(self.cap_cal.freq_hz, count)
        self._last_cap_quality = {
            'method': 'none',
            'valid_count': 0,
            'total_count': len(samples),
            'cv': None,
        }
        cap_vals = []
        fallback_vals = []
        for s in samples:
            z = self._raw_to_impedance_ohm(s, self.cap_cal.gain_factor)
            if z is None:
                continue
            ph = self._correct_phase(s, self.cap_cal.system_phase_deg)

            # Phase-qualified path from the reference implementation.
            if CAP_PHASE_MIN_DEG < ph < CAP_PHASE_MAX_DEG:
                # 对接近纯电容的场景，优先用容抗近似；若相位偏离较多，则取虚部对应容抗
                if ph < -70.0:
                    xc = z
                else:
                    xc = abs(z * math.sin(math.radians(ph)))

                if xc > 0:
                    cap_vals.append(1e12 / (2 * math.pi * self.cap_cal.freq_hz * xc))

            # If the phase gate rejects otherwise stable raw data, fall back to
            # the same capacitive magnitude approximation used near -90 deg.
            fallback_vals.append(1e12 / (2 * math.pi * self.cap_cal.freq_hz * z))

        min_valid = max(3, int(count * CAP_MIN_VALID_RATIO)) if reject_if_low_valid else 1

        cap_vals = mad_filter(cap_vals) if cap_vals else []
        if len(cap_vals) >= min_valid:
            self._last_cap_quality = {
                'method': 'phase',
                'valid_count': len(cap_vals),
                'total_count': len(samples),
                'cv': coefficient_of_variation(cap_vals),
            }
            return trimmed_mean(cap_vals, 0.2)

        fallback_vals = mad_filter(fallback_vals) if fallback_vals else []
        if not fallback_vals:
            return None

        fallback_cv = coefficient_of_variation(fallback_vals)
        if reject_if_low_valid and (len(fallback_vals) < min_valid or fallback_cv > CAP_FALLBACK_MAX_CV):
            self._last_cap_quality = {
                'method': 'magnitude_unstable',
                'valid_count': len(fallback_vals),
                'total_count': len(samples),
                'cv': fallback_cv,
            }
            return None

        self._last_cap_quality = {
            'method': 'magnitude',
            'valid_count': len(fallback_vals),
            'total_count': len(samples),
            'cv': fallback_cv,
        }
        return trimmed_mean(fallback_vals, 0.2)

    def locate_open_fault(self, count: int = CAP_SAMPLE_COUNT) -> AnalysisResult:
        if self.cap_cal is None:
            return AnalysisResult(status='need_cap_cal', message='请先执行电容模式校准')

        cap_pf = self.measure_capacitance_pf(count=count)
        if cap_pf is None:
            q = getattr(self, '_last_cap_quality', {})
            valid = q.get('valid_count')
            total = q.get('total_count')
            cv = q.get('cv')
            extra = ''
            if valid is not None and total is not None:
                extra = f'，有效样本 {valid}/{total}'
            if cv is not None:
                extra += f'，CV={cv:.2f}'
            return AnalysisResult(status='unstable', message=f'电容模式数据不足或波动过大，无法定位{extra}', valid_count=valid, total_count=total)

        temp_c = self.driver.get_temperature_c()
        cap_comp_pf = cap_pf * (1.0 + self.cap_cal.temp_coeff_per_c * (temp_c - self.cap_cal.ref_temp_c))
        net_cap_pf = cap_comp_pf - self.cap_cal.stray_cap_pf

        if net_cap_pf <= 0:
            return AnalysisResult(status='invalid_cap', message='扣除杂散电容后结果非正，无法定位', capacitance_pf=cap_comp_pf, temperature_c=temp_c)

        distance_m = net_cap_pf / self.cap_cal.cable_pf_per_m
        q = getattr(self, '_last_cap_quality', {})
        method = q.get('method')
        msg = '检测到开路型容性故障'
        if method == 'magnitude':
            msg = '检测到稳定电缆电容（相位校正未通过，已按阻抗幅值近似）'
        cv = q.get('cv')
        if cv is not None and cv > 0.20:
            msg += f'，样本波动较大(CV={cv:.2f})，结果为近似值'
        return AnalysisResult(
            status='open_fault',
            message=msg,
            capacitance_pf=cap_comp_pf,
            distance_m=distance_m,
            temperature_c=temp_c,
            valid_count=q.get('valid_count'),
            total_count=q.get('total_count'),
        )

    def analyze_cable(self) -> AnalysisResult:
        low_result = self.check_low_impedance()
        if low_result.status == 'connected':
            return low_result
        if low_result.status in ('need_low_cal', 'unstable'):
            return low_result

        cap_result = self.locate_open_fault()
        if cap_result.status != 'open_fault':
            return cap_result

        normal_distance = self.cap_cal.normal_distance_m if self.cap_cal else 0.0
        tolerance = self.cap_cal.length_tolerance_m if self.cap_cal else NORMAL_LENGTH_TOLERANCE_M
        if tolerance <= 0:
            tolerance = NORMAL_LENGTH_TOLERANCE_M

        if normal_distance > 0 and cap_result.distance_m is not None:
            delta_m = cap_result.distance_m - normal_distance
            if abs(delta_m) <= tolerance:
                return AnalysisResult(
                    status='normal',
                    message=f'电缆电容与正常基准一致，未检测到通断/断路故障（基准 {normal_distance:.3f} m，偏差 {delta_m:+.3f} m）',
                    capacitance_pf=cap_result.capacitance_pf,
                    distance_m=cap_result.distance_m,
                    temperature_c=cap_result.temperature_c,
                    valid_count=cap_result.valid_count,
                    total_count=cap_result.total_count,
                )
            if delta_m < -tolerance:
                return AnalysisResult(
                    status='open_fault',
                    message=f'检测到开路型容性故障：等效长度 {cap_result.distance_m:.3f} m，短于正常基准 {normal_distance:.3f} m；{cap_result.message}',
                    capacitance_pf=cap_result.capacitance_pf,
                    distance_m=cap_result.distance_m,
                    temperature_c=cap_result.temperature_c,
                    valid_count=cap_result.valid_count,
                    total_count=cap_result.total_count,
                )
            return AnalysisResult(
                status='capacitance_high',
                message=f'电容高于正常基准：等效长度 {cap_result.distance_m:.3f} m，正常基准 {normal_distance:.3f} m，请检查线缆型号、受潮或接线状态；{cap_result.message}',
                capacitance_pf=cap_result.capacitance_pf,
                distance_m=cap_result.distance_m,
                temperature_c=cap_result.temperature_c,
                valid_count=cap_result.valid_count,
                total_count=cap_result.total_count,
            )

        return AnalysisResult(
            status='open_location',
            message=f'检测到稳定容性电缆，但未建立正常电缆基准；当前只能给出等效开路位置，不能直接判为故障；{cap_result.message}',
            capacitance_pf=cap_result.capacitance_pf,
            distance_m=cap_result.distance_m,
            temperature_c=cap_result.temperature_c,
            valid_count=cap_result.valid_count,
            total_count=cap_result.total_count,
        )

    # ========== 受潮检测功能 ==========

    def measure_loss_factor_at_freq(self, freq_hz: int, count: int = 15) -> Optional[float]:
        """在指定频率测量损耗因子 tan(δ)

        Args:
            freq_hz: 测量频率
            count: 采样数量

        Returns:
            损耗因子，如果测量失败返回None
        """
        if self.cap_cal is None:
            return None

        samples = self._measure_samples(freq_hz, count)
        loss_factors = []

        for s in samples:
            z = self._raw_to_impedance_ohm(s, self.cap_cal.gain_factor)
            if z is None:
                continue
            ph = self._correct_phase(s, self.cap_cal.system_phase_deg)

            # 放宽相位范围，接受更广的容性相位。
            if not (CAP_PHASE_MIN_DEG < ph < CAP_PHASE_MAX_DEG):
                continue

            # 阻抗相位接近 -90 deg 时，损耗角 δ = 90 - |phase|。
            # 文档示例中 -91 deg / -89 deg 对应 tan(δ)≈0.017。
            tan_delta = abs(math.tan(math.radians(90.0 - abs(ph))))

            if tan_delta < 100:
                loss_factors.append(tan_delta)

        # 降低有效样本要求
        if len(loss_factors) < max(3, count // 3):
            return None

        loss_factors = mad_filter(loss_factors)
        return trimmed_mean(loss_factors, 0.2)

    def measure_multi_frequency_loss_factors(self) -> dict:
        """多频率测量损耗因子

        Returns:
            {频率: 损耗因子} 字典
        """
        frequencies = [5_000, 10_000, 30_000, 50_000]
        results = {}

        for freq in frequencies:
            loss_factor = self.measure_loss_factor_at_freq(freq)
            if loss_factor is not None:
                results[freq] = loss_factor

        return results

    def calibrate_moisture_baseline(self) -> dict:
        """建立干燥状态基准

        在电缆完全干燥时调用，记录各频率的损耗因子作为基准

        Returns:
            基准数据字典
        """
        if self.cap_cal is None:
            raise RuntimeError('请先完成电容模式校准')

        print('正在建立干燥基准，请稍候...')
        baseline = self.measure_multi_frequency_loss_factors()

        # 只要有至少1个频率成功即可
        if len(baseline) < 1:
            raise RuntimeError('基准测量失败，所有频率点都无效')

        # 保存基准到文件
        import json
        baseline_file = self.cal_mgr.cap_path.parent / 'moisture_baseline.json'
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)

        print(f'干燥基准已保存: {baseline_file}')
        print(f'成功测量 {len(baseline)} 个频率点')
        return baseline

    def load_moisture_baseline(self) -> Optional[dict]:
        """加载干燥基准数据"""
        baseline_file = self.cal_mgr.cap_path.parent / 'moisture_baseline.json'
        if not baseline_file.exists():
            return None

        import json
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
            if isinstance(baseline, dict) and 'frequencies' in baseline:
                return {
                    int(freq): float(data['loss_factor'])
                    for freq, data in baseline.get('frequencies', {}).items()
                    if isinstance(data, dict) and 'loss_factor' in data
                }
            # 转换键为整数
            return {int(k): v for k, v in baseline.items()}

    def detect_moisture(self) -> AnalysisResult:
        """检测电缆受潮

        基于多频率损耗因子分析，对比干燥基准判断受潮程度

        Returns:
            分析结果
        """
        if self.cap_cal is None:
            return AnalysisResult(status='need_cap_cal', message='请先执行电容模式校准')

        # 加载基准
        baseline = self.load_moisture_baseline()
        if baseline is None:
            return AnalysisResult(status='need_baseline', message='请先建立干燥基准（菜单选项10）')

        # 当前测量
        print('正在多频率测量...')
        current = self.measure_multi_frequency_loss_factors()

        if len(current) < 1:
            return AnalysisResult(status='unstable', message='多频率测量失败，所有频率点都无效')

        # 计算变化率。受潮应当表现为多频点损耗因子同步升高；
        # 单个低频点的相对变化容易被很小的基线值放大，不能直接判故障。
        changes = {}
        max_change = -999.0
        max_change_freq = 0
        reliable_rises = []
        severe_rises = []
        min_abs_delta = 0.05
        severe_abs_delta = 0.08
        light_ratio_threshold = 0.5
        severe_ratio_threshold = 1.0

        for freq in baseline:
            if freq in current:
                base = float(baseline[freq])
                now = float(current[freq])
                if base <= 0:
                    continue
                delta = now - base
                change_ratio = delta / base
                changes[freq] = {
                    'baseline_loss_factor': base,
                    'current_loss_factor': now,
                    'delta': delta,
                    'change_ratio': change_ratio,
                    'reliable_rise': (
                        change_ratio >= light_ratio_threshold
                        and delta >= min_abs_delta
                    ),
                    'severe_rise': (
                        change_ratio >= severe_ratio_threshold
                        and delta >= severe_abs_delta
                    ),
                }
                if change_ratio > max_change:
                    max_change = change_ratio
                    max_change_freq = freq
                if changes[freq]['reliable_rise']:
                    reliable_rises.append(freq)
                if changes[freq]['severe_rise']:
                    severe_rises.append(freq)

        if not changes:
            return AnalysisResult(status='unstable', message='无法对比基准数据，没有共同的频率点')

        if len(changes) < 2:
            return AnalysisResult(
                status='unstable',
                message='受潮检测共同频点不足，请重新建立干燥基线后复测',
            )

        # 判定受潮程度
        # 阈值：至少两个频点同时满足相对增幅和绝对增量，才判为受潮。
        if len(severe_rises) >= 2:
            status = 'severe_moisture'
            message = (
                f'检测到严重受潮！{len(severe_rises)} 个频点损耗因子明显升高，'
                f'最大变化在{max_change_freq/1000:.0f}kHz，增加{max_change*100:.1f}%'
            )
        elif len(reliable_rises) >= 2:
            status = 'moisture_detected'
            message = (
                f'检测到轻度受潮，{len(reliable_rises)} 个频点损耗因子升高，'
                f'最大变化在{max_change_freq/1000:.0f}kHz，增加{max_change*100:.1f}%'
            )
        else:
            status = 'dry'
            if reliable_rises:
                message = (
                    f'电缆状态正常。仅 {reliable_rises[0]/1000:.0f}kHz 单频点升高，'
                    '未形成多频一致受潮证据，建议保持接线稳定后复测。'
                )
            else:
                message = (
                    f'电缆状态正常，最大变化{max_change*100:.1f}%'
                    f'（需至少2个频点同时超过{light_ratio_threshold*100:.0f}%且绝对增加≥{min_abs_delta:.2f}）'
                )

        # 构建详细信息
        detail_lines = [message, '\n频率对比:']
        for freq in sorted(changes.keys()):
            item = changes[freq]
            marker = '采纳' if item['reliable_rise'] else '未采纳'
            detail_lines.append(
                f'  {freq/1000:.0f}kHz: 基准={item["baseline_loss_factor"]:.4f}, '
                f'当前={item["current_loss_factor"]:.4f}, '
                f'变化={item["change_ratio"]*100:+.1f}%, '
                f'绝对变化={item["delta"]:+.4f}（{marker}）'
            )

        return AnalysisResult(
            status=status,
            message='\n'.join(detail_lines),
            details={
                'changes': changes,
                'reliable_rise_freqs': reliable_rises,
                'severe_rise_freqs': severe_rises,
                'min_abs_delta': min_abs_delta,
                'severe_abs_delta': severe_abs_delta,
                'light_ratio_threshold': light_ratio_threshold,
                'severe_ratio_threshold': severe_ratio_threshold,
            },
        )
