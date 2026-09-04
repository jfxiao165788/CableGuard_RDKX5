#!/usr/bin/env python3
"""
RDK X5 + AD5933 电缆检测系统（整合受潮检测功能）
"""
import traceback

from ad5933_driver import AD5933Driver
from cable_analyzer import CableAnalyzer


def print_raw_sample(driver: AD5933Driver, freq_hz: int):
    s = driver.measure_raw_at_freq(freq_hz)
    print(f"freq={s.freq_hz}Hz re={s.re} im={s.im} mag={s.mag:.2f} phase_raw={s.phase_raw_deg:.2f}°")


def print_result(r):
    print('\n===== 检测结果 =====')
    print(f'状态: {r.status}')
    print(f'说明: {r.message}')
    if r.impedance_ohm is not None:
        print(f'阻抗: {r.impedance_ohm:.2f} Ω')
    if r.phase_deg is not None:
        print(f'相位: {r.phase_deg:.2f} °')
    if r.capacitance_pf is not None:
        print(f'电容: {r.capacitance_pf:.2f} pF')
    if r.distance_m is not None:
        print(f'距离: {r.distance_m:.3f} m')
    if r.temperature_c is not None:
        print(f'温度: {r.temperature_c:.2f} °C')
    if r.valid_count is not None and r.total_count is not None:
        print(f'有效样本: {r.valid_count}/{r.total_count}')


def diagnose_capacitance(analyzer: CableAnalyzer):
    """诊断电容测量问题"""
    print('\n===== 电容测量诊断 =====')

    if analyzer.cap_cal is None:
        print('错误: 未进行电容模式校准')
        return

    print(f'测量频率: {analyzer.cap_cal.freq_hz} Hz')
    print(f'杂散电容: {analyzer.cap_cal.stray_cap_pf:.2f} pF')
    print(f'电缆单位电容: {analyzer.cap_cal.cable_pf_per_m:.2f} pF/m')
    print(f'\n正在测量当前电容...')

    # 测量原始数据
    samples = analyzer._measure_samples(analyzer.cap_cal.freq_hz, 10)

    print(f'\n原始测量数据 (共{len(samples)}个样本):')
    cap_list = []
    fallback_list = []
    for i, s in enumerate(samples):
        z = analyzer._raw_to_impedance_ohm(s, analyzer.cap_cal.gain_factor)
        ph = analyzer._correct_phase(s, analyzer.cap_cal.system_phase_deg)

        if z:
            import math
            fallback_pf = 1e12 / (2 * math.pi * analyzer.cap_cal.freq_hz * z)
            fallback_list.append(fallback_pf)
            if -140 < ph < -10:
                xc = z if ph < -70 else abs(z * math.sin(math.radians(ph)))
                c_pf = 1e12 / (2 * math.pi * analyzer.cap_cal.freq_hz * xc) if xc > 0 else 0
                cap_list.append(c_pf)
                print(f'  样本{i+1}: Z={z:.1f}Ω, 相位={ph:.1f}°, 电容={c_pf:.2f}pF')
            else:
                print(f'  样本{i+1}: Z={z:.1f}Ω, 相位={ph:.1f}°，幅值近似电容={fallback_pf:.2f}pF')
        else:
            print(f'  样本{i+1}: 无效样本')

    measured_cap = analyzer.measure_capacitance_pf(count=10, reject_if_low_valid=False)
    quality = getattr(analyzer, '_last_cap_quality', {})
    if measured_cap is not None:
        avg_cap = measured_cap
        net_cap = avg_cap - analyzer.cap_cal.stray_cap_pf
        print(f"\n采用算法: {quality.get('method', 'unknown')}, 有效样本: {quality.get('valid_count', 0)}/{quality.get('total_count', 0)}, CV={quality.get('cv', 0):.3f}")
        print(f'\n平均电容: {avg_cap:.2f} pF')
        print(f'扣除杂散: {net_cap:.2f} pF')
        if net_cap > 0:
            length = net_cap / analyzer.cap_cal.cable_pf_per_m
            print(f'计算长度: {length:.3f} m ({length*100:.1f} cm)')
        else:
            print('警告: 扣除杂散后为负值，可能原因:')
            print('  1. 杂散电容校准时有电缆连接')
            print('  2. 当前未连接电缆或电缆很短')
            print('  3. 测量频率不合适')
    else:
        print('\n错误: 没有有效的电容测量数据')
        print('可能原因:')
        print('  1. 未连接电缆或连接不良')
        print('  2. 电缆太短，电容太小')
        print('  3. 测量频率不合适')


def show_moisture_baseline(analyzer: CableAnalyzer):
    """显示当前的干燥基准"""
    baseline = analyzer.load_moisture_baseline()
    if baseline is None:
        print('\n尚未建立干燥基准')
        print('请先使用菜单选项10建立基准')
        return

    print('\n===== 当前干燥基准 =====')
    print('频率(kHz)  损耗因子tan(δ)')
    print('-' * 30)
    for freq in sorted(baseline.keys()):
        print(f'{freq/1000:8.0f}    {baseline[freq]:.6f}')
    print('-' * 30)


def show_normal_baseline(analyzer: CableAnalyzer):
    """显示当前正常电缆基准"""
    if analyzer.cap_cal is None:
        print('\n尚未完成电容模式校准')
        return

    print('\n===== 当前正常电缆基准 =====')
    print(f'电容测量频率: {analyzer.cap_cal.freq_hz} Hz')
    print(f'夹具杂散电容: {analyzer.cap_cal.stray_cap_pf:.2f} pF')
    print(f'单位长度电容: {analyzer.cap_cal.cable_pf_per_m:.2f} pF/m')
    if analyzer.cap_cal.normal_distance_m > 0:
        print(f'正常等效长度: {analyzer.cap_cal.normal_distance_m:.3f} m')
        print(f'正常净电容: {analyzer.cap_cal.normal_net_cap_pf:.2f} pF')
        print(f'长度容差: ±{analyzer.cap_cal.length_tolerance_m:.3f} m')
    else:
        print('尚未建立正常电缆基准')
        print('请在正常电缆接好时使用菜单选项13建立基准')


def main():
    driver = AD5933Driver()
    analyzer = CableAnalyzer(driver)

    menu = """
========================================
RDK X5 + AD5933 电缆检测系统
（整合受潮检测功能）
========================================
【基础校准】
1. 查看单次原始数据(30kHz)
2. 低阻模式校准(接标准电阻)
3. 电容模式校准(接标准电阻)
4. 杂散电容校准(夹具接好，但不接电缆)
5. 设置电缆单位电容(pF/m)

【故障检测】
6. 执行通断/断路综合检测
7. 只做低阻导通判断
8. 只做开路定位
9. 电容测量诊断(详细分析)

【受潮检测】⭐新功能
10. 建立干燥基准(在干燥状态下执行)
11. 执行受潮检测(对比基准判断)
12. 查看当前干燥基准

【正常电缆基准】
13. 建立正常电缆基准(正常电缆接好时执行)
14. 查看当前正常电缆基准

q. 退出
========================================
"""

    try:
        while True:
            print(menu)
            cmd = input('请输入命令: ').strip().lower()

            if cmd == '1':
                print_raw_sample(driver, 30_000)
            elif cmd == '2':
                r = float(input('请输入标准电阻值(Ω): ').strip())
                cal = analyzer.calibrate_low_z(r)
                print('低阻校准完成:')
                print(cal)
            elif cmd == '3':
                r = float(input('请输入标准电阻值(Ω): ').strip())
                cal = analyzer.calibrate_cap_mode_with_resistor(r)
                print('电容模式校准完成:')
                print(cal)
            elif cmd == '4':
                stray = analyzer.calibrate_stray_cap()
                print(f'杂散电容校准完成: stray_cap = {stray:.2f} pF')
            elif cmd == '5':
                c0 = float(input('请输入单位长度电容(pF/m): ').strip())
                analyzer.set_cable_profile(c0)
                print('电缆单位电容已保存')
            elif cmd == '6':
                print_result(analyzer.analyze_cable())
            elif cmd == '7':
                print_result(analyzer.check_low_impedance())
            elif cmd == '8':
                print_result(analyzer.locate_open_fault())
            elif cmd == '9':
                diagnose_capacitance(analyzer)
            elif cmd == '10':
                print('\n⚠️  请确保电缆处于完全干燥状态！')
                confirm = input('确认继续？(y/n): ').strip().lower()
                if confirm == 'y':
                    try:
                        baseline = analyzer.calibrate_moisture_baseline()
                        print('\n✓ 干燥基准建立成功！')
                        print('\n基准数据:')
                        for freq, loss in sorted(baseline.items()):
                            print(f'  {freq/1000:.0f}kHz: tan(δ) = {loss:.6f}')
                    except Exception as e:
                        print(f'\n✗ 基准建立失败: {e}')
                else:
                    print('已取消')
            elif cmd == '11':
                print('\n开始受潮检测...')
                result = analyzer.detect_moisture()
                print_result(result)
            elif cmd == '12':
                show_moisture_baseline(analyzer)
            elif cmd == '13':
                print('\n请确认当前接入的是正常电缆，且夹具连接稳定。')
                tol_text = input('请输入长度容差(m，默认0.05): ').strip()
                tol = float(tol_text) if tol_text else 0.05
                print_result(analyzer.calibrate_normal_cable(length_tolerance_m=tol))
            elif cmd == '14':
                show_normal_baseline(analyzer)
            elif cmd == 'q':
                break
            else:
                print('无效命令')
    except KeyboardInterrupt:
        print('\n用户中断')
    except Exception:
        traceback.print_exc()
    finally:
        driver.close()


if __name__ == '__main__':
    main()
