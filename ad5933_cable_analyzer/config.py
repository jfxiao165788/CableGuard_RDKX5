from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

# RDK X5 + AD5933 基本配置
I2C_BUS = 5
AD5933_ADDR = 0x0D
MCLK_HZ = 16_776_000

# 两种工作模式
LOW_Z_FREQ_HZ = 30_000      # 低阻连通判断
CAP_FREQ_HZ = 10_000        # 开路电容定位（提高频率增加灵敏度）

# 采样数量
LOW_Z_SAMPLE_COUNT = 15
CAP_SAMPLE_COUNT = 30

# 判定阈值（先给经验值，后续根据你实测再调）
LOW_Z_THRESHOLD_OHM = 2_000.0
LOW_Z_PHASE_THRESHOLD_DEG = 20.0
CAP_PHASE_MIN_DEG = -140.0
CAP_PHASE_MAX_DEG = -10.0
MIN_VALID_RATIO = 0.70
CAP_MIN_VALID_RATIO = 0.33
CAP_FALLBACK_MAX_CV = 1.00
NORMAL_LENGTH_TOLERANCE_M = 0.05

# 默认量程标签，仅用于你现阶段兼容旧工程的字段命名
DEFAULT_RANGE_NAME = '10k'

# 校准文件
LOW_Z_CAL_FILE = DATA_DIR / 'cal_low.json'
CAP_CAL_FILE = DATA_DIR / 'cal_cap.json'
CABLE_PROFILE_FILE = DATA_DIR / 'cable_profile.json'


@dataclass
class LowZCal:
    freq_hz: int
    gain_factor: float
    system_phase_deg: float
    z_threshold_ohm: float = LOW_Z_THRESHOLD_OHM
    phase_threshold_deg: float = LOW_Z_PHASE_THRESHOLD_DEG


@dataclass
class CapCal:
    freq_hz: int
    gain_factor: float
    system_phase_deg: float
    stray_cap_pf: float = 0.0
    cable_pf_per_m: float = 100.0
    normal_net_cap_pf: float = 0.0
    normal_distance_m: float = 0.0
    length_tolerance_m: float = NORMAL_LENGTH_TOLERANCE_M
    ref_temp_c: float = 25.0
    temp_coeff_per_c: float = 0.0
