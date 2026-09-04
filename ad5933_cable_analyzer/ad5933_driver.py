import math
import time
from typing import Tuple

try:
    import smbus2
except Exception:  # pragma: no cover
    smbus2 = None

from config import AD5933_ADDR, DEFAULT_RANGE_NAME, I2C_BUS, MCLK_HZ
from measurement_models import RawMeasurement

REG_CTRL_H = 0x80
REG_CTRL_L = 0x81
REG_FREQ_H = 0x82
REG_FREQ_M = 0x83
REG_FREQ_L = 0x84
REG_INC_H = 0x85
REG_INC_M = 0x86
REG_INC_L = 0x87
REG_POINTS_H = 0x88
REG_POINTS_L = 0x89
REG_SETTLE_H = 0x8A
REG_SETTLE_L = 0x8B
REG_STATUS = 0x8F
REG_TEMP_H = 0x92
REG_TEMP_L = 0x93
REG_RE_H = 0x94
REG_RE_L = 0x95
REG_IM_H = 0x96
REG_IM_L = 0x97


class AD5933Driver:
    """只负责芯片寄存器读写和原始复数测量。"""

    def __init__(self, bus_num: int = I2C_BUS, addr: int = AD5933_ADDR):
        if smbus2 is None:
            raise RuntimeError('smbus2 未安装，板端请先 pip install smbus2')
        self.bus_num = bus_num
        self.addr = addr
        self.bus = smbus2.SMBus(bus_num)
        self.range_name = DEFAULT_RANGE_NAME

    def close(self) -> None:
        try:
            self.bus.close()
        except Exception:
            pass

    def _wr(self, reg: int, val: int) -> None:
        self.bus.write_byte_data(self.addr, reg, val & 0xFF)
        time.sleep(0.001)

    def _rd(self, reg: int) -> int:
        self.bus.write_byte_data(self.addr, 0xB0, reg)
        time.sleep(0.001)
        return self.bus.read_byte(self.addr)

    @staticmethod
    def _freq_code(freq_hz: int) -> int:
        return int(freq_hz * 4.0 * 134217728.0 / MCLK_HZ)

    def configure_single_frequency(self, freq_hz: int, settling_cycles: int = 63) -> None:
        # 先进入待机模式并设置PGA增益×1和输出电压范围
        # bit[1:0]=01: PGA增益×1
        # bit[3:2]=00: 输出电压范围2Vp-p (默认)
        # bit[7:4]=1011: 待机模式
        self._wr(REG_CTRL_H, 0xB1)
        time.sleep(0.01)

        code = self._freq_code(freq_hz)
        self._wr(REG_FREQ_H, (code >> 16) & 0xFF)
        self._wr(REG_FREQ_M, (code >> 8) & 0xFF)
        self._wr(REG_FREQ_L, code & 0xFF)

        self._wr(REG_INC_H, 0)
        self._wr(REG_INC_M, 0)
        self._wr(REG_INC_L, 0)
        self._wr(REG_POINTS_H, 0)
        self._wr(REG_POINTS_L, 0)
        self._wr(REG_SETTLE_H, (settling_cycles >> 8) & 0xFF)
        self._wr(REG_SETTLE_L, settling_cycles & 0xFF)

        # 待机 -> 初始化起始频率 -> 开始扫频
        # PGA增益×1: bit[1:0]=01, 其他位保持命令功能
        self._wr(REG_CTRL_H, 0xB1)  # 1011 0001: 待机 + 增益×1
        self._wr(REG_CTRL_L, 0x00)
        self._wr(REG_CTRL_L, 0x10)  # 初始化起始频率
        time.sleep(0.1)
        self._wr(REG_CTRL_L, 0x00)
        self._wr(REG_CTRL_H, 0x11)  # 0001 0001: 初始化 + 增益×1
        time.sleep(0.03)
        self._wr(REG_CTRL_H, 0x21)  # 0010 0001: 开始扫频 + 增益×1

    def trigger_measurement(self) -> None:
        self._wr(REG_CTRL_H, 0x41)  # 保留增益×1

    def read_raw_complex(self, max_wait_s: float = 0.5) -> Tuple[int, int]:
        t0 = time.time()
        while time.time() - t0 < max_wait_s:
            if self._rd(REG_STATUS) & 0x02:
                break
            time.sleep(0.005)

        rh, rl = self._rd(REG_RE_H), self._rd(REG_RE_L)
        ih, il = self._rd(REG_IM_H), self._rd(REG_IM_L)

        re_raw = (rh << 8) | rl
        im_raw = (ih << 8) | il
        re = re_raw - 0x10000 if re_raw >= 0x8000 else re_raw
        im = im_raw - 0x10000 if im_raw >= 0x8000 else im_raw
        return re, im

    def measure_raw_at_freq(self, freq_hz: int) -> RawMeasurement:
        self.configure_single_frequency(freq_hz)
        self.trigger_measurement()
        re, im = self.read_raw_complex()
        mag = math.hypot(re, im)
        phase_raw_deg = math.degrees(math.atan2(im, re))
        return RawMeasurement(freq_hz=freq_hz, re=re, im=im, mag=mag, phase_raw_deg=phase_raw_deg)

    def get_temperature_c(self) -> float:
        th = self._rd(REG_TEMP_H)
        tl = self._rd(REG_TEMP_L)
        raw = (th << 8) | tl
        if raw & 0x2000:
            raw -= 0x4000
        return raw / 32.0
