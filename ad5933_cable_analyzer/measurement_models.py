from dataclasses import dataclass
from typing import Optional


@dataclass
class RawMeasurement:
    freq_hz: int
    re: int
    im: int
    mag: float
    phase_raw_deg: float


@dataclass
class AnalysisResult:
    status: str
    message: str = ''
    impedance_ohm: Optional[float] = None
    phase_deg: Optional[float] = None
    capacitance_pf: Optional[float] = None
    distance_m: Optional[float] = None
    temperature_c: Optional[float] = None
    valid_count: Optional[int] = None
    total_count: Optional[int] = None
    details: Optional[dict] = None
