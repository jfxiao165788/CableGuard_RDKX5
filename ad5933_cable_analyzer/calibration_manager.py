import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from config import CapCal, LowZCal


class CalibrationManager:
    def __init__(self, low_path: Path, cap_path: Path):
        self.low_path = Path(low_path)
        self.cap_path = Path(cap_path)

    def save_low_z(self, cal: LowZCal) -> None:
        self.low_path.write_text(json.dumps(asdict(cal), ensure_ascii=False, indent=2), encoding='utf-8')

    def save_cap(self, cal: CapCal) -> None:
        self.cap_path.write_text(json.dumps(asdict(cal), ensure_ascii=False, indent=2), encoding='utf-8')

    def load_low_z(self) -> Optional[LowZCal]:
        if not self.low_path.exists():
            return None
        data = json.loads(self.low_path.read_text(encoding='utf-8'))
        return LowZCal(**data)

    def load_cap(self) -> Optional[CapCal]:
        if not self.cap_path.exists():
            return None
        data = json.loads(self.cap_path.read_text(encoding='utf-8'))
        return CapCal(**data)
