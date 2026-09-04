import math
from statistics import median
from typing import Iterable, List


def _clean(values: Iterable[float]) -> List[float]:
    out = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        out.append(float(v))
    return out


def med(values: Iterable[float]) -> float:
    vals = _clean(values)
    if not vals:
        raise ValueError('empty values')
    return float(median(vals))


def trimmed_mean(values: Iterable[float], trim_ratio: float = 0.2) -> float:
    vals = sorted(_clean(values))
    if not vals:
        raise ValueError('empty values')
    if len(vals) < 3:
        return sum(vals) / len(vals)
    k = int(len(vals) * trim_ratio)
    if 2 * k >= len(vals):
        k = max(0, len(vals) // 2 - 1)
    vals = vals[k: len(vals) - k] if k > 0 else vals
    return sum(vals) / len(vals)


def coefficient_of_variation(values: Iterable[float]) -> float:
    vals = _clean(values)
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    if mean == 0:
        return float('inf')
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var) / abs(mean)


def mad_filter(values: Iterable[float], thresh: float = 3.5) -> List[float]:
    vals = _clean(values)
    if len(vals) < 3:
        return vals
    m = median(vals)
    deviations = [abs(x - m) for x in vals]
    mad = median(deviations)
    if mad == 0:
        return vals
    filtered = []
    for x in vals:
        score = 0.6745 * abs(x - m) / mad
        if score <= thresh:
            filtered.append(x)
    return filtered if filtered else vals
