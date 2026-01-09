# ml/schema.py
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd

# 모델이 사용할 feature 목록(순서 고정)
FEATURES: List[str] = [
    # Patient (LLM)
    "severity",
    "cond_trauma",
    "need_icu",
    "need_ventilator",
    "need_ct",
    "need_mri",
    "llm_confidence",

    # Hospital (Public API)
    "er_beds",
    "icu_beds",
    "trauma_icu_beds",
    "ct_available",
    "ventilator_available",

    # Distance / Transport
    "distance_km",
    "travel_time_min",

    # Location
    "same_district",
    "district_level",

    # Time Context
    "hour",
    "is_night",
    "is_weekend",

    # Rule-based filtering result
    "filter_level",
]

def payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    입력 JSON(payload)을 모델 입력용 DataFrame(1행)으로 변환한다.
    누락된 값은 에러를 내서(빨리 발견) 데이터 파이프라인을 안정적으로 만든다.
    """
    missing = [k for k in FEATURES if k not in payload]
    if missing:
        raise KeyError(f"Missing keys in payload: {missing}")

    row = {k: payload[k] for k in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)
