from datetime import datetime

SEVERITY_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def to_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except:
        return default


def to_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except:
        return default


def safe_ratio(num, denom, default=0.0):
    try:
        num = float(num)
        denom = float(denom)
        if denom <= 0:
            return default
        return num / denom
    except:
        return default


def build_ml_features(hospital_row, patient_info: dict) -> dict:
    now = datetime.now()
    hour = now.hour
    is_night = 1 if (hour >= 22 or hour <= 6) else 0
    is_weekend = 1 if now.weekday() >= 5 else 0

    # =========================
    # Severity 변환
    # =========================
    sev = patient_info.get("severity")
    if isinstance(sev, str):
        severity_val = SEVERITY_MAP.get(sev.upper(), 1)
    else:
        severity_val = to_int(sev, default=1)

    # =========================
    # 실시간 병상 수
    # =========================
    er_beds = to_int(hospital_row.get("hvec", 0))
    icu_beds = to_int(hospital_row.get("hvicc", 0))

    # =========================
    # 전체 병상 수 (정적)
    # =========================
    total_er_beds = to_int(hospital_row.get("total_hvec", 0))
    total_icu_beds = to_int(hospital_row.get("total_hvicc", 0))
    total_beds = to_int(hospital_row.get("total_hpbdn", 0))

    features = {
        # =========================
        # Hospital Identity
        # =========================
        "hospital_name": hospital_row.get("dutyname"),
        "hospital_id": hospital_row.get("hpid"),
        "hospital_phone": hospital_row.get("dutytel3"),

        # =========================
        # Patient (LLM-derived)
        # =========================
        "severity": severity_val,
        "cond_trauma": 1 if patient_info.get("suspected_condition") == "TRAUMA" else 0,
        "need_icu": to_int(patient_info.get("required_resources", {}).get("need_icu", 0)),
        "need_ventilator": to_int(patient_info.get("required_resources", {}).get("need_ventilator", 0)),
        "need_ct": to_int(patient_info.get("required_resources", {}).get("need_ct", 0)),
        "need_mri": to_int(patient_info.get("required_resources", {}).get("need_mri", 0)),
        "llm_confidence": to_float(patient_info.get("confidence", 0.0)),

        # =========================
        # Hospital (Realtime)
        # =========================
        "er_beds": er_beds,
        "icu_beds": icu_beds,
        "trauma_icu_beds": to_int(hospital_row.get("hv9", 0)),
        "ct_available": 1 if hospital_row.get("hvctayn") == "Y" else 0,
        "ventilator_available": 1 if hospital_row.get("hvventiayn") == "Y" else 0,

        # =========================
        # Hospital (Static - TOTAL)
        # =========================
        "total_er_beds": total_er_beds,
        "total_icu_beds": total_icu_beds,
        "total_beds": total_beds,

        # =========================
        # Ratio Features
        # =========================
        "er_bed_ratio": safe_ratio(er_beds, total_er_beds),
        "icu_bed_ratio": safe_ratio(icu_beds, total_icu_beds),

        # =========================
        # Distance / Transport
        # =========================
        "distance_km": to_float(hospital_row.get("distance_km", 0.0)),
        "travel_time_min": to_float(hospital_row.get("estimated_travel_time_min", 0.0)),

        # =========================
        # Location
        # =========================
        "same_district": to_int(hospital_row.get("same_district", 0)),
        "district_level": to_int(hospital_row.get("district_level", 2)),

        # =========================
        # Time Context
        # =========================
        "hour": hour,
        "is_night": is_night,
        "is_weekend": is_weekend,

        # =========================
        # Filtering
        # =========================
        "filter_level": to_int(hospital_row.get("filter_level", 3)),
    }

    return features