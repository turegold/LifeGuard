from datetime import datetime


def build_ml_features(
    hospital_row,
    patient_info: dict
) -> dict:

    now = datetime.now()
    hour = now.hour
    is_night = 1 if (hour >= 22 or hour <= 6) else 0
    is_weekend = 1 if now.weekday() >= 5 else 0

    features = {
        # Patient (LLM-derived)
        "severity": patient_info.get("severity"),
        "cond_trauma": 1 if patient_info.get("suspected_condition") == "TRAUMA" else 0,
        "need_icu": int(patient_info["required_resources"]["need_icu"]),
        "need_ventilator": int(patient_info["required_resources"]["need_ventilator"]),
        "need_ct": int(patient_info["required_resources"]["need_ct"]),
        "need_mri": int(patient_info["required_resources"]["need_mri"]),
        "llm_confidence": patient_info.get("confidence", 0.0),

        # Hospital (Public API)
        "er_beds": hospital_row.get("hvec", 0),
        "icu_beds": hospital_row.get("hvicc", 0),
        "trauma_icu_beds": hospital_row.get("hv9", 0),
        "ct_available": 1 if hospital_row.get("hvctayn") == "Y" else 0,
        "ventilator_available": 1 if hospital_row.get("hvventiayn") == "Y" else 0,


        # Distance / Transport
        "distance_km": hospital_row.get("distance_km"),
        "travel_time_min": hospital_row.get("estimated_travel_time_min"),


        # Location (Administrative)
        "same_district": hospital_row.get("same_district"),
        "district_level": hospital_row.get("district_level"),


        # Time Context
        "hour": hour,
        "is_night": is_night,
        "is_weekend": is_weekend,


        # Filtering
        "filter_level": hospital_row.get("filter_level"),
    }

    return features