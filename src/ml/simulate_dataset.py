# ml/simulate_dataset.py
import os
import random
import numpy as np
import pandas as pd

from ml.schema import FEATURES

SEED = 42

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def compute_filter_level(p: dict) -> int:


    # ---- Level 0: 모든 요구조건 완전 충족 ----
    if p["need_ct"] == 1 and p["ct_available"] == 0:
        pass
    elif p["need_ventilator"] == 1 and p["ventilator_available"] == 0:
        pass
    elif p["need_icu"] == 1 and p["icu_beds"] <= 0:
        pass
    elif p["need_icu"] == 1 and p["cond_trauma"] == 1 and p["trauma_icu_beds"] <= 0:
        pass
    else:
        return 0

    # ---- Level 1: 외상 전용 ICU 조건 완화 (일반 ICU 있으면 ok) ----
    # CT/ventilator 조건은 여전히 현실적으로 중요하니 유지
    if p["need_ct"] == 1 and p["ct_available"] == 0:
        pass
    elif p["need_ventilator"] == 1 and p["ventilator_available"] == 0:
        pass
    elif p["need_icu"] == 1 and p["icu_beds"] <= 0:
        pass
    else:
        return 1

    # ---- Level 2: ICU 조건 완화 (응급처치 가능 수준으로) ----
    # CT/ventilator는 여전히 필요하면 있는 곳 선호하지만, 여기선 "완화"니까 컷을 약하게
    # 단, ER bed가 0이면 응급처치도 못한다고 보고 탈락시킴
    if p["er_beds"] <= 0:
        pass
    else:
        return 2

    # ---- Level 3: 최후 후보 ----
    return 3

def sample_payload(rng: random.Random) -> dict:
    # Patient - 가정으로 생성
    severity = rng.choice([0, 1, 2])  # LOW/MED/HIGH
    cond_trauma = 1 if rng.random() < 0.2 else 0

    # 중증일수록 자원 필요 확률↑ (현실성)
    need_icu = 1 if rng.random() < (0.10 + 0.25 * severity) else 0
    need_ventilator = 1 if rng.random() < (0.05 + 0.20 * severity) else 0
    need_ct = 1 if rng.random() < (0.20 + 0.25 * severity) else 0
    need_mri = 1 if rng.random() < (0.05 + 0.15 * severity) else 0

    llm_confidence = round(rng.uniform(0.55, 0.95), 2)

    # Hospital (Public API) - 가정으로 생성
    er_beds = rng.randint(0, 40)
    icu_beds = rng.randint(0, 12)
    trauma_icu_beds = rng.randint(0, 6) if rng.random() < 0.3 else 0

    ct_available = 1 if rng.random() < 0.85 else 0
    ventilator_available = 1 if rng.random() < 0.80 else 0

    # Distance / Transport
    distance_km = round(rng.uniform(0.5, 30.0), 2)
    travel_time_min = round(distance_km * 2.0 + rng.uniform(0, 8), 1)

    # Location
    district_level = rng.choice([0, 1, 2])
    same_district = 1 if district_level == 0 else 0

    # Time Context
    hour = rng.randint(0, 23)
    is_night = 1 if (hour >= 22 or hour <= 6) else 0
    is_weekend = 1 if rng.random() < (2 / 7) else 0

    payload = {
        "severity": severity,
        "cond_trauma": cond_trauma,
        "need_icu": need_icu,
        "need_ventilator": need_ventilator,
        "need_ct": need_ct,
        "need_mri": need_mri,
        "llm_confidence": llm_confidence,
        "er_beds": er_beds,
        "icu_beds": icu_beds,
        "trauma_icu_beds": trauma_icu_beds,
        "ct_available": ct_available,
        "ventilator_available": ventilator_available,
        "distance_km": distance_km,
        "travel_time_min": travel_time_min,
        "same_district": same_district,
        "district_level": district_level,
        "hour": hour,
        "is_night": is_night,
        "is_weekend": is_weekend,
    }

    # filter_level 계산해서 payload에 포함
    payload["filter_level"] = compute_filter_level(payload)

    return payload

def simulate_accept_prob(p: dict) -> float:
    # 필수 조건 컷
    if p["need_ct"] == 1 and p["ct_available"] == 0:
        return 0.01
    if p["need_ventilator"] == 1 and p["ventilator_available"] == 0:
        return 0.01
    if p["need_icu"] == 1 and p["icu_beds"] <= 0:
        return 0.02
    if p["cond_trauma"] == 1 and p["need_icu"] == 1 and p["trauma_icu_beds"] <= 0:
        return 0.03

    # 확률 점수
    z = -1.0
    z += 0.05 * p["er_beds"]
    z += 0.10 * p["icu_beds"]
    z += 0.08 * p["trauma_icu_beds"]

    z -= 0.06 * p["travel_time_min"]
    z -= 0.5 * p["is_night"]
    z -= 0.3 * p["is_weekend"]
    z -= 0.25 * p["severity"]

    z += 0.3 * (p["llm_confidence"] - 0.75)

    if p["district_level"] == 0:
        z += 0.2
    elif p["district_level"] == 1:
        z += 0.05

    # filter_level이 낮을수록 수용 확률이 약간 더 높게
    z -= 0.35 * p["filter_level"]

    prob = float(sigmoid(z))
    return max(0.01, min(0.99, prob))

def main(n_rows: int = 30000, out_path: str = "data/processed/ml_input_features.csv"):
    rng = random.Random(SEED)

    # 폴더 없으면 생성
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    for _ in range(n_rows):
        payload = sample_payload(rng)
        prob = simulate_accept_prob(payload)
        accept = 1 if rng.random() < prob else 0

        row = {k: payload[k] for k in FEATURES}
        row["accept"] = accept
        rows.append(row)

    df = pd.DataFrame(rows, columns=FEATURES + ["accept"])
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] saved: {out_path}")
    print(f"rows={len(df)}  accept_rate={df['accept'].mean():.3f}")
    print("filter_level distribution:")
    print(df["filter_level"].value_counts(normalize=True).sort_index().round(3))

if __name__ == "__main__":
    main()
