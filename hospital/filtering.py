import pandas as pd


# NaN, None을 안전하게 int 변환
def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# 환자 상태를 기반으로 수용 가능한 병원 후보를 필터링
def filter_hospitals(hospital_df: pd.DataFrame, patient_info: dict) -> pd.DataFrame:


    df = hospital_df.copy()

    # 1. 응급실 기본 조건
    df["hvec"] = df["hvec"].apply(safe_int)
    df = df[df["hvec"] > 0]


    # 2. ICU 필요 여부
    if patient_info["required_resources"]["need_icu"]:
        df["hvicc"] = df["hvicc"].apply(safe_int)
        df = df[df["hvicc"] > 0]


    # 3. 인공호흡기 필요 여부
    if patient_info["required_resources"]["need_ventilator"]:
        df = df[df["hvventiayn"] == "Y"]

    # 4. CT / MRI 필요 여부
    if patient_info["required_resources"]["need_ct"]:
        df = df[df["hvctayn"] == "Y"]

    if patient_info["required_resources"]["need_mri"]:
        df = df[df["hvmriayn"] == "Y"]


    # 5. 의심 질환별 필터링
    condition = patient_info["suspected_condition"]

    if condition == "CARDIAC":
        df["hvccc"] = df["hvccc"].apply(safe_int)
        df = df[df["hvccc"] > 0]

    elif condition == "RESPIRATORY":
        df = df[df["hvventiayn"] == "Y"]

    elif condition == "NEURO":
        df["hvcc"] = df["hvcc"].apply(safe_int)
        df = df[df["hvcc"] > 0]

    elif condition == "TRAUMA":
        df["hv9"] = df["hv9"].apply(safe_int)
        df = df[df["hv9"] > 0]

    elif condition == "BURN":
        df["hv8"] = df["hv8"].apply(safe_int)
        df = df[df["hv8"] > 0]

    elif condition == "PEDIATRIC":
        df = df[(df["hv10"] == "Y") | (df["hv11"] == "Y")]

    # =========================
    # 결과 반환
    # =========================
    return df.reset_index(drop=True)