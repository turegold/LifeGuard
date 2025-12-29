import pandas as pd


# NaN, None을 안전하게 int 변환
def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# 컬럼이 없으면 기본값으로 생성
def ensure_column(df: pd.DataFrame, col: str, default):
    if col not in df.columns:
        df[col] = default
    return df


# 환자 상태를 기반으로 수용 가능한 병원 후보를 필터링
def filter_hospitals(hospital_df: pd.DataFrame, patient_info: dict) -> pd.DataFrame:

    df = hospital_df.copy()


    # 0. 필수 컬럼 보정
    int_columns = [
        "hvec", "hvicc", "hvccc", "hvcc",
        "hv9", "hv8"
    ]

    yn_columns = [
        "hvventiayn", "hvctayn", "hvmriayn",
        "hv10", "hv11"
    ]

    for col in int_columns:
        df = ensure_column(df, col, 0)
        df[col] = df[col].apply(safe_int)

    for col in yn_columns:
        df = ensure_column(df, col, "N")


    # 1. 응급실 기본 조건
    df = df[df["hvec"] > 0]


    # 2. ICU 필요 여부
    if patient_info["required_resources"]["need_icu"]:
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
        df = df[df["hvccc"] > 0]

    elif condition == "RESPIRATORY":
        df = df[df["hvventiayn"] == "Y"]

    elif condition == "NEURO":
        df = df[df["hvcc"] > 0]

    elif condition == "TRAUMA":
        df = df[df["hv9"] > 0]

    elif condition == "BURN":
        df = df[df["hv8"] > 0]

    elif condition == "PEDIATRIC":
        df = df[(df["hv10"] == "Y") | (df["hv11"] == "Y")]

    return df.reset_index(drop=True)