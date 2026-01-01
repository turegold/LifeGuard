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




def filter_hospitals_strict(
    hospital_df: pd.DataFrame,
    patient_info: dict
) -> pd.DataFrame:

    df = hospital_df.copy()

    # 필수 컬럼 보정
    int_columns = ["hvec", "hvicc", "hvccc", "hvcc", "hv9", "hv8"]
    yn_columns = ["hvventiayn", "hvctayn", "hvmriayn", "hv10", "hv11"]

    for col in int_columns:
        df = ensure_column(df, col, 0)
        df[col] = df[col].apply(safe_int)

    for col in yn_columns:
        df = ensure_column(df, col, "N")

    # 1. 응급실 필수
    df = df[df["hvec"] > 0]

    # 2. ICU 필요 여부
    if patient_info["required_resources"]["need_icu"]:
        df = df[df["hvicc"] > 0]

    # 3. 인공호흡기
    if patient_info["required_resources"]["need_ventilator"]:
        df = df[df["hvventiayn"] == "Y"]

    # 4. CT / MRI
    if patient_info["required_resources"]["need_ct"]:
        df = df[df["hvctayn"] == "Y"]

    if patient_info["required_resources"]["need_mri"]:
        df = df[df["hvmriayn"] == "Y"]

    # 5. 질환별 조건
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




"""
    단계별로 필터링을 완화하며,
    병원별 최초 통과 filter_level을 기록하여 반환
"""
def filter_hospitals(
    hospital_df: pd.DataFrame,
    patient_info: dict,
    min_candidates: int = 5
) -> pd.DataFrame:


    collected = []
    seen_hpid = set()

    # Level 0: 완전 매칭
    df0 = filter_hospitals_strict(hospital_df, patient_info)
    if not df0.empty:
        df0 = df0.copy()
        df0["filter_level"] = 0
        collected.append(df0)
        seen_hpid.update(df0["hpid"])

    # Level 1: 질환 조건 완화
    relaxed_info_1 = patient_info.copy()
    relaxed_info_1["suspected_condition"] = "UNKNOWN"

    df1 = filter_hospitals_strict(hospital_df, relaxed_info_1)
    df1 = df1[~df1["hpid"].isin(seen_hpid)]

    if not df1.empty:
        df1 = df1.copy()
        df1["filter_level"] = 1
        collected.append(df1)
        seen_hpid.update(df1["hpid"])

    # Level 2: ICU 요구 완화
    relaxed_info_2 = relaxed_info_1.copy()
    relaxed_info_2["required_resources"]["need_icu"] = False

    df2 = filter_hospitals_strict(hospital_df, relaxed_info_2)
    df2 = df2[~df2["hpid"].isin(seen_hpid)]

    if not df2.empty:
        df2 = df2.copy()
        df2["filter_level"] = 2
        collected.append(df2)
        seen_hpid.update(df2["hpid"])

    # Level 3: 응급실만 있으면 OK
    df3 = hospital_df.copy()
    df3 = ensure_column(df3, "hvec", 0)
    df3["hvec"] = df3["hvec"].apply(safe_int)
    df3 = df3[df3["hvec"] > 0]
    df3 = df3[~df3["hpid"].isin(seen_hpid)]

    if not df3.empty:
        df3 = df3.copy()
        df3["filter_level"] = 3
        collected.append(df3)

    if not collected:
        return pd.DataFrame()

    result = pd.concat(collected).reset_index(drop=True)
    return result