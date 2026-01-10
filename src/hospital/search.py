import os
import pandas as pd

from src.utils.region import get_search_districts
from src.opendata.fetch_hospital_api import fetch_emergency_data
from src.opendata.fetch_hospital_static_api import fetch_hospital_static_by_hpid
from src.hospital.filtering import filter_hospitals
from src.hospital.distance import add_distance_features


# =========================
# 정적 병원 정보 CSV 경로
# =========================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)
STATIC_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "hospital_static.csv")


# =========================
# 정적 병원 정보 캐시 로드
# =========================
def load_hospital_static_cache() -> pd.DataFrame:
    if not os.path.exists(STATIC_CSV_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(STATIC_CSV_PATH)
        df.columns = df.columns.str.lower()
        return df
    except Exception:
        return pd.DataFrame()


def save_hospital_static_cache(df: pd.DataFrame):
    os.makedirs(os.path.dirname(STATIC_CSV_PATH), exist_ok=True)
    df.to_csv(STATIC_CSV_PATH, index=False, encoding="utf-8-sig")


# =========================
# hpid 기준 정적 정보 조회 (CSV → API fallback)
# =========================
def get_static_info_by_hpid(
    hpid: str,
    static_df: pd.DataFrame
) -> tuple[dict | None, pd.DataFrame]:

    if not static_df.empty and "total_hpid" in static_df.columns:
        hit = static_df[static_df["total_hpid"] == hpid]
        if not hit.empty:
            return hit.iloc[0].to_dict(), static_df

    # ❌ 캐시에 없음 → API 호출
    static_info = fetch_hospital_static_by_hpid(hpid)
    if static_info is None:
        return None, static_df

    new_df = pd.DataFrame([static_info])
    updated_df = (
        pd.concat([static_df, new_df], ignore_index=True)
        if not static_df.empty
        else new_df
    )

    save_hospital_static_cache(updated_df)

    return static_info, updated_df


# =========================
# 메인 함수 (이름 유지)
# =========================
def search_nearby_hospitals(
    city: str,
    district: str,
    patient_info: dict,
    user_lat: float,
    user_lon: float
) -> pd.DataFrame:
    """
    현재 시/구 기준으로 인접 구까지 확장 탐색하여
    실시간 응급 데이터 + 정적 병원 정보를 결합한 병원 후보 반환
    """

    all_results = []
    seen_hpid = set()

    # 🔹 정적 병원 정보 캐시 (1회 로드)
    static_df = load_hospital_static_cache()

    # 탐색할 구 + district_level 목록
    search_targets = get_search_districts(city, district)

    for target_district, district_level in search_targets:
        # =========================
        # 1. 실시간 응급 데이터
        # =========================
        hospital_df = fetch_emergency_data(
            stage1=city,
            stage2=target_district
        )

        if hospital_df.empty:
            continue

        # =========================
        # 2. 환자 조건 필터링
        # =========================
        filtered_df = filter_hospitals(hospital_df, patient_info)

        if filtered_df.empty:
            continue

        filtered_df = filtered_df.copy()
        filtered_df["district_level"] = district_level
        filtered_df["same_district"] = 1 if district_level == 0 else 0

        # =========================
        # 3. 거리 / 시간 feature
        # =========================
        filtered_df = add_distance_features(
            filtered_df,
            user_lat=user_lat,
            user_lon=user_lon
        )

        filtered_df = filtered_df.dropna(
            subset=["distance_km", "estimated_travel_time_min"]
        )

        if filtered_df.empty:
            continue

        # =========================
        # 4. 정적 병원 정보 병합 (hpid 단위)
        # =========================
        static_rows = []

        for _, row in filtered_df.iterrows():
            static_info, static_df = get_static_info_by_hpid(
                row["hpid"], static_df
            )
            static_rows.append(static_info or {})

        static_part_df = pd.DataFrame(static_rows)

        if not static_part_df.empty:
            filtered_df = pd.concat(
                [filtered_df.reset_index(drop=True),
                 static_part_df.reset_index(drop=True)],
                axis=1
            )

        # =========================
        # 5. 중복 병원 제거
        # =========================
        filtered_df = filtered_df[~filtered_df["hpid"].isin(seen_hpid)]
        seen_hpid.update(filtered_df["hpid"])

        all_results.append(filtered_df)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results).reset_index(drop=True)