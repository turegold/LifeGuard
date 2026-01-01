import pandas as pd

from src.utils.region import get_search_districts
from src.opendata.fetch_hospital_api import fetch_emergency_data
from src.hospital.filtering import filter_hospitals
from src.hospital.distance import add_distance_features


def search_nearby_hospitals(
    city: str,
    district: str,
    patient_info: dict,
    user_lat: float,
    user_lon: float
) -> pd.DataFrame:
    """
    현재 시/구 기준으로 인접 구까지 확장 탐색하여 병원 후보 반환
    """

    all_results = []
    seen_hpid = set()

    # 탐색할 구 + district_level 목록
    search_targets = get_search_districts(city, district)

    for target_district, district_level in search_targets:
        # 병원 데이터 호출
        hospital_df = fetch_emergency_data(
            stage1=city,
            stage2=target_district
        )

        if hospital_df.empty:
            continue

        # 필터링 (filter_level 포함됨)
        filtered_df = filter_hospitals(hospital_df, patient_info)

        if filtered_df.empty:
            continue

        # district_level 컬럼 추가
        filtered_df = filtered_df.copy()
        filtered_df["district_level"] = district_level
        filtered_df["same_district"] = 1 if district_level == 0 else 0

        # 거리 / 시간 feature
        filtered_df = add_distance_features(
            filtered_df,
            user_lat=user_lat,
            user_lon=user_lon
        )

        # 위치 정보 없는 병원 제거
        filtered_df = filtered_df.dropna(
            subset=["distance_km", "estimated_travel_time_min"]
        )

        if filtered_df.empty:
            continue

        # 중복 병원 제거
        filtered_df = filtered_df[~filtered_df["hpid"].isin(seen_hpid)]
        seen_hpid.update(filtered_df["hpid"])

        all_results.append(filtered_df)



    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results).reset_index(drop=True)