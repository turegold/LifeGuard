from hospital.filtering import filter_hospitals
from data.fetch_hospital_api import fetch_emergency_data
from llm.emergency_parser import parse_emergency_text
from hospital.distance import add_distance_features

def main():
    # (임시) 내 위치
    user_lat = 37.6213508
    user_lon = 127.0562448
    # 1. 환자 입력 → LLM 파싱
    patient_info = parse_emergency_text(
        "아버지가 숨을 잘 못 쉬고 가슴 통증이 심해요. 식은땀을 흘리고 있습니다."
    )

    # 2. 병원 데이터 불러오기
    hospital_df = fetch_emergency_data(
        stage1="서울특별시",
        stage2="강남구"
    )

    # 3. 후보 병원 필터링
    filtered_df = filter_hospitals(hospital_df, patient_info)

    # 4. 거리/ 시간 Feature 추가
    filtered_df = add_distance_features(
        filtered_df,
        user_lat=user_lat,
        user_lon=user_lon
    )

    print(f"전체 병원 수: {len(hospital_df)}")
    print(f"수용 가능 병원 수: {len(filtered_df)}")

    print(
        filtered_df[
            [
                "dutyname",
                "dutytel3",
                "distance_km",
                "estimated_travel_time_min",
            ]
        ].sort_values("distance_km")
    )


if __name__ == "__main__":
    main()