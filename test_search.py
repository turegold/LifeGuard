import pandas as pd
from src.hospital.search import search_nearby_hospitals
from src.llm.emergency_parser import parse_emergency_text


def main():
    user_lat = 37.6213508
    user_lon = 127.0562448

    emergency_text = "아버지가 칼에 흉부를 찔려 쓰러져 있고 피가 많이 납니다."

    # =========================
    # Step 1: 환자 상태 구조화 (이미 PPT에 사용)
    # =========================
    patient_info = parse_emergency_text(emergency_text)

    # =========================
    # Step 2: 후보 병원 탐색
    # =========================
    result_df = search_nearby_hospitals(
        city="서울특별시",
        district="강남구",
        patient_info=patient_info,
        user_lat=user_lat,
        user_lon=user_lon
    )

    if result_df.empty:
        print("❌ 후보 병원 없음")
        return

    # =========================
    # Step2 PPT용 핵심 컬럼 정리
    # =========================
    ppt_df = result_df[
        [
            "hpid",                       # 병원 ID
            "dutyname",                   # 병원명
            "distance_km",                # 거리
            "estimated_travel_time_min",  # 이동 시간
            "hvec",                       # 응급실 병상
            "hvoc",                       # ICU 병상
            "hvventiayn",                 # 인공호흡기
            "hvctayn",                    # CT 가능
            "hvmriayn"                    # MRI 가능
        ]
    ].copy()

    # 컬럼명 PPT용으로 보기 좋게 변경
    ppt_df = ppt_df.rename(columns={
        "hpid": "병원 ID",
        "dutyname": "병원명",
        "distance_km": "거리(km)",
        "estimated_travel_time_min": "이동시간(분)",
        "hvec": "응급실 병상 수",
        "hvoc": "ICU 병상 수",
        "hvventiayn": "인공호흡기 가능",
        "hvctayn": "CT 가능",
        "hvmriayn": "MRI 가능"
    })

    # 정렬 (가까운 병원 우선)
    ppt_df = ppt_df.sort_values("이동시간(분)")

    print("\n📌 Step2 후보 병원 결과")
    print(ppt_df.head(10))

    # =========================
    # CSV 저장 (PPT 캡처용)
    # =========================
    ppt_df.to_csv(
        "step2_candidate_hospitals.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n✅ step2_candidate_hospitals.csv 저장 완료")


if __name__ == "__main__":
    main()