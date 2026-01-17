"""
본 파일은 실제 API 서버와는 별도로,
자연어 응급 상황 입력부터 병원 추천 및 응급 행동 가이드 제공까지의
전체 시스템 흐름을 시나리오 형태로 한 번에 실행하기 위한 테스트/데모용 스크립트임.
실제 서비스에서는 FastAPI 서버를 사용
"""

import os
import pandas as pd
import json

from src.hospital.search import search_nearby_hospitals
from src.llm.emergency_parser import parse_emergency_text
from src.ml.feature_builder import build_ml_features
from src.ml.recommend import recommend_hospitals
from src.utils.rank_merger import merge_rank_with_payloads
from src.llm.hospital_explainer import explain_hospital_ranking
from src.rag.rag_guidance import generate_emergency_guidance


def main():
    # 내 위치(임시)
    user_lat = 37.6213508
    user_lon = 127.0562448

    emergency_text = "어떤 남자가 흉부에 칼을 찔려서 쓰려져있습니다."

    # LLM에게 보낼 텍스트
    patient_info = parse_emergency_text(
        emergency_text
    )

    # 주변 병원 후보 탐색
    result_df = search_nearby_hospitals(
        city="서울특별시",
        district="강남구",
        patient_info=patient_info,
        user_lat=user_lat,
        user_lon=user_lon
    )


    # 후보 병원 CSV 저장
    candidate_columns = [
        "hpid",
        "total_dutyname",
        "distance_km",
        "estimated_travel_time_min",
        "hvec",
        "hvicc",
        "hvventiayn",
        "hvctayn",
        "hvmriayn",
        "hv9",
        "filter_level",
    ]

    candidate_df = result_df[
        [c for c in candidate_columns if c in result_df.columns]
    ]

    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    candidate_csv_path = os.path.join(DATA_DIR, "candidate_hospitals.csv")
    candidate_df.to_csv(
        candidate_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"후보 병원 CSV 저장 완료: {candidate_csv_path}")
    print(candidate_df.head())

    if result_df.empty:
        print("병원 후보 없음")
        return


    # ML Feature 생성
    ml_features = []
    hospital_payloads = []  # recommend_hospitals에 넘길 리스트

    for _, row in result_df.iterrows():
        feature = build_ml_features(row, patient_info)
        ml_features.append(feature)

        # meta는 row가 아니라 feature에서 꺼내기 (None 방지)
        hospital_payloads.append({
            "meta": {
                "hospital_name": feature.get("hospital_name"),
                "hospital_id": feature.get("hospital_id"),
                "hospital_phone": feature.get("hospital_phone"),
            },
            "features": feature
        })

    ml_df = pd.DataFrame(ml_features)

    # CSV 저장
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, "ml_input_features.csv")
    ml_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"ML 입력용 feature CSV 생성 완료: {csv_path}")
    print(ml_df.head())

    # ML 추천 (확률 예측 + threshold + Top-K)
    recommendations = recommend_hospitals(
        hospital_payloads,
        threshold=0.01,
        top_k=5,
        max_filter_level=2,
    )

    print("\n추천 병원 Top-5 결과:")
    if not recommendations:
        print("조건(필터/threshold)에 의해 추천 결과가 없습니다.")
        return

    for rank, r in enumerate(recommendations, start=1):
        name = r.get("hospital_name", "UNKNOWN")
        hid = r.get("hospital_id", "UNKNOWN")
        phone = r.get("hospital_phone", "UNKNOWN")
        prob = r.get("accept_prob", None)

        if prob is None:
            print(f"{rank}위 | {name} (ID: {hid}) | 전화: {phone}")
        else:
            print(f"{rank}위 | {name} (ID: {hid}) | 전화: {phone} | 수용확률={prob:.3f}")

    final_results = merge_rank_with_payloads(
        recommendations,
        hospital_payloads
    )


    # 병원 랭킹 설명 (LLM)
    explanation = explain_hospital_ranking(
        final_results,
        patient_info
    )

    print("\n 병원 추천 이유 설명:")
    print(explanation)

    # 응급 대응 지침 (RAG)
    guidance = generate_emergency_guidance(emergency_text)

    print("\n응급 대응 지침")
    print(json.dumps(guidance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
