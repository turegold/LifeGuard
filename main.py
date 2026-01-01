import os
import pandas as pd

from src.hospital.search import search_nearby_hospitals
from src.llm.emergency_parser import parse_emergency_text
from src.ml.feature_builder import build_ml_features


def main():
    # 내 위치(임시)
    user_lat = 37.6213508
    user_lon = 127.0562448

    # LLM에게 보낼 텍스트
    patient_info = parse_emergency_text(
        "아버지가 칼에 흉부를 찔려 쓰러져 있고 피가 많이 납니다."
    )


    result_df = search_nearby_hospitals(
        city="서울특별시",
        district="강남구",
        patient_info=patient_info,
        user_lat=user_lat,
        user_lon=user_lon
    )

    if result_df.empty:
        print("❌ 병원 후보 없음")
        return


    # ML Feature 생성
    ml_features = []

    for _, row in result_df.iterrows():
        feature = build_ml_features(row, patient_info)
        ml_features.append(feature)

    ml_df = pd.DataFrame(ml_features)


    # 저장 (임시)
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, "ml_input_features.csv")

    ml_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ ML 입력용 feature CSV 생성 완료: {csv_path}")
    print(ml_df.head())


if __name__ == "__main__":
    main()