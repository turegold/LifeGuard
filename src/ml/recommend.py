# src/ml/recommend.py
import joblib
import pandas as pd
from typing import List, Dict

from src.ml.schema import FEATURES, payload_to_df

MODEL_PATH = "models/accept_model.joblib"


def recommend_hospitals(
    hospital_payloads: List[Dict],
    *,
    threshold: float = 0.3,
    top_k: int = 5,
    max_filter_level: int = 1,
):
    """
    hospital_payloads: [
      {
        "meta": {...},      # 병원 식별 정보 (이름, id 등)
        "features": {...}   # ML feature dict (LLM/feature_builder 결과)
      },
      ...
    ]
    """

    # 1️⃣ 모델 로드 (학습된 모델)
    model = joblib.load(MODEL_PATH)

    rows = []
    metas = []

    # 2️⃣ payload → DataFrame 변환
    for item in hospital_payloads:
        features = item["features"]
        meta = item["meta"]

        # filter_level 컷 (1차 규칙 필터)
        if features.get("filter_level", 99) > max_filter_level:
            continue

        df_row = payload_to_df(features)
        rows.append(df_row)

        # 거리랑 시간 추가함 -용민-
        metas.append({
            **meta,
            "distance_km": features.get("distance_km"),
            "travel_time_min": features.get("travel_time_min"),
        })

    if not rows:
        return []

    X = pd.concat(rows, ignore_index=True)

    # 3️⃣ ML 확률 예측
    probs = model.predict_proba(X)[:, 1]

    # 4️⃣ 결과 합치기
    result_df = pd.DataFrame(metas)
    result_df["accept_prob"] = probs

    # 5️⃣ soft threshold (너무 낮은 확률 제거)
    result_df = result_df[result_df["accept_prob"] >= threshold]

    if result_df.empty:
        return []

    # 6️⃣ Top-K 랭킹
    result_df = result_df.sort_values(
        "accept_prob", ascending=False
    ).head(top_k)

    print("\n[DEBUG] Ranked result_df:")
    print(result_df)

    return result_df.to_dict(orient="records")

    