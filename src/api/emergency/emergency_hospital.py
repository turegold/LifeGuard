from fastapi import APIRouter, HTTPException

from src.api.emergency.schemas import (
    EmergencyHospitalRequest,
    EmergencyHospitalResponse,
    HospitalItem,
    HospitalExplanation,
    HospitalReason,
)

from src.hospital.search import search_nearby_hospitals
from src.llm.emergency_parser import parse_emergency_text
from src.ml.feature_builder import build_ml_features
from src.ml.recommend import recommend_hospitals
from src.utils.rank_merger import merge_rank_with_payloads
from src.llm.hospital_explainer import explain_hospital_ranking

router = APIRouter(prefix="/api/emergency", tags=["Emergency"])


@router.post("/hospitals", response_model=EmergencyHospitalResponse)
def recommend_emergency_hospitals(req: EmergencyHospitalRequest):
    # =========================
    # 1. 환자 정보 파싱
    # =========================
    patient_info = parse_emergency_text(req.emergency_text)

    # =========================
    # 2. 병원 후보 탐색
    # =========================
    result_df = search_nearby_hospitals(
        city=None,
        district=None,
        patient_info=patient_info,
        user_lat=req.user_location.lat,
        user_lon=req.user_location.lon,
    )

    if result_df.empty:
        raise HTTPException(status_code=404, detail="병원 후보를 찾을 수 없습니다.")

    # =========================
    # 3. ML Feature 생성
    # =========================
    hospital_payloads = []

    for _, row in result_df.iterrows():
        feature = build_ml_features(row, patient_info)

        hospital_payloads.append({
            "meta": {
                "hospital_id": feature.get("hospital_id"),
                "hospital_name": feature.get("hospital_name"),
                "hospital_phone": feature.get("hospital_phone"),
            },
            "features": feature,
        })

    # =========================
    # 4. 병원 추천 (ML)
    # =========================
    recommendations = recommend_hospitals(
        hospital_payloads,
        threshold=0.01,
        top_k=5,
        max_filter_level=2,
    )

    if not recommendations:
        raise HTTPException(status_code=404, detail="추천 가능한 병원이 없습니다.")

    # =========================
    # 5. 추천 결과 정리 (상세 정보 포함)
    # =========================
    hospitals = []

    # hospital_id → features 매핑
    feature_map = {
        p["meta"]["hospital_id"]: p["features"]
        for p in hospital_payloads
    }

    for rank, r in enumerate(recommendations, start=1):
        hid = r.get("hospital_id")
        features = feature_map.get(hid, {})

        hospitals.append(
            HospitalItem(
                rank=rank,
                hospital_id=hid,
                hospital_name=r.get("hospital_name"),
                hospital_phone=r.get("hospital_phone"),
                accept_prob=r.get("accept_prob"),

                # 거리 / 이동
                distance_km=r.get("distance_km"),
                travel_time_min=r.get("travel_time_min"),

                # 병상 (실시간)
                er_beds=features.get("er_beds"),
                icu_beds=features.get("icu_beds"),
                trauma_icu_beds=features.get("trauma_icu_beds"),

                # 병상 (정적)
                total_er_beds=features.get("total_er_beds"),
                total_icu_beds=features.get("total_icu_beds"),
                total_beds=features.get("total_beds"),

                # 장비
                ct_available=bool(features.get("ct_available")),
                ventilator_available=bool(features.get("ventilator_available")),

                # 기타
                filter_level=features.get("filter_level"),
                district_level=features.get("district_level"),
                same_district=features.get("same_district"),
            )
        )

    # =========================
    # 6. 병원 추천 이유 (LLM)
    # =========================
    final_results = merge_rank_with_payloads(
        recommendations,
        hospital_payloads
    )

    explanation_text = explain_hospital_ranking(
        final_results,
        patient_info
    )

    explanation = HospitalExplanation(
        summary=explanation_text.get("summary", ""),
        details=[
            HospitalReason(
                hospital_id=d["hospital_id"],
                reason=d["reason"]
            )
            for d in explanation_text.get("details", [])
        ],
    )

    # =========================
    # 7. 응답 반환
    # =========================
    return EmergencyHospitalResponse(
        hospitals=hospitals,
        ranking_explanation=explanation
    )