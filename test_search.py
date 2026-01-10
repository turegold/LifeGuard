from src.hospital.search import search_nearby_hospitals

# 🔹 테스트용 더미 환자 정보
patient_info = {
    "suspected_condition": "chest_stab",
    "emergency_type": "stab",
    "body_part": "chest",
    "bleeding": True,
    "conscious": False,
    "severity": "critical",
    "required_resources": {
        "need_icu": True,
        "need_surgery": True,
        "need_ct": True,
        "need_mri": False,
        "need_ventilator": True
    }
}

df = search_nearby_hospitals(
    city="서울특별시",
    district="강남구",
    patient_info=patient_info,
    user_lat=37.6213508,
    user_lon=127.0562448
)

print("====== RESULT ======")
print(df.head())
print("\n컬럼 목록:")
print(df.columns.tolist())
print("\n병원 수:", len(df))