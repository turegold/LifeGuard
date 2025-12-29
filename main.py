from hospital.filtering import filter_hospitals
from data.fetch_hospital_api import fetch_emergency_data
from llm.emergency_parser import parse_emergency_text


def main():
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

    print(f"전체 병원 수: {len(hospital_df)}")
    print(f"수용 가능 병원 수: {len(filtered_df)}")
    print(filtered_df[["dutyname", "dutytel3"]])


if __name__ == "__main__":
    main()