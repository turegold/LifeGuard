from pydantic import BaseModel
from typing import List, Optional

class EmergencyGuidanceRequest(BaseModel):
    emergency_text: str


class EmergencyGuidanceResponse(BaseModel):
    situation_summary: str
    immediate_actions: List[str]
    do_not_do: List[str]

class Location(BaseModel):
    lat: float
    lon: float


class EmergencyHospitalRequest(BaseModel):
    emergency_text: str
    user_location: Location


class HospitalItem(BaseModel):
    rank: int
    hospital_id: str
    hospital_name: str
    hospital_phone: Optional[str]

    # 추천 결과
    accept_prob: Optional[float]

    # 거리 / 이동
    distance_km: Optional[float]
    travel_time_min: Optional[float]

    # 병상 현황 (실시간)
    er_beds: Optional[int]
    icu_beds: Optional[int]
    trauma_icu_beds: Optional[int]

    # 병상 규모 (정적)
    total_er_beds: Optional[int]
    total_icu_beds: Optional[int]
    total_beds: Optional[int]

    # 장비
    ct_available: Optional[bool]
    ventilator_available: Optional[bool]

    # 기타
    filter_level: Optional[int]
    district_level: Optional[int]
    same_district: Optional[int]


class HospitalReason(BaseModel):
    hospital_id: str
    reason: str


class HospitalExplanation(BaseModel):
    summary: str
    details: List[HospitalReason]


class EmergencyHospitalResponse(BaseModel):
    hospitals: List[HospitalItem]
    ranking_explanation: HospitalExplanation