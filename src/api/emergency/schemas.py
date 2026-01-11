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
    accept_prob: Optional[float]
    distance_km: Optional[float]
    travel_time_min: Optional[float]


class HospitalReason(BaseModel):
    hospital_id: str
    reason: str


class HospitalExplanation(BaseModel):
    summary: str
    details: List[HospitalReason]


class EmergencyHospitalResponse(BaseModel):
    hospitals: List[HospitalItem]
    ranking_explanation: HospitalExplanation