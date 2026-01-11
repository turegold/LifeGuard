from pydantic import BaseModel
from typing import List


class EmergencyGuidanceRequest(BaseModel):
    emergency_text: str


class EmergencyGuidanceResponse(BaseModel):
    situation_summary: str
    immediate_actions: List[str]
    do_not_do: List[str]