from fastapi import APIRouter, HTTPException
from src.rag.rag_guidance import generate_emergency_guidance
from src.llm.emergency_parser import parse_emergency_text
from src.api.emergency.schemas import (
    EmergencyGuidanceRequest,
    EmergencyGuidanceResponse
)

router = APIRouter(
    prefix="/api/emergency",
    tags=["Emergency"]
)


@router.post(
    "/guidance",
    response_model=EmergencyGuidanceResponse
)
def get_emergency_guidance(
    request: EmergencyGuidanceRequest
):
    emergency_text = request.emergency_text.strip()

    if not emergency_text:
        raise HTTPException(
            status_code=400,
            detail="emergency_text is required"
        )

    try:
        patient_info = parse_emergency_text(emergency_text)
        condition = patient_info.get("suspected_condition", "UNKNOWN")
        print("[DEBUG]")
        print(condition)

        guidance = generate_emergency_guidance(
            query=emergency_text,
            condition=condition
        )

        return guidance

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate guidance: {str(e)}"
        )