from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.emergency.guidance import router as guidance_router
from src.api.emergency.emergency_hospital import router as emergency_hospital_router

app = FastAPI(
    title="Emergency AI API",
    version="0.1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router 등록
app.include_router(guidance_router)
app.include_router(emergency_hospital_router)

@app.get("/")
def health_check():
    return {"status": "ok"}