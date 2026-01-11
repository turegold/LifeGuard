from fastapi import FastAPI
from src.api.emergency.guidance import router as guidance_router
from src.api.emergency.emergency_hospital import router as emergency_hospital_router

app = FastAPI(
    title="Emergency AI API",
    version="0.1.0"
)

app.include_router(guidance_router)
app.include_router(emergency_hospital_router)

@app.get("/")
def health_check():
    return {"status": "ok"}