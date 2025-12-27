import os
from dotenv import load_dotenv

load_dotenv()

SERVICE_KEY = os.getenv("SERVICE_KEY")
API_URL = os.getenv("API_URL")

if not SERVICE_KEY:
    raise RuntimeError("SERVICE_KEY is not set in .env")

if not API_URL:
    raise RuntimeError("API_URL is not set in .env")