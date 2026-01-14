# src/utils/geocode.py
import requests
from dotenv import load_dotenv
from openai import OpenAI
from config import KAKAO_REST_API_KEY

# 환경 변수 로드
load_dotenv()
def latlon_to_region(lat: float, lon: float) -> tuple[str, str]:
    """
    위경도를 시/구로 변환 (예: 서울특별시, 강남구)
    """
    url = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }
    params = {
        "x": lon,
        "y": lat
    }

    res = requests.get(url, headers=headers, params=params).json()
    region = res["documents"][0]

    city = region["region_1depth_name"]
    district = region["region_2depth_name"]

    return city, district