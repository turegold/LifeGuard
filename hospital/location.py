import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

KAKAO_API_KEY = os.getenv("KAKAO_REST_API_KEY")

KAKAO_SEARCH_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE_DIR, "data", "hospital_location_cache.json")



# 캐시 로드 / 저장
def load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache: dict):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)



# 카카오 API 호출
def fetch_location_from_kakao(hospital_name: str) -> dict | None:
    headers = {
        "Authorization": f"KakaoAK {KAKAO_API_KEY}"
    }

    params = {
        "query": hospital_name,
        "size": 1
    }

    response = requests.get(
        KAKAO_SEARCH_URL,
        headers=headers,
        params=params,
        timeout=5
    )
    response.raise_for_status()

    documents = response.json().get("documents", [])

    if not documents:
        return None

    return {
        "lat": float(documents[0]["y"]),
        "lon": float(documents[0]["x"])
    }


# 외부에서 쓰는 메인 함수
def get_hospital_location(hospital_name: str) -> dict | None:
    cache = load_cache()

    # 캐시에 있으면 바로 반환
    if hospital_name in cache:
        return cache[hospital_name]

    # 없으면 카카오 API 호출
    location = fetch_location_from_kakao(hospital_name)

    if location:
        cache[hospital_name] = location
        save_cache(cache)

    return location

if __name__ == "__main__":
    test_hospitals = [
        "삼성서울병원",
        "연세대학교의과대학강남세브란스병원",
        "존재하지않는병원테스트"
    ]

    for name in test_hospitals:
        print(f"\n병원명: {name}")
        loc = get_hospital_location(name)

        if loc:
            print(f"  위도(lat): {loc['lat']}")
            print(f"  경도(lon): {loc['lon']}")
        else:
            print("  ❌ 좌표를 찾을 수 없습니다.")