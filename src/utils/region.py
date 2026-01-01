import json
import os

REGION_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "regions.json"
)


def load_regions() -> dict:
    with open(REGION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_search_districts(
    city: str,
    current_district: str,
    max_level: int = 2
) -> list[tuple[str, int]]:
    """
    현재 시/구를 기준으로 탐색할 구 목록과 district_level 반환

    district_level 의미:
    0: 동일 구
    1: 같은 시, 다른 구
    2: (확장용, 현재는 동일 시 전체)
    """

    regions = load_regions()

    if city not in regions:
        raise ValueError(f"지원하지 않는 시입니다: {city}")

    districts = regions[city]

    if current_district not in districts:
        raise ValueError(f"{city}에 {current_district}가 존재하지 않습니다")

    result = []

    # Level 0: 현재 구
    result.append((current_district, 0))

    if max_level >= 1:
        for d in districts:
            if d != current_district:
                result.append((d, 1))

    return result