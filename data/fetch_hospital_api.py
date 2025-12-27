import requests
import pandas as pd
import xml.etree.ElementTree as ET
from config import SERVICE_KEY, API_URL



# 공공 응급의료 API 호출 후 DataFrame 반환

def fetch_emergency_data(
    service_key: str,
    stage1: str,
    stage2: str,
    page_no: int = 1,
    num_of_rows: int = 100
) -> pd.DataFrame:

    params = {
        "serviceKey": service_key,
        "STAGE1": stage1,
        "STAGE2": stage2,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
    }

    response = requests.get(API_URL, params=params, timeout=10)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    # 결과 코드 확인
    result_code = root.findtext(".//resultCode")
    result_msg = root.findtext(".//resultMsg")

    if result_code != "00":
        raise RuntimeError(f"API Error {result_code}: {result_msg}")

    items = root.findall(".//item")

    rows = []
    for item in items:
        row = {}
        for child in item:
            row[child.tag] = child.text
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":

    df = fetch_emergency_data(
        service_key=SERVICE_KEY,
        stage1="서울특별시",
        stage2="강남구",
        page_no=1,
        num_of_rows=50,
    )

    print(f"데이터 개수: {len(df)}")
    print(df.head())