import requests
import xml.etree.ElementTree as ET
from config import SERVICE_KEY

BASE_URL = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/getEgytBassInfoInqire"


def fetch_hospital_static_by_hpid(hpid: str) -> dict | None:
    params = {
        "serviceKey": SERVICE_KEY,
        "HPID": hpid,
        "pageNo": 1,
        "numOfRows": 1,
    }

    res = requests.get(BASE_URL, params=params, timeout=10)
    res.raise_for_status()

    root = ET.fromstring(res.content)
    item = root.find(".//item")

    if item is None:
        return None

    row = {}
    for child in item:
        row[f"total_{child.tag.lower()}"] = child.text

    return row