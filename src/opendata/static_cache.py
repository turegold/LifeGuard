import os
import pandas as pd

from src.opendata.fetch_hospital_static_api import fetch_hospital_static_by_hpid

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "hospital_static.csv")


def load_static_cache() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return pd.DataFrame()

    return df


def save_static_cache(df: pd.DataFrame):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


def get_static_info_by_hpid(
    hpid: str,
    static_df: pd.DataFrame
) -> tuple[dict | None, pd.DataFrame]:

    if not static_df.empty and "total_hpid" in static_df.columns:
        hit = static_df[static_df["total_hpid"] == hpid]
        if not hit.empty:
            return hit.iloc[0].to_dict(), static_df

    # CSV에 없으면 → API 호출
    static_info = fetch_hospital_static_by_hpid(hpid)
    if static_info is None:
        return None, static_df

    # CSV에 누적 저장
    new_df = pd.DataFrame([static_info])
    updated_df = (
        pd.concat([static_df, new_df], ignore_index=True)
        if not static_df.empty
        else new_df
    )

    save_static_cache(updated_df)

    return static_info, updated_df