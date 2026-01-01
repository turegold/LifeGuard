import math
from src.hospital.location import get_hospital_location

# 두 좌표 간 거리 계산
def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# 거리 계산은 haversine_distance 방식으로 계산
# 시속 30km 일 때를 가정
def add_distance_features(df, user_lat, user_lon):

    distances = []
    times = []

    for _, row in df.iterrows():
        hospital_name = row["dutyname"]
        location = get_hospital_location(hospital_name)

        if not location:
            distances.append(None)
            times.append(None)
            continue

        dist = haversine_distance(
            user_lat,
            user_lon,
            location["lat"],
            location["lon"]
        )

        distances.append(round(dist, 2))
        times.append(round(dist * 2, 1))

    df["distance_km"] = distances
    df["estimated_travel_time_min"] = times

    return df