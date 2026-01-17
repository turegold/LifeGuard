def merge_rank_with_payloads(recommendations, hospital_payloads):

    payload_map = {
        p["meta"]["hospital_id"]: p
        for p in hospital_payloads
    }

    merged = []

    for rank, rec in enumerate(recommendations, start=1):
        hid = rec.get("hospital_id")
        payload = payload_map.get(hid)

        if not payload:
            continue

        merged.append({
            "rank": rank,
            "accept_prob": rec.get("accept_prob"),
            "meta": payload["meta"],
            "features": payload["features"],
        })

    return merged