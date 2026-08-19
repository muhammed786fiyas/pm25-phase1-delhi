import argparse
import os
import math
import itertools
import pandas as pd

DISTANCE_LIMIT_KM = 1.5

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    distance = 2 * R * math.asin(math.sqrt(a))
    return distance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", required=True, help="Station metadata CSV")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.stations)
    df = df[df["status"] == "KEEP"]
    print(f"Checking {len(df)} KEEP stations")

    close_pairs = []

    for (i, row_a), (j, row_b) in itertools.combinations(df.iterrows(), 2):
        dist = haversine_km(row_a["latitude"], row_a["longitude"],
                             row_b["latitude"], row_b["longitude"])

        if dist < DISTANCE_LIMIT_KM:
            close_pairs.append({
                "location_id_a": row_a["location_id"],
                "name_a": row_a["name"],
                "location_id_b": row_b["location_id"],
                "name_b": row_b["name"],
                "distance_km": round(dist, 3),
            })

    print(f"Found {len(close_pairs)} pairs under {DISTANCE_LIMIT_KM} km")

    result = pd.DataFrame(close_pairs)
    result = result.sort_values("distance_km").reset_index(drop=True)

    # assign a cluster_id: pairs that share a station join the same cluster
    station_to_cluster = {}
    next_cluster_id = 1

    for i, row in result.iterrows():
        a = row["location_id_a"]
        b = row["location_id_b"]

        a_has_cluster = a in station_to_cluster
        b_has_cluster = b in station_to_cluster

        if not a_has_cluster and not b_has_cluster:
            station_to_cluster[a] = next_cluster_id
            station_to_cluster[b] = next_cluster_id
            next_cluster_id = next_cluster_id + 1
        elif a_has_cluster and not b_has_cluster:
            station_to_cluster[b] = station_to_cluster[a]
        elif b_has_cluster and not a_has_cluster:
            station_to_cluster[a] = station_to_cluster[b]
        elif station_to_cluster[a] != station_to_cluster[b]:
            print(f"warning: {a} and {b} are in different clusters, not merging automatically")

    result["cluster_id"] = result["location_id_a"].map(station_to_cluster)

    out_path = os.path.join(args.outdir, "station_clusters.csv")
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(result)

if __name__ == "__main__":
    main()