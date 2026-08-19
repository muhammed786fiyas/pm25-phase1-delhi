import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", required=True, help="Daily aggregated PM2.5 CSV")
    parser.add_argument("--clusters", required=True, help="station_clusters.csv from find_station_clusters.py")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    daily = pd.read_csv(args.daily)
    pairs = pd.read_csv(args.clusters)
    print(f"Loaded {len(daily)} station-days, {len(pairs)} station pairs to check")

    results = []

    for i, row in pairs.iterrows():
        station_a = row["location_id_a"]
        station_b = row["location_id_b"]

        data_a = daily[daily["location_id"] == station_a][["date", "pm25_daily"]]
        data_b = daily[daily["location_id"] == station_b][["date", "pm25_daily"]]

        data_a = data_a.rename(columns={"pm25_daily": "value_a"})
        data_b = data_b.rename(columns={"pm25_daily": "value_b"})

        merged = pd.merge(data_a, data_b, on="date")

        matched_days = len(merged)

        if matched_days < 2:
            print(f"{station_a} vs {station_b}: only {matched_days} matched days, skipping")
            continue

        correlation = merged["value_a"].corr(merged["value_b"])

        print(f"{station_a} vs {station_b}: correlation {round(correlation, 3)}, {matched_days} matched days")

        results.append({
            "location_id_a": station_a,
            "name_a": row["name_a"],
            "location_id_b": station_b,
            "name_b": row["name_b"],
            "distance_km": row["distance_km"],
            "cluster_id": row["cluster_id"],
            "matched_days": matched_days,
            "correlation": round(correlation, 3),
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("correlation")

    out_path = os.path.join(args.outdir, "station_pair_correlation.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(result_df)

if __name__ == "__main__":
    main()