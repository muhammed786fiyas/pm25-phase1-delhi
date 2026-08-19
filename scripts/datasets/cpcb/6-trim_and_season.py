import argparse
import os
import pandas as pd

WINDOW_START = "2025-03-01"
WINDOW_END = "2026-02-28"

SEASON_MONTHS = {
    3: "summer", 4: "summer", 5: "summer",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
    12: "winter", 1: "winter", 2: "winter",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help = "Combined CSV")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--config", default="configs/seasons.yaml")
    args = parser.parse_args()

    os.makedirs(args.outdir,exist_ok=True)

    df = pd.read_csv(args.input)
    df["dt_local"] = pd.to_datetime(df["datetime_from_local"])

    # TRIMMING
    before = len(df)
    df = df[(df["dt_local"] >= WINDOW_START) & (df["dt_local"] <= WINDOW_END + " 23:59:59")]
    print(f"Trimmed to {WINDOW_START} .. {WINDOW_END}: {before} -> {len(df)} rows "
          f"(dropped {before - len(df)})")

    df["month"] = df["dt_local"].dt.month
    df["season"] = df["month"].map(SEASON_MONTHS)

    # DAILY WINDOW FILTERING
    df = df[(df["dt_local"].dt.hour >=10) & (df["dt_local"].dt.hour <=13)]

    out_path = os.path.join(args.outdir ,"pm25_delhi_mar25_feb26_seasoned.csv")
    df.drop(columns=["dt_local","month"]).to_csv(out_path, index=False)
    print(f"wrote {out_path}\n")

    print("=== Rows and calendar span per season ===")
    for season in ["summer", "monsoon", "post_monsoon", "winter"]:
        sub = df[df["season"] == season]
        if len(sub) == 0:
            print(f"{season} 0 rows")
            continue
        print(f"{season:14s} {len(sub):>7} rows | {sub['dt_local'].min()} | {sub['dt_local'].max()} | {sub['location_id'].nunique()} stations")

    print("\n=== Rows per station per season (completeness check) ===")
    pivot = df.pivot_table(index= "location_id",
                           columns= "season",
                           values= "value",
                           aggfunc= "count",
                           fill_value= 0)
    pivot = pivot[["summer","monsoon","post_monsoon","winter"]]
    pivot_path = os.path.join(args.outdir,"rows_per_station_per_season.csv")
    pivot.to_csv(pivot_path)
    print(f"Wrote {pivot_path}")
    print(pivot)

if __name__ == "__main__":
    main()