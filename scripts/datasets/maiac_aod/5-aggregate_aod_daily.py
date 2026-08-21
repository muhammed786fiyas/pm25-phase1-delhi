import argparse
import os
import pandas as pd

# CREA/Kumar et al. 4-season scheme, same as used for CPCB - fixed definition, not tunable


def get_season(date):
    month = date.month
    if month in [3, 4, 5]:
        return "summer"
    if month in [6, 7, 8, 9]:
        return "monsoon"
    if month in [10, 11]:
        return "post_monsoon"
    return "winter"  # 12, 1, 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="maiac_aod_qa_filtered.csv")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, parse_dates=["date"])
    print(f"Loaded rows: {len(df)}")

    # output 1: daily aggregation - mean of valid overpasses per station-day
    daily_df = (
        df.groupby(["location_id", "date"])
        .agg(
            aod_055=("aod_055", "mean"),
            aod_047=("aod_047", "mean"),
            aod_uncertainty=("aod_uncertainty", "mean"),
            n_overpasses=("aod_055", "count"),
        )
        .reset_index()
    )

    print(f"Station-days after daily aggregation: {len(daily_df)}")
    print(f"Stations covered: {daily_df['location_id'].nunique()}")

    # add season column to the daily file itself, so it's available for EDA
    # without recomputing it later
    daily_df["season"] = daily_df["date"].apply(get_season)

    daily_path = os.path.join(args.outdir, "maiac_aod_daily.csv")
    daily_df.to_csv(daily_path, index=False)
    print(f"Wrote {daily_path}")

    # output 2: per-station-per-season row counts, same shape as CPCB's
    # rows_per_station_per_season.csv, for comparison
    season_counts = (
        daily_df.groupby(["location_id", "season"])
        .size()
        .reset_index(name="n_days")
    )
    season_pivot = season_counts.pivot(
        index="location_id", columns="season", values="n_days"
    ).fillna(0).astype(int)

    season_pivot["total"] = season_pivot.sum(axis=1)

    season_path = os.path.join(args.outdir, "rows_per_station_per_season_aod.csv")
    season_pivot.to_csv(season_path)
    print(f"Wrote {season_path}")

    # output 3: n_overpasses summary - how many overpasses typically went
    # into each daily average, so this can inform a future minimum-overpass
    # decision without having thrown away the information now
    overpass_summary = (
        daily_df["n_overpasses"].value_counts().sort_index().reset_index()
    )
    overpass_summary.columns = ["n_overpasses", "n_station_days"]

    overpass_path = os.path.join(args.outdir, "n_overpasses_summary.csv")
    overpass_summary.to_csv(overpass_path, index=False)
    print(f"Wrote {overpass_path}")

    print("\nn_overpasses distribution:")
    print(overpass_summary.to_string(index=False))


if __name__ == "__main__":
    main()