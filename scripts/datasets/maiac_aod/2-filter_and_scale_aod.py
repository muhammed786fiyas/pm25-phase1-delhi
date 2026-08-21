import argparse
import os
import yaml
import pandas as pd

FILL_VALUE = -28672


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw maiac_aod_raw.csv")
    parser.add_argument("--stations", required=True, help="finalized station status CSV")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)["aod_filter_scale"]

    scale_aod = params["scale_aod"]
    scale_uncertainty = params["scale_uncertainty"]
    aod_valid_min = params["aod_valid_min"]
    aod_valid_max = params["aod_valid_max"]
    uncertainty_valid_min = params["uncertainty_valid_min"]
    uncertainty_valid_max = params["uncertainty_valid_max"]

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded raw AOD rows: {len(df)}")

    # step 1: filter to finalized KEEP stations only
    stations_df = pd.read_csv(args.stations)
    keep_ids = stations_df.loc[stations_df["status"] == "KEEP", "location_id"]
    df = df[df["location_id"].isin(keep_ids)].copy()
    print(f"Rows after filtering to finalized stations: {len(df)}")
    print(f"Stations covered: {df['location_id'].nunique()}")

    # step 2: replace fill values with NaN before scaling
    fill_cols = ["aod_055", "aod_047", "aod_uncertainty"]
    for col in fill_cols:
        fill_count = (df[col] == FILL_VALUE).sum()
        if fill_count > 0:
            print(f"{col}: {fill_count} fill values found, set to NaN")
        df[col] = df[col].replace(FILL_VALUE, pd.NA)

    # step 3: apply scale factors
    df["aod_055"] = df["aod_055"] * scale_aod
    df["aod_047"] = df["aod_047"] * scale_aod
    df["aod_uncertainty"] = df["aod_uncertainty"] * scale_uncertainty

    # step 4: duplicate check
    dupe_mask = df.duplicated(subset=["location_id", "image_id"], keep=False)
    dupe_count = dupe_mask.sum()
    if dupe_count > 0:
        print(f"WARNING: {dupe_count} duplicate rows found (same location_id + image_id)")
    else:
        print("No duplicates found")

    # step 5: range check (on scaled values)
    aod_out_of_range = (
        (df["aod_055"] < aod_valid_min) | (df["aod_055"] > aod_valid_max)
        | (df["aod_047"] < aod_valid_min) | (df["aod_047"] > aod_valid_max)
    )
    aod_out_count = aod_out_of_range.sum()
    print(f"AOD values out of valid range [{aod_valid_min}, {aod_valid_max}]: {aod_out_count}")

    uncertainty_out_of_range = (
        (df["aod_uncertainty"] < uncertainty_valid_min)
        | (df["aod_uncertainty"] > uncertainty_valid_max)
    )
    uncertainty_out_count = uncertainty_out_of_range.sum()
    print(f"Uncertainty values out of valid range [{uncertainty_valid_min}, {uncertainty_valid_max}]: {uncertainty_out_count}")

    # step 6: save output
    out_path = os.path.join(args.outdir, "maiac_aod_filtered_scaled.csv")
    df.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Final row count: {len(df)}")
    print(f"aod_055 range: {df['aod_055'].min():.3f} to {df['aod_055'].max():.3f}")
    print(f"aod_047 range: {df['aod_047'].min():.3f} to {df['aod_047'].max():.3f}")


if __name__ == "__main__":
    main()