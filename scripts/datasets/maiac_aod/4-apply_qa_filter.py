import argparse
import os
import yaml
import pandas as pd

# qa_aod code meanings, from NASA MCD19A2 User Guide V61, Table 5.4
# which codes are kept for each strictness level is a judgment call,
# made by a human after reviewing qa_summary.csv - not decided by this script

STRICTNESS_ALLOWED_CODES = {
    "strict": [0],              # Best quality only
    "moderate": [0, 3],         # Best quality + 1 neighbor cloud
    "lenient": [0, 1, 3, 4, 7, 10],  # everything except no-retrieval/glint/research-quality codes
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="maiac_aod_qa_decoded.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)["aod_qa_filter"]

    strictness = params["strictness"]

    # guard: stop the pipeline here if no real decision has been made yet
    if strictness == "NOT_SET":
        raise ValueError(
            "aod_qa_filter.strictness is still NOT_SET in params.yaml. "
            "Review data/interim/maiac_aod/qa_decode/qa_summary.csv, "
            "pick a strictness level ('strict', 'moderate', or 'lenient'), "
            "set it in params.yaml, then rerun."
        )

    if strictness not in STRICTNESS_ALLOWED_CODES:
        raise ValueError(
            f"Unknown strictness '{strictness}'. "
            f"Must be one of: {list(STRICTNESS_ALLOWED_CODES.keys())}"
        )

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded rows: {len(df)}")
    print(f"Strictness setting: {strictness}")

    allowed_codes = STRICTNESS_ALLOWED_CODES[strictness]
    keep_mask = df["qa_aod"].isin(allowed_codes)

    kept_df = df[keep_mask].copy()
    dropped_df = df[~keep_mask].copy()

    print(f"Rows kept: {len(kept_df)} ({100 * len(kept_df) / len(df):.1f}%)")
    print(f"Rows dropped: {len(dropped_df)} ({100 * len(dropped_df) / len(df):.1f}%)")

    if len(dropped_df) > 0:
        print("\nDropped rows by qa_aod_label:")
        print(dropped_df["qa_aod_label"].value_counts())

    # check no station lost entirely
    stations_before = df["location_id"].nunique()
    stations_after = kept_df["location_id"].nunique()
    print(f"\nStations before: {stations_before}, after: {stations_after}")
    if stations_after < stations_before:
        print("WARNING: at least one station has zero rows left after filtering")

    out_path = os.path.join(args.outdir, "maiac_aod_qa_filtered.csv")
    kept_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()