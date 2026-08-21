import argparse
import os
import pandas as pd

# Bit layout from NASA MCD19A2 User Guide V61, Table 5.4 (AOD_QA, 16-bit)
# These are fixed spec values, not tunable - not put in params.yaml

CLOUD_MASK_LABELS = {
    0: "Undefined",
    1: "Clear",
    2: "Possibly Cloudy",
    3: "Cloudy",
    5: "Cloud Shadow",
    6: "Hot spot of fire",
    7: "Water Sediments",
}

ADJACENCY_MASK_LABELS = {
    0: "Normal/Clear",
    1: "Adjacent to clouds",
    2: "Surrounded by >4 cloudy pixels",
    3: "Adjacent to a single cloudy pixel",
    4: "Adjacent to snow",
    5: "Snow previously detected",
}

QA_AOD_LABELS = {
    0: "Best quality",
    1: "Water Sediments detected",
    3: "1 neighbor cloud",
    4: ">1 neighbor clouds",
    5: "No retrieval",
    6: "No retrieval near snow",
    7: "Climatology AOD (high altitude)",
    8: "No retrieval due to sun glint",
    9: "Very low AOD due to glint",
    10: "AOD replaced near coastline",
    11: "Research quality, possibly cloudy",
}

AEROSOL_MODEL_LABELS = {
    0: "Background",
    1: "Smoke",
    2: "Dust",
}


def decode_qa(qa_value):
    cloud_mask = qa_value & 0b111
    adjacency_mask = (qa_value >> 5) & 0b111
    qa_aod = (qa_value >> 8) & 0b1111
    glint_mask = (qa_value >> 12) & 0b1
    aerosol_model = (qa_value >> 13) & 0b11
    return cloud_mask, adjacency_mask, qa_aod, glint_mask, aerosol_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="maiac_aod_filtered_scaled.csv")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded rows: {len(df)}")

    cloud_mask_list = []
    adjacency_mask_list = []
    qa_aod_list = []
    glint_mask_list = []
    aerosol_model_list = []

    for qa_value in df["aod_qa"]:
        cloud_mask, adjacency_mask, qa_aod, glint_mask, aerosol_model = decode_qa(int(qa_value))
        cloud_mask_list.append(cloud_mask)
        adjacency_mask_list.append(adjacency_mask)
        qa_aod_list.append(qa_aod)
        glint_mask_list.append(glint_mask)
        aerosol_model_list.append(aerosol_model)

    df["cloud_mask"] = cloud_mask_list
    df["adjacency_mask"] = adjacency_mask_list
    df["qa_aod"] = qa_aod_list
    df["glint_mask"] = glint_mask_list
    df["aerosol_model"] = aerosol_model_list

    df["cloud_mask_label"] = df["cloud_mask"].map(CLOUD_MASK_LABELS)
    df["adjacency_mask_label"] = df["adjacency_mask"].map(ADJACENCY_MASK_LABELS)
    df["qa_aod_label"] = df["qa_aod"].map(QA_AOD_LABELS)
    df["aerosol_model_label"] = df["aerosol_model"].map(AEROSOL_MODEL_LABELS)

    # print summary counts, so a human can decide the filtering strictness later
    print("\nCloud Mask counts:")
    print(df["cloud_mask_label"].value_counts())

    print("\nAdjacency Mask counts:")
    print(df["adjacency_mask_label"].value_counts())

    print("\nQA for AOD counts:")
    print(df["qa_aod_label"].value_counts())

    print("\nGlint Mask counts:")
    print(df["glint_mask"].value_counts())

    print("\nAerosol Model counts:")
    print(df["aerosol_model_label"].value_counts())

    # save full decoded output
    out_path = os.path.join(args.outdir, "maiac_aod_qa_decoded.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # save a separate summary CSV, easier to review than scrolling terminal output
    summary_rows = []
    for label, count in df["qa_aod_label"].value_counts().items():
        summary_rows.append({"field": "qa_aod", "category": label, "count": count})
    for label, count in df["cloud_mask_label"].value_counts().items():
        summary_rows.append({"field": "cloud_mask", "category": label, "count": count})
    for label, count in df["adjacency_mask_label"].value_counts().items():
        summary_rows.append({"field": "adjacency_mask", "category": label, "count": count})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, "qa_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()