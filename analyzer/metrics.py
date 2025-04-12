def measure_overlap(pred, true):
    pred = set(pred.split())
    true = set(true.split())

    num_pred, num_true = len(pred), len(true)
    if len(pred) == 0 or len(true) == 0:
        return 0.0, 0.0

    num_overlap = len(true.intersection(pred))

    return num_overlap / num_true, num_overlap / num_pred


def competition_metric(pred_df, true_df):
    true_df = true_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df = pred_df.reset_index(drop=True).copy()

    pred_df["pred_id"] = pred_df.index
    true_df["true_id"] = true_df.index

    comb_df = pred_df.merge(
        true_df,
        left_on=["id", "class"],
        right_on=["id", "class"],
        how="outer",
        suffixes=("_pred", "_true"),
    )
    comb_df["predictionstring_true"] = comb_df["predictionstring_true"].fillna(" ")
    comb_df["predictionstring_pred"] = comb_df["predictionstring_pred"].fillna(" ")

    comb_df[["overlap_true", "overlap_pred"]] = comb_df.apply(
        lambda row: measure_overlap(
            row.predictionstring_pred, row.predictionstring_true
        ),
        axis=1,
        result_type="expand",
    )
    comb_df["overlap_max"] = comb_df[["overlap_true", "overlap_pred"]].max(axis=1)
    comb_df["TP"] = (comb_df["overlap_true"] >= 0.5) & (comb_df["overlap_pred"] >= 0.5)

    tp_pred_ids = (
        comb_df.query("TP")
        .sort_values("overlap_max", ascending=False)
        .groupby(["id", "predictionstring_true"])
        .first()["pred_id"]
        .values
    )

    fp_pred_ids = [p for p in comb_df["pred_id"].unique() if p not in tp_pred_ids]
    matched_true_ids = comb_df.query("TP")["true_id"].unique()
    unmatched_true_ids = [
        c for c in comb_df["true_id"].unique() if c not in matched_true_ids
    ]

    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_true_ids)
    f1_score = TP / (TP + 0.5 * (FP + FN))

    return f1_score
