import torch
import pandas as pd
from collections import Counter
from scipy.cluster.hierarchy import fcluster, linkage

from infer import logits2pred, submission


def post_process(test_df, config, dataloader, network_ls, trainer, id2label):
    operations = config["postprocess"]["operations"]
    logits_ls = []

    word_ids = []
    for batch in dataloader:
        word_ids.extend(batch["word_ids"].cpu().numpy())

    for network in network_ls:
        logits = trainer.predict(network, dataloaders=dataloader)
        logits_ls.append(torch.concat(logits, axis=0))

    if "soft_ensemble" in operations:
        logits = torch.mean(torch.stack(logits_ls, axis=0), axis=0)
        predictions = logits2pred(logits, id2label, word_ids)
        sub_df = submission(test_df, predictions)

    elif "hard_ensemble" in operations:
        predictions_ls = [
            logits2pred(logits, id2label, word_ids) for logits in logits_ls
        ]
        predictions = []
        num_samples = len(predictions_ls[0])

        for i in range(num_samples):
            preds_i = [predictions_m[i] for predictions_m in predictions_ls]
            preds = hard_voting(preds_i)
            predictions.append(preds)
        sub_df = submission(test_df, predictions)

    elif "bound_average" in operations:
        predictions_ls = [
            logits2pred(logits, id2label, word_ids) for logits in logits_ls
        ]
        sub_df_ls = [submission(test_df, predictions) for predictions in predictions_ls]
        if "repair_spans" in operations:
            sub_df_ls = [
                sub_df.groupby(["id", "class"])
                .apply(combine_broken_spans)
                .reset_index(name="predictionstring")
                for sub_df in sub_df_ls
            ]
        sub_df = bound_average(sub_df_ls)

    sub_df = (
        sub_df.groupby(["id", "class"])
        .apply(combine_broken_spans)
        .reset_index(name="predictionstring")
    )
    sub_df = merge_common_spans(sub_df)

    return sub_df


def hard_voting(data):
    max_len = max(len(sublist) for sublist in data)
    final_predictions = []

    for pos in range(max_len):
        elements = [sublist[pos] for sublist in data if pos < len(sublist)]
        counts = Counter(elements)
        most_common = counts.most_common(1)[0][0]
        final_predictions.append(most_common)

    return final_predictions


def combine_broken_spans(group):
    merged = []
    current_preds = []

    for _, row in group.iterrows():
        preds = list(map(int, row["predictionstring"].split()))
        start, end = preds[0], preds[-1]

        if not current_preds:
            current_preds = preds
        elif start - current_preds[-1] <= 20:
            current_preds.extend(preds)
        else:
            merged.append(" ".join(map(str, current_preds)))
            current_preds = preds

    if current_preds:
        merged.append(" ".join(map(str, current_preds)))

    return " ".join(merged)


def bound_average(sub_df_ls):
    def get_start_end(prediction_string):
        tokens = list(map(int, prediction_string.split()))
        return tokens[0], tokens[-1]

    for df in sub_df_ls:
        df[["start", "end"]] = df["predictionstring"].apply(
            lambda x: pd.Series(get_start_end(x))
        )

    combined_df = pd.concat(sub_df_ls).reset_index(drop=True)

    def cluster_rows(group, threshold):
        if len(group) == 1:
            group["cluster"] = 1
            return group

        positions = group[["start", "end"]].values
        if positions.shape[0] < 2:
            return group

        linkage_matrix = linkage(positions, method="single")
        group["cluster"] = fcluster(linkage_matrix, t=threshold, criterion="distance")
        return group

    clustered_df = combined_df.groupby(["id", "class"], group_keys=False).apply(
        lambda group: cluster_rows(group, threshold=30)
    )
    clustered_avg_df = (
        clustered_df.groupby(["id", "class", "cluster"])
        .agg({"start": "mean", "end": "mean"})
        .round()
        .astype(int)
        .reset_index()
    )

    clustered_avg_df["predictionstring"] = clustered_avg_df.apply(
        lambda row: " ".join(map(str, range(row["start"], row["end"] + 1))), axis=1
    )
    final_clustered_df = clustered_avg_df[["id", "class", "predictionstring"]]

    return final_clustered_df


def merge_common_spans(df):
    def get_token_set(prediction_string):
        return set(map(int, prediction_string.split()))

    df["token_set"] = df["predictionstring"].apply(get_token_set)

    def merge_subsets(group):
        unique_rows = []
        for i, row_i in group.iterrows():
            is_subset = False
            for j, row_j in group.iterrows():
                if i != j and row_i["token_set"].issubset(row_j["token_set"]):
                    is_subset = True
                    break
            if not is_subset:
                unique_rows.append(row_i)
        return pd.DataFrame(unique_rows)

    merged_df = (
        df.groupby(["id", "class"], group_keys=False)
        .apply(merge_subsets)
        .reset_index(drop=True)
    )
    merged_df.drop(columns=["token_set"], inplace=True)

    return merged_df
