import os
import sys
import yaml
import logging
import datetime
import pandas as pd
from tqdm import tqdm


def preprocess_labels(label_df, essay_df, datapath):
    annot_path = os.path.join(datapath, "train_essays_annotated.csv")
    if os.path.exists(annot_path):
        essay_df = pd.read_csv(annot_path).iloc[:20]
    else:
        predictions = []
        for _, row in tqdm(essay_df.iterrows()):
            seq_len = len(row["text"].split())
            preds = ["O"] * seq_len
            for _, det_row in label_df[label_df["id"] == row["id"]].iterrows():
                string = [int(k) for k in det_row["predictionstring"].split()]
                discourse_type = det_row["discourse_type"]

                preds[string[0]] = f"B-{discourse_type}"
                for id_ in string[1:]:
                    preds[id_] = f"I-{discourse_type}"
            predictions.append(preds)

        essay_df["prediction"] = predictions
        essay_df.to_csv(annot_path, index=False)

    unq_discourses = label_df["discourse_type"].unique()
    unq_labels = (
        ["O"] + [f"B-{i}" for i in unq_discourses] + [f"I-{i}" for i in unq_discourses]
    )

    label2id = {k: i for i, k in enumerate(unq_labels)}
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)

    return essay_df, [label2id, id2label, num_classes]


def prepare_test_data(datapath):
    test_paths = os.listdir(os.path.join(datapath, "test"))
    test_texts = [
        open(os.path.join(datapath, "test", f), "r").read() for f in test_paths
    ]
    test_ids = [i.split("/")[-1].replace(".txt", "") for i in test_paths]
    test_df = pd.DataFrame.from_dict({"id": test_ids, "text": test_texts})

    return test_df


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)

    logging.info("Config File Loaded")
    return config_dict


def get_logger(log_folder):
    os.makedirs(log_folder, exist_ok=True)
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d-%H.%M.%S")

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(log_folder, f"logs-{timestamp}.txt"),
        filemode="w",
        format="%(asctime)-8s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s",
    )
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
    )
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
