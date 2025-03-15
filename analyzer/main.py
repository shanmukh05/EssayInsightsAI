import os
import glob
import json
import argparse
import pandas as pd

from utils import load_config, get_logger, preprocess_labels, prepare_test_data
from train import train_model, load_saved_model, load_saved_models
from infer import submission, inference
from postprocess import post_submission

parser = argparse.ArgumentParser()
parser.add_argument(
    "-C", "--config_path", type=str, required=True, help="Path to Config File"
)
parser.add_argument(
    "-O", "--output_folder", type=str, required=True, help="Path to Log Folder"
)
parser.add_argument(
    "-T",
    "--task",
    type=str,
    default=False,
    required=True,
    help="Inference or Training or Postprocessing",
    choices=["Train", "Inference", "Postprocess"],
)
parser.add_argument(
    "-P", "--ckpt_path", type=str, required=False, help="Path to Model Checkpoint"
)

args = parser.parse_args()


def main():
    # Initializing log folder
    output_folder = args.output_folder
    get_logger(output_folder)

    # Loading config into a dictionary
    config_path = args.config_path
    config = load_config(config_path)

    task = args.task
    datapath = config["paths"]["data"]

    if task == "Train":
        # Load data
        essay_df = pd.read_csv(os.path.join(datapath, "train_essay.csv"))
        label_df = pd.read_csv(os.path.join(datapath, "train_labels.csv"))
        essay_df, label_info = preprocess_labels(label_df, essay_df, datapath)
        [label2id, id2label, num_classes] = label_info

        # Train model
        _, _, _ = train_model(essay_df, config, label_info, output_folder)
    elif task == "Inference":
        # Load Model
        with open(os.path.join(datapath, "label2id.json"), "r") as f:
            label2id = json.load(f)
        num_classes = len(label2id)
        id2label = {v: k for k, v in label2id.items()}
        data, network, trainer = load_saved_model(
            config, [label2id, id2label, num_classes], args.ckpt_path, output_folder
        )

        # Inference
        test_df = prepare_test_data(datapath)
        test_loader = data.test_dataloader(test_df)
        predictions = inference(test_loader, network, trainer, id2label)
        sub_df = submission(test_df, predictions)
        sub_df.to_csv(os.path.join(output_folder, "submission.csv"), index=False)
    elif task == "Postprocess":
        # Load Model
        with open(os.path.join(datapath, "label2id.json"), "r") as f:
            label2id = json.load(f)
        num_classes = len(label2id)
        id2label = {v: k for k, v in label2id.items()}
        data, network_ls, trainer = load_saved_model(
            config, [label2id, id2label, num_classes], args.ckpt_path, output_folder
        )

        # Inference
        test_df = prepare_test_data(datapath)
        test_loader = data.test_dataloader(test_df)

        sub_df = post_submission(test_df, test_loader, network_ls, trainer, id2label)
        sub_df.to_csv(os.path.join(output_folder, "submission.csv"), index=False)


if __name__ == "__main__":
    # python main.py -C "D:\Learning\NLP\Projects\EssayInsightAI\analyzer\configs\config.yaml" -O "D:\Learning\NLP\Projects\EssayInsightAI\analyzer\output\distilbert" -T "Train"
    # python main.py -C "D:\Learning\NLP\Projects\EssayInsightAI\analyzer\configs\config.yaml" -O "D:\Learning\NLP\Projects\EssayInsightAI\analyzer\output\distilbert" -T "Inference" -P "D:\Learning\NLP\Projects\EssayInsightAI\analyzer\output\distilbert\train_val\lightning_logs\version_0\checkpoints\epoch=9-step=50.ckpt"
    main()
