import pandas as pd
from huggingface_hub import hf_hub_download

from analyzer.train import load_saved_model
from analyzer.infer import inference

label2id = {
    "O": 0,
    "B-Lead": 1,
    "I-Lead": 2,
    "B-Position": 3,
    "I-Position": 4,
    "B-Evidence": 5,
    "I-Evidence": 6,
    "B-Claim": 7,
    "I-Claim": 8,
    "B-Concluding Statement": 9,
    "I-Concluding Statement": 10,
    "B-Counterclaim": 11,
    "I-Counterclaim": 12,
    "B-Rebuttal": 13,
    "I-Rebuttal": 14,
}
num_classes = len(label2id)
id2label = {v: k for k, v in label2id.items()}

config_dict = {
    "paths": {
        "tokenizer": "distilbert/distilbert-base-uncased",
        "model": "distilbert/distilbert-base-uncased",
    },
    "data": {"max_len": 512, "strategy": "train_val"},
    "test_ds": {
        "batch_size": 2,
        "shuffle": False,
        "num_workers": 1,
        "pin_memory": False,
    },
}

color_map = {
    "Claim": "red",
    "Evidence": "green",
    "Lead": "blue",
    "Position": "orange",
    "Counterclaim": "purple",
    "Rebuttal": "teal",
    "ConcludingStatement": "brown",
    "None": "gray",
}

legend_html = "<h6>Color Mapping:</h6><ul>"
for cls, color in color_map.items():
    legend_html += f"<li><span style='display:inline-block;width:12px;height:12px;background-color:{color};border-radius:50%;margin-right:6px;'></span>{cls}</li>"
legend_html += "</ul>"


def analyze_essay(input_text):
    ckpt_path = hf_hub_download(
        repo_id="aine05/EssayInsightsAI-DistilBERT",
        filename="train_val/lightning_logs/version_0/checkpoints/epoch=4-step=1100.ckpt",
        repo_type="model",
    )

    data, network, trainer = load_saved_model(
        config_dict,
        [label2id, id2label, num_classes],
        ckpt_path,
        "logs",
    )
    df = pd.DataFrame.from_dict({"id": ["UserInput"], "text": [input_text]})
    data_loader = data.test_dataloader(df)
    predictions = inference(data_loader, network, trainer, id2label)[0]

    words = input_text.split()
    predictions = [
        i[2:].replace(" ", "") if "I-" in i else i.replace(" ", "") for i in predictions
    ]
    predictions = [i if i != "O" else "None" for i in predictions]

    # words = ["According", "to", "experts,", "global", "warming", "is", "not", "universally", "agreed", "upon."]
    # predictions = ["Lead", "Lead", "Evidence", "Claim", "Claim", "Counterclaim", "Counterclaim", "Rebuttal", "Rebuttal", "Concluding"]

    colored_text = " ".join(
        [
            f'<span style="color:{color_map.get(cls, "gray")};">{word}</span>'
            for word, cls in zip(words, predictions)
        ]
    )

    final_text = legend_html + "<h4> Segmented Essay: </h4>" + f"<p>{colored_text}</p>"

    return final_text, predictions
