import os
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModelForTokenClassification

from dataset import FeedbackPrizeDataModule
from model import FeebackPrizeNetwork


def train_model(essay_df, config, label_info, output_folder):
    [label2id, id2label, num_classes] = label_info
    data = FeedbackPrizeDataModule(config, label2id, essay_df)
    data.prepare_data()
    data.setup()

    model_path = config["paths"]["model"]
    if os.path.exists(model_path):
        config_model = AutoConfig.from_pretrained(
            config["paths"]["model"] + "/config.json"
        )
        config_model.num_labels = num_classes
        config_model.id2label = id2label
        config_model.label2id = label2id
        model = AutoModelForTokenClassification.from_pretrained(
            config["paths"]["model"],
            config=config_model,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            config["paths"]["model"],
            label2id=label2id,
            id2label=id2label,
            num_labels=num_classes,
        )

    network = FeebackPrizeNetwork(model, config, num_classes)

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="auto",
        devices=1,
        default_root_dir=output_folder,
    )
    trainer.fit(network, datamodule=data)

    return data, network, trainer


def load_saved_model(config, label_info, ckpt_path, output_folder):
    [label2id, id2label, num_classes] = label_info
    data = FeedbackPrizeDataModule(config, label2id)
    data.prepare_data()

    model_path = config["paths"]["model"]
    if os.path.exists(model_path):
        config_model = AutoConfig.from_pretrained(
            config["paths"]["model"] + "/config.json"
        )
        config_model.num_labels = num_classes
        config_model.id2label = id2label
        config_model.label2id = label2id
        model = AutoModelForTokenClassification.from_pretrained(
            config["paths"]["model"],
            config=config_model,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            config["paths"]["model"],
            label2id=label2id,
            id2label=id2label,
            num_labels=num_classes,
        )

    network = FeebackPrizeNetwork.load_from_checkpoint(
        ckpt_path, model=model, config=config, num_classes=num_classes
    )

    trainer = pl.Trainer(default_root_dir=output_folder)

    return data, network, trainer
