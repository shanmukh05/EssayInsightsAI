import os
import pytorch_lightning as pl

from analyzer.dataset import FeedbackPrizeDataModule
from analyzer.model import FeebackPrizeNetwork, FeedbackModel


def train_model(essay_df, config, label_info, output_folder):
    [label2id, id2label, num_classes] = label_info
    data = FeedbackPrizeDataModule(config, label2id, essay_df)
    data.prepare_data()

    data_split = config["data"]["strategy"]
    if data_split == "train_val":
        data.setup()

        model = FeedbackModel(config, num_classes, label2id, id2label)
        network = FeebackPrizeNetwork(model, config, num_classes)

        trainer = pl.Trainer(
            max_epochs=config["training"]["epochs"],
            accelerator="auto",
            devices=1,
            default_root_dir=os.path.join(output_folder, data_split),
        )
        trainer.fit(network, datamodule=data)
    elif data_split in ["kfold", "stratifiedkfold"]:
        num_folds = config["data"]["num_folds"]

        for k in range(num_folds):
            data.k = k
            data.setup()

            model = FeedbackModel(config, num_classes, label2id, id2label)
            network = FeebackPrizeNetwork(model, config, num_classes)

            trainer = pl.Trainer(
                max_epochs=config["training"]["epochs"],
                accelerator="auto",
                devices=1,
                default_root_dir=os.path.join(output_folder, f"{data_split}_{k+1}"),
            )
            trainer.fit(network, datamodule=data)

    return data, network, trainer


def load_saved_model(config, label_info, ckpt_path, output_folder):
    [label2id, id2label, num_classes] = label_info
    data = FeedbackPrizeDataModule(config, label2id)
    data.prepare_data()

    model = FeedbackModel(config, num_classes, label2id, id2label)
    network = FeebackPrizeNetwork.load_from_checkpoint(
        ckpt_path, model=model, config=config, num_classes=num_classes
    )

    trainer = pl.Trainer(default_root_dir=output_folder)

    return data, network, trainer


def load_saved_models(config, label_info, output_folder):
    ckpt_paths = config["postprocess"]["models"]
    [label2id, id2label, num_classes] = label_info
    data = FeedbackPrizeDataModule(config, label2id)
    data.prepare_data()

    trainer = pl.Trainer(default_root_dir=output_folder)

    network_ls = []
    for ckpt_path in ckpt_paths:
        model = FeedbackModel(config, num_classes, label2id, id2label)
        network = FeebackPrizeNetwork.load_from_checkpoint(
            ckpt_path, model=model, config=config, num_classes=num_classes
        )
        network_ls.append(network)

    return data, network_ls, trainer
