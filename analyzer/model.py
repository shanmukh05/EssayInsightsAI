import os
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class FeebackPrizeNetwork(pl.LightningModule):
    def __init__(self, model, config, num_classes):
        super().__init__()

        self.model = model
        self.config = config
        self.num_classes = num_classes
        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        ids = batch["input_ids"].to(dtype=torch.long)
        attn_mask = batch["attention_mask"].to(dtype=torch.long)
        labels = batch["labels"].to(dtype=torch.long)

        loss, logits = self.model(
            input_ids=ids, attention_mask=attn_mask, labels=labels
        )

        acc = self.calc_acc(labels, logits, self.train_acc)
        self.log("train_loss", loss, on_step=True)

        return {"loss": loss, "acc": acc}

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        ids = batch["input_ids"].to(dtype=torch.long)
        attn_mask = batch["attention_mask"].to(dtype=torch.long)
        labels = batch["labels"].to(dtype=torch.long)

        loss, logits = self.model(
            input_ids=ids, attention_mask=attn_mask, labels=labels
        )

        acc = self.calc_acc(labels, logits, self.val_acc)

        self.log("val_loss", loss, on_step=True)
        return {"loss": loss, "acc": acc}

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute())
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        ids = batch["input_ids"].to(dtype=torch.long)
        attn_mask = batch["attention_mask"].to(dtype=torch.long)

        logits = self.model(input_ids=ids, attention_mask=attn_mask)

        return logits

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(), lr=self.config["training"]["lr"]
        )

        scheduler = CosineAnnealingLR(optim, T_max=10, eta_min=1e-5)

        return {"optimizer": optim, "lr_scheduler": scheduler}
    
    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)

        return [early_stop]

    def calc_acc(self, labels, logits, acc_fn):
        labels = labels.view(-1)
        flat_logits = logits.view(-1, self.num_classes)

        preds = torch.argmax(flat_logits, axis=1)
        mask = labels != -100

        acc = acc_fn(
            torch.masked_select(preds, mask), torch.masked_select(labels, mask)
        )

        return acc


class FeedbackModel(nn.Module):
    def __init__(self, config, num_classes, label2id, id2label):
        super(FeedbackModel, self).__init__()

        self.model, config_model = self.init_model(
            config, num_classes, label2id, id2label
        )
        self.drop_out = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config_model.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        emb = self.model(input_ids)[0]
        preds1 = self.output(self.dropout1(emb))
        preds2 = self.output(self.dropout2(emb))
        preds3 = self.output(self.dropout3(emb))
        preds4 = self.output(self.dropout4(emb))
        preds5 = self.output(self.dropout5(emb))
        preds = (preds1 + preds2 + preds3 + preds4 + preds5) / 5

        logits = torch.softmax(preds, dim=-1)
        if labels is not None:
            loss = self.get_loss(preds, labels, attention_mask)
            return loss, logits
        else:
            return logits

    def init_model(self, config, num_classes, label2id, id2label):
        model_path = config["paths"]["model"]
        if os.path.exists(model_path):
            config_model = AutoConfig.from_pretrained(
                config["paths"]["model"] + "/config.json"
            )
            config_model.num_labels = num_classes
            config_model.id2label = id2label
            config_model.label2id = label2id
            model = AutoModel.from_pretrained(
                config["paths"]["model"],
                config=config_model,
            )
        else:
            config_model = AutoConfig.from_pretrained(config["paths"]["model"])
            model = AutoModel.from_pretrained(
                config["paths"]["model"],
                label2id=label2id,
                id2label=id2label,
                num_labels=num_classes,
            )

        return model, config_model

    def get_loss(self, outputs, targets, attention_mask):
        loss_fn = nn.CrossEntropyLoss()
        active_logits = outputs.reshape(-1, outputs.shape[-1])
        true_labels = targets.reshape(-1)

        idxs = attention_mask.reshape(-1) == 1
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = loss_fn(active_logits, true_labels)

        return loss
