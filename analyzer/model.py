import os
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


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
            self.model.parameters(), lr=self.config["training"]["lr"], weight_decay=0.01
        )

        # scheduler = CosineAnnealingLR(optim, T_max=10, eta_min=1e-7)
        scheduler = MultiStepLR(optim, milestones=[2, 4, 6, 8], gamma=0.2)

        return {"optimizer": optim, "lr_scheduler": scheduler}

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
        model_ckpt = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best_model_{epoch:02d}-{val_loss:.3f}",
            save_weights_only=True,
        )

        return [early_stop, model_ckpt]

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

        self.backbone, config_backbone = self._init_backbone(
            config, num_classes, label2id, id2label
        )
        self._freeze_layers()
        self.num_classes = num_classes

        self.dropout1 = nn.Dropout(0.3)

        input_size = config_backbone.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, num_classes),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        pretrain_output = self.backbone(input_ids, attention_mask=attention_mask)
        embeds = pretrain_output.last_hidden_state

        output = self.dropout1(embeds)
        preds = self.classifier(output)

        logits = torch.softmax(preds, dim=-1)
        if labels is not None:
            loss = self.get_loss(preds, labels, attention_mask)
            return loss, logits
        else:
            return logits

    def get_loss(self, outputs, targets, attention_mask):
        outputs = outputs.view(-1, self.num_classes)
        targets = targets.view(-1)
        attention_mask = attention_mask.view(-1)

        mask = attention_mask != 0
        outputs = outputs[mask]
        targets = targets[mask]

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, targets)

        return loss

    def _init_backbone(self, config, num_classes, label2id, id2label):
        path = config["paths"]["model"]
        if os.path.exists(path):
            config_backbone = AutoConfig.from_pretrained(
                config["paths"]["model"] + "/config.json"
            )
            config_backbone.num_labels = num_classes
            config_backbone.id2label = id2label
            config_backbone.label2id = label2id
            backbone = AutoModel.from_pretrained(
                config["paths"]["model"],
                config=config_backbone,
            )
        else:
            config_backbone = AutoConfig.from_pretrained(config["paths"]["model"])
            backbone = AutoModel.from_pretrained(
                config["paths"]["model"],
                label2id=label2id,
                id2label=id2label,
                num_labels=num_classes,
            )

        return backbone, config_backbone

    def _freeze_layers(self):
        backbone_name = self.backbone.__class__.__name__

        if backbone_name == "FunnelModel":
            self.backbone.embeddings.requires_grad_(False)
            self.backbone.encoder.attention_structure.requires_grad_(False)
            self.backbone.encoder.blocks[0:2].requires_grad_(False)
        elif backbone_name == "DistilBertModel":
            return
        else:
            self.backbone.embeddings.requires_grad_(False)
            self.backbone.encoder.layer[:20].requires_grad_(False)
