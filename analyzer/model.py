import torch
import torchmetrics
import pytorch_lightning as pl


class FeebackPrizeNetwork(pl.LightningModule):
    def __init__(self, model, config, num_classes):
        super().__init__()
        
        self.model = model
        self.config = config
        self.num_classes=num_classes
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)

    def training_step(self, batch, batch_idx):
        ids = batch["input_ids"].to(dtype=torch.long)
        attn_mask = batch["attention_mask"].to(dtype=torch.long)
        labels = batch["labels"].to(dtype=torch.long)

        loss, logits = self.model(input_ids=ids, attention_mask=attn_mask, labels=labels, return_dict=False)

        acc = self.calc_acc(labels, logits, self.train_acc)

        return {"loss": loss, "acc": acc}

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        ids = batch["input_ids"].to(dtype=torch.long)
        attn_mask = batch["attention_mask"].to(dtype=torch.long)
        labels = batch["labels"].to(dtype=torch.long)

        loss, logits = self.model(input_ids=ids, attention_mask=attn_mask, labels=labels, return_dict=False)

        acc = self.calc_acc(labels, logits, self.val_acc)

        return {"loss": loss, "acc": acc}

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        ids = batch["input_ids"].to(dtype=torch.long)
        attn_mask = batch["attention_mask"].to(dtype=torch.long)

        logits = self.model(input_ids=ids, attention_mask=attn_mask, return_dict=False)[0]

        return logits

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr = self.config["training"]["lr"]
        )

        return optim

    def calc_acc(self, labels, logits, acc_fn):
        labels = labels.view(-1)
        flat_logits = logits.view(-1, self.num_classes)
        
        preds = torch.argmax(flat_logits, axis=1)
        mask = labels != -100

        acc = acc_fn(
            torch.masked_select(preds, mask),
            torch.masked_select(labels, mask)
        )

        return acc

