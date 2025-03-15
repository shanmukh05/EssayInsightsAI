import ast
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


class FeedbackPrizeDataset(Dataset):
    def __init__(self, df, config, label2id, tokenizer, return_label=True):
        self.df = df
        self.max_len = config["data"]["max_len"]
        self.tokenizer = tokenizer
        self.return_label = return_label
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        fold_label = self.df.iloc[idx]["fold_label"]
        text_encoding = self.tokenizer(
            text.split(),
            max_length=self.max_len,
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
        )
        word_ids = text_encoding.word_ids()

        output = {k: torch.as_tensor(v) for k, v in text_encoding.items()}

        if self.return_label:
            word_labels = ast.literal_eval(self.df.iloc[idx]["prediction"])

            prev_idx = None
            labels = []

            for idx in word_ids:
                if idx is None:
                    labels.append(-100)
                elif idx != prev_idx:
                    labels.append(self.label2id[word_labels[idx]])
                else:
                    labels.append(-100)
                prev_idx = idx

            output["labels"] = torch.as_tensor(labels)

        output["word_ids"] = torch.as_tensor(
            [i if i is not None else -1 for i in word_ids]
        )
        output["fold_label"] = fold_label

        return output


class FeedbackPrizeDataModule(pl.LightningDataModule):
    def __init__(self, config, label2id, df=None):
        super().__init__()

        self.config = config
        self.data_split = config["data"]["strategy"]
        self.label2id = label2id
        self.df = df

    def prepare_data(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["paths"]["tokenizer"]
        )

    def setup(self, stage=None):
        if self.data_split == "train_val":
            self.train_df, self.val_df = train_test_split(
                self.df, test_size=self.config["data"]["val_split"]
            )
        elif self.data_split == "kfold":
            kfold = KFold(n_splits=self.config["data"]["num_folds"])
            train_ids, val_ids = list(kfold.split(self.df))[self.k]
            self.train_df, self.val_df = self.df.iloc[train_ids], self.df.iloc[val_ids]
        elif self.data_split == "stratifiedkfold":
            kfold = StratifiedKFold(n_splits=self.config["data"]["num_folds"])
            train_ids, val_ids = list(kfold.split(self.df, self.df["fold_label"]))[
                self.k
            ]
            self.train_df, self.val_df = self.df.iloc[train_ids], self.df.iloc[val_ids]

    def train_dataloader(self):
        train_ds = FeedbackPrizeDataset(
            self.train_df, self.config, self.label2id, self.tokenizer, return_label=True
        )
        return DataLoader(train_ds, **self.config["train_ds"])

    def val_dataloader(self):
        val_ds = FeedbackPrizeDataset(
            self.val_df, self.config, self.label2id, self.tokenizer, return_label=True
        )
        return DataLoader(val_ds, **self.config["test_ds"])

    def test_dataloader(self, test_df):
        test_ds = FeedbackPrizeDataset(
            test_df, self.config, self.label2id, self.tokenizer, return_label=False
        )
        return DataLoader(test_ds, **self.config["test_ds"])
