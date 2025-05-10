import torch
import pandas as pd


def inference(dataloader, network, trainer, id2label):
    logits = trainer.predict(network, dataloaders=dataloader)
    logits = torch.concat(logits, axis=0)

    word_ids = []
    for batch in dataloader:
        word_ids.extend(batch["word_ids"].cpu().numpy())

    predictions = logits2pred(logits, id2label, word_ids)

    return predictions


def logits2pred(logits, id2label, word_ids):
    preds = torch.argmax(logits, axis=-1).cpu().numpy()
    predictions = []
    for i, pred in enumerate(preds):
        pred_token = [id2label[j] for j in pred]
        word_id = word_ids[i]
        prev_wid = -1
        prediction = []

        for id_, wid in enumerate(word_id):
            if wid == -1:
                pass
            elif wid != prev_wid:
                prediction.append(pred_token[id_])
                prev_wid = wid
        predictions.append(prediction)
    return predictions


def submission(df, predictions):
    act_predictions = []

    for i in range(len(df)):

        idx = df.id.values[i]
        pred = predictions[i]
        start = 0
        while start < len(pred):
            cls = pred[start]

            if cls == "O":
                start += 1
            else:
                end = start + 1
                cls = cls.replace("B", "I")
                while end < len(pred) and pred[end] == cls:
                    end += 1

                if end - start > 7:
                    act_predictions.append(
                        (
                            idx,
                            cls.replace("I-", ""),
                            " ".join(map(str, list(range(start, end)))),
                        )
                    )
                start = end

    sub_df = pd.DataFrame(act_predictions)
    sub_df.columns = ["id", "class", "predictionstring"]

    return sub_df
