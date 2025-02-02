import torch
import pandas as pd

def inference(dataloader, network, trainer, id2label):
    logits = trainer.predict(network, dataloaders=dataloader)
    preds = torch.argmax(
        torch.concat(logits, axis=0), axis=-1
        ).cpu().numpy()
    
    word_ids = []
    for batch in dataloader:
        word_ids.extend(batch["word_ids"].numpy())

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

def submission(df, dataloader, network, trainer, id2label):
    predictions = inference(dataloader, network, trainer, id2label)
    act_predictions = []
    
    for i in range(len(df)):

        idx = df.id.values[i]
        pred = predictions[i] 
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O': j += 1
            else: cls = cls.replace('B','I') 
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            
            if cls != 'O' and cls != '' and end - j > 7:
                act_predictions.append((idx, cls.replace('I-',''),
                                     ' '.join(map(str, list(range(j, end))))))
        
            j = end
        
    sub_df = pd.DataFrame(act_predictions)
    sub_df.columns = ['id','class','predictionstring']

    return sub_df