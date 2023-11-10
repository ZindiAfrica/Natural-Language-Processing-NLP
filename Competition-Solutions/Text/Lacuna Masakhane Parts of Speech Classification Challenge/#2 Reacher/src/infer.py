from .model import TransformerModel
import torch
import numpy as np
from tqdm import tqdm
def inference(loader,FOLD, exp, ids_to_labels, PRETRAINED_MODEL, lang):
    model = TransformerModel(PRETRAINED_MODEL).to('cuda')
    model.load_state_dict(torch.load(f"models/{exp}/fold{FOLD}.pt"))
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    final_preds = []
    preds = []
    for batch in tqdm(loader):
        # MOVE BATCH TO GPU AND INFER
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        outputs = model(ids, attention_mask=mask)
        all_preds = torch.argmax(outputs, axis=-1).cpu().numpy() 
        preds.append(torch.nn.functional.softmax(outputs, dim=-1).detach().cpu().numpy())
    
        # INTERATE THROUGH EACH TEXT AND GET PRED
        predictions = []
        for k,text_preds in enumerate(all_preds):
            token_preds = [ids_to_labels[i] for i in text_preds]
    
            prediction = []
            word_ids = batch['wids'][k].numpy()  
            previous_word_idx = -1
            for idx,word_idx in enumerate(word_ids):                            
                if word_idx == -1:
                    pass
                elif word_idx != previous_word_idx:              
                    prediction.append(token_preds[idx])
                    previous_word_idx = word_idx
            predictions.append(prediction)
        final_preds.append(predictions)
    
    preds = np.concatenate(preds)
    np.save(f'{lang}/TEST_fold{FOLD}.npy', preds)
    return final_preds

def folds_inference(loader, ensemble, ids_to_labels):
    final_preds = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for idx, batch in tqdm(enumerate(loader),total=len(loader)):
        # MOVE BATCH TO GPU AND INFER
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        outputs = ensemble[idx]
        all_preds = np.argmax(outputs, axis=-1)
    
        # INTERATE THROUGH EACH TEXT AND GET PRED
        predictions = []
        for k,text_preds in enumerate([all_preds]):
            token_preds = [ids_to_labels[i] for i in text_preds]
    
            prediction = []
            word_ids = batch['wids'][k].numpy()  
            previous_word_idx = -1
            for idx,word_idx in enumerate(word_ids):                            
                if word_idx == -1:
                    pass
                elif word_idx != previous_word_idx:              
                    prediction.append(token_preds[idx])
                    previous_word_idx = word_idx
            predictions.append(prediction)
        final_preds.append(predictions)
    
    return final_preds