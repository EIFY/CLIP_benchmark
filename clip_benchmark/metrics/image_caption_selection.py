import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm
from open_clip import METRICS

def evaluate(model, dataloader, tokenizer,  device, amp=True, recall_k_list=[5]):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision
    
    Returns
    -------
    
    dict of accuracy metric
    """
    metric = METRICS[model.geometry]
    autocast = torch.cuda.amp.autocast if amp else suppress
    preds = []
    for batch_images, batch_texts in tqdm(dataloader):
        batch_images = batch_images.to(device)
        # tokenize all texts in the batch
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        nb_texts_for_each_image = [len(texts) for texts in batch_texts]

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            output = model(image=batch_images, text=batch_texts_tok)
            if isinstance(output, dict):
                batch_images_emb, batch_texts_emb, curvature = output['image_features'], output['text_features'], output['curvature']
            else:
                batch_images_emb, batch_texts_emb,  _, _, curvature = output
            batch_images_emb = batch_images_emb.cpu()
            batch_texts_emb = batch_texts_emb.cpu()
        start = 0
        for i, nb in enumerate(nb_texts_for_each_image):
            end = start + nb
            image_emb = batch_images_emb[i:i+1]
            texts_emb = batch_texts_emb[start:end]
            scores = metric(image_emb, texts_emb, curvature)
            scores = scores[0]
            pred = scores.argmax().item()
            start = end 
            preds.append(pred)
    pred = torch.Tensor(preds).long()
    acc = (pred==0).float().mean().item() # 0 is the index of the caption, the rest (>0) are considered negative captions
    metrics = {}
    metrics[f"acc"] = acc
    return metrics