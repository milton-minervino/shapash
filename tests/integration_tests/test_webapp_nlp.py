import datasets
import pandas as pd
import transformers

import torch
import numpy as np

from shapash.explainer.nlp_explainer import NlpExplainer
import pacmap


def project_embeddings(embeddings, n_components=2):

    projector = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=5,
        MN_ratio=0.5,
        FP_ratio=2.0,
        # random_state=1
    )

    return projector.fit_transform(
        embeddings, init="pca"
    )


def get_embeddings(texts, classifier, tokenizer, batch_size=32):

    device = classifier.device


    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i+batch_size], padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}  # fix 1: move to GPU
        with torch.no_grad():
            out = classifier(**batch, output_hidden_states=True)  # fix 2: request hidden states
        mask = batch["attention_mask"].unsqueeze(-1)
        hidden = out.hidden_states[-1]  # last layer, shape [batch, seq_len, hidden_dim]
        emb = (hidden * mask).sum(1) / mask.sum(1)
        all_embeddings.append(emb.cpu().numpy())
    return np.vstack(all_embeddings)


if __name__ == "__main__":

    N = 100

    # load the emotion dataset
    dataset = datasets.load_dataset("dair-ai/emotion", split="train")
    data = pd.DataFrame({"text": dataset["text"], "emotion": dataset["label"]})

    sentences = dataset["text"][:N]
    labels = dataset["label"][:N]

    # load the model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "bhadresh-savani/distilbert-base-uncased-emotion",
        #"nateraw/bert-base-uncased-emotion",
        use_fast=True)
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(
        "bhadresh-savani/distilbert-base-uncased-emotion",
        #"nateraw/bert-base-uncased-emotion"
        ).cuda()

    # build a pipeline object to do predictions
    pred = transformers.pipeline(
        "text-classification",
        model=classifier,
        tokenizer=tokenizer,
        batch_size=4,
        top_k=None,
    )

    embeddings = get_embeddings(sentences, classifier, tokenizer)
    projected = project_embeddings(embeddings, n_components=2)

    label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    # from shapash.backend.nlp_lime_backend import NlpLimeBackend
    # lime_backend = NlpLimeBackend(
    #     pred,
    #     label_names=label_names,
    #     explainer_compute_args={
    #         "num_samples": 300,   # default is 5000 — this is the main lever
    #         "num_features": 10,   # words explained per sample per label
    #         }
    #     )
    # xpl = NlpExplainer(pred, label_names=label_names, backend=lime_backend)

    xpl = NlpExplainer(pred, label_names=label_names)

    xpl.compile(sentences, y_true=[label_names[i] for i in labels])
    xpl.run_app(port=8050, debug=False, scatter_xy=projected)
