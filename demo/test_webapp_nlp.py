from pathlib import Path

import datasets
import numpy as np
import pacmap
import pandas as pd
import torch
import transformers

from shapash.explainer.nlp_explainer import NlpExplainer

_HERE = Path(__file__).parent


def project_embeddings(embeddings, n_components=2):
    """Project high-dimensional embeddings to a lower-dimensional space using PaCMAP.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, hidden_dim)
        Dense vector representations of the input texts.
    n_components : int, optional
        Number of output dimensions. Default is 2 (for 2-D scatter plots).

    Returns
    -------
    np.ndarray, shape (n_samples, n_components)
        Low-dimensional projection initialised with PCA.
    """
    projector = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=5,
        MN_ratio=0.5,
        FP_ratio=2.0,
        # random_state=1
    )

    return projector.fit_transform(embeddings, init="pca")


def get_embeddings(texts, classifier, tokenizer, batch_size=32):
    """Extract mean-pooled last-layer hidden states from a HuggingFace classifier.

    Runs inference in batches without gradient computation. Tokens are pooled
    using the attention mask so padding tokens do not contribute to the mean.

    Parameters
    ----------
    texts : list of str
        Raw input texts to encode.
    classifier : transformers.PreTrainedModel
        A HuggingFace model that accepts ``output_hidden_states=True``.
        The model's ``device`` attribute is used to move batches automatically.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer matching the classifier; applied with padding and truncation.
    batch_size : int, optional
        Number of texts processed per forward pass. Default is 32.

    Returns
    -------
    np.ndarray, shape (n_samples, hidden_dim)
        One embedding vector per input text.
    """
    device = classifier.device

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i : i + batch_size], padding=True, truncation=True, return_tensors="pt")
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
        # "nateraw/bert-base-uncased-emotion",
        use_fast=True,
    )
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(
        "bhadresh-savani/distilbert-base-uncased-emotion",
        # "nateraw/bert-base-uncased-emotion"
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
    xpl.save_snapshot(_HERE / "explainer_snapshot.pkl", scatter_xy=projected)
    xpl.run_app(port=8050, debug=False, scatter_xy=projected)
