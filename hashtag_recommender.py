import csv
import json
import numpy as np
import faiss

# ── Preprocessing (pure Python) ───────────────────────────────────────────────

HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&apos;": "'", "&nbsp;": " ",
}

def _decode_html(text):
    for e, c in HTML_ENTITIES.items():
        text = text.replace(e, c)
    return text

def _remove_mentions(text):
    out, i = [], 0
    while i < len(text):
        if text[i] == '@':
            i += 1
            while i < len(text) and text[i] not in ' \t\n':
                i += 1
        else:
            out.append(text[i]); i += 1
    return ''.join(out)

def _remove_urls(text):
    out, i = [], 0
    while i < len(text):
        matched = False
        for prefix in ("https://", "http://", "www."):
            if text[i:i+len(prefix)].lower() == prefix:
                i += len(prefix)
                while i < len(text) and text[i] not in ' \t\n':
                    i += 1
                matched = True; break
        if not matched:
            out.append(text[i]); i += 1
    return ''.join(out)

def _remove_hashtags(text):
    out, i = [], 0
    while i < len(text):
        if text[i] == '#':
            i += 1
            while i < len(text) and text[i] not in ' \t\n.,!?':
                i += 1
        else:
            out.append(text[i]); i += 1
    return ''.join(out)

def _collapse_ws(text):
    out, prev = [], True
    for ch in text:
        if ch in ' \t\n\r':
            if not prev: out.append(' ')
            prev = True
        else:
            out.append(ch); prev = False
    return ''.join(out).strip()

def preprocess_fasttext(text):
    text = _decode_html(text)
    text = _remove_mentions(text)
    text = _remove_urls(text)
    text = _remove_hashtags(text)
    text = _collapse_ws(text)
    return text.lower()             

def preprocess_bert(text):
    text = _decode_html(text)
    text = _remove_mentions(text)
    text = _remove_urls(text)
    text = _remove_hashtags(text)
    text = _collapse_ws(text)
    return text                    

# ── Cosine normalisation ──────────────────────────────────────────────────────

def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalise so inner product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms

# ══════════════════════════════════════════════════════════════════════════════
# FASTTEXT  (via gensim — supports Python 3.13 on Windows)
# ══════════════════════════════════════════════════════════════════════════════

def _get_fasttext_vector(ft_model, text: str) -> np.ndarray:
    tokens = text.strip().split()
    if not tokens:
        return np.zeros(ft_model.vector_size, dtype=np.float32)
    vecs = np.array([ft_model.wv[t] for t in tokens], dtype=np.float32)
    return vecs.mean(axis=0)


def build_fasttext_index(
    csv_path:   str = "tweets_fasttext.csv",
    model_path: str = "cc.en.300.bin",
    index_path: str = "fasttext_index.faiss",
    meta_path:  str = "fasttext_meta.json",
):
    """
    Reads tweets_fasttext.csv, embeds every tweet, builds a FAISS index.
    Loads the pretrained FastText .bin using gensim.
    """
    from gensim.models.fasttext import load_facebook_model

    print("Loading FastText model via gensim (this may take a minute)...")
    ft  = load_facebook_model(model_path)
    dim = ft.vector_size              # 300 for cc.en.300.bin
    print(f"Model loaded — vector size: {dim}")

    tweets, hashtags_list = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tweet = row["clean_tweet"].strip()
            if tweet:
                tweets.append(tweet)
                hashtags_list.append(row["hashtags"].strip())

    print(f"Embedding {len(tweets)} tweets (FastText)...")
    embeddings = np.array(
        [_get_fasttext_vector(ft, t) for t in tweets],
        dtype=np.float32
    )
    embeddings = _l2_normalize(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"tweets": tweets, "hashtags": hashtags_list}, f)

    print(f"Saved → {index_path}  ({len(tweets)} vectors, dim={dim})")
    print(f"Saved → {meta_path}")


# ══════════════════════════════════════════════════════════════════════════════
# BERT
# ══════════════════════════════════════════════════════════════════════════════

def build_bert_index(
    csv_path:   str = "tweets_bert.csv",
    model_name: str = "bert-base-cased",
    index_path: str = "bert_index.faiss",
    meta_path:  str = "bert_meta.json",
    batch_size: int = 64,
):
    import torch
    from transformers import BertTokenizer, BertModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model     = BertModel.from_pretrained(model_name).to(device)
    model.eval()

    tweets, hashtags_list = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tweet = row["clean_tweet"].strip()
            if tweet:
                tweets.append(tweet)
                hashtags_list.append(row["hashtags"].strip())

    print(f"Embedding {len(tweets)} tweets (BERT) in batches of {batch_size}...")
    all_vecs = []

    for start in range(0, len(tweets), batch_size):
        batch  = tweets[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_vecs.append(cls_vecs.astype(np.float32))

        if (start // batch_size + 1) % 10 == 0:
            print(f"  {start + len(batch)} / {len(tweets)}")

    embeddings = np.vstack(all_vecs)
    dim        = embeddings.shape[1]     # 768 for bert-base
    embeddings = _l2_normalize(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"tweets": tweets, "hashtags": hashtags_list}, f)

    print(f"Saved → {index_path}  ({len(tweets)} vectors, dim={dim})")
    print(f"Saved → {meta_path}")


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL  (shared for both models)
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_hashtags(scores, indices, hashtags_list):
    tag_scores = {}
    for score, idx in zip(scores, indices):
        tags = [t.strip() for t in hashtags_list[idx].split(",") if t.strip()]
        for tag in tags:
            tag_scores[tag] = tag_scores.get(tag, 0.0) + float(score)
    return sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)


def recommend(
    tweet:          str,
    model_type:     str,        # "fasttext" or "bert"
    top_k:          int = 5,    # number of neighbours to retrieve
    top_n:          int = 3,    # number of hashtags to recommend
    # FastText artifacts
    ft_model        = None,
    ft_index        = None,
    ft_meta         = None,
    # BERT artifacts
    bert_tokenizer  = None,
    bert_model      = None,
    bert_index      = None,
    bert_meta       = None,
    bert_device     = None,
):
    
    if model_type == "fasttext":
        clean = preprocess_fasttext(tweet)
        vec   = _get_fasttext_vector(ft_model, clean)
        vec   = _l2_normalize(vec.reshape(1, -1))
        index = ft_index
        meta  = ft_meta

    elif model_type == "bert":
        import torch
        from transformers import BertTokenizer, BertModel

        clean  = preprocess_bert(tweet)
        inputs = bert_tokenizer(
            clean,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(bert_device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        vec = _l2_normalize(vec.reshape(1, -1)).astype(np.float32)
        index = bert_index
        meta  = bert_meta

    else:
        raise ValueError(f"model_type must be 'fasttext' or 'bert', got '{model_type}'")

    scores, indices = index.search(vec.astype(np.float32), top_k)
    scores  = scores[0].tolist()
    indices = indices[0].tolist()

    ranked_tags = _aggregate_hashtags(scores, indices, meta["hashtags"])

    neighbours = [
        {
            "tweet":    meta["tweets"][idx],
            "hashtags": meta["hashtags"][idx],
            "score":    round(score, 4),
        }
        for score, idx in zip(scores, indices)
    ]

    return {
        "input_tweet":     tweet,
        "clean_tweet":     clean,
        "recommendations": ranked_tags[:top_n],
        "neighbours":      neighbours,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOADER HELPERS  — call once at startup, reuse across queries
# ══════════════════════════════════════════════════════════════════════════════

def load_fasttext_artifacts(model_path, index_path, meta_path):
    """Load gensim FastText model + FAISS index + metadata."""
    from gensim.models.fasttext import load_facebook_model
    print("Loading FastText artifacts...")
    ft    = load_facebook_model(model_path)
    index = faiss.read_index(index_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    print("FastText artifacts loaded.")
    return ft, index, meta


def load_bert_artifacts(model_name, index_path, meta_path):
    """Load BERT tokenizer + model + FAISS index + metadata."""
    import torch
    from transformers import BertTokenizer, BertModel
    print("Loading BERT artifacts...")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model     = BertModel.from_pretrained(model_name).to(device)
    model.eval()
    index = faiss.read_index(index_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"BERT artifacts loaded (device: {device}).")
    return tokenizer, model, index, meta, device

if __name__ == "__main__":

    # ── FASTTEXT ──────────────────────────────────────────────────────────────
    # Step 1 — build index once (skip if already done)
    build_fasttext_index(
        csv_path   = "tweets_fasttext.csv",
        model_path = "cc.en.300.bin",
        index_path = "fasttext_index.faiss",
        meta_path  = "fasttext_meta.json",
    )

    # Step 2 — load artifacts
    ft, ft_idx, ft_meta = load_fasttext_artifacts(
        "cc.en.300.bin", "fasttext_index.faiss", "fasttext_meta.json"
    )

    # ── BERT ──────────────────────────────────────────────────────────────────
    # Step 1 — build index once (skip if already done)
    build_bert_index(
        csv_path   = "tweets_bert.csv",
        model_name = "bert-base-cased",
        index_path = "bert_index.faiss",
        meta_path  = "bert_meta.json",
        batch_size = 64,
    )

    # Step 2 — load artifacts
    tok, bm, b_idx, b_meta, dev = load_bert_artifacts(
        "bert-base-cased", "bert_index.faiss", "bert_meta.json"
    )