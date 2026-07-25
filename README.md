# HashTag Recommendation — FastText vs BERT

A hashtag recommendation system for tweets that compares two embedding approaches — **FastText** (word-vector averaging) and **BERT** (contextual `[CLS]` embeddings) — using semantic similarity search over a large tweet corpus. Given a new tweet, the system retrieves the most similar historical tweets and aggregates their hashtags into ranked recommendations.

## How it works

1. **Dataset preparation** (`dataset.py`) — Loads the [Sentiment140](http://help.sentiment140.com/for-students) dataset (1.6M tweets), extracts tweets that contain hashtags, and trims to a working subset.
2. **Preprocessing** (`preprocess_fasttext_bert.py`) — Cleans raw tweets (strips HTML entities, @mentions, URLs, and `#hashtag` tokens, collapses whitespace) and produces two parallel cleaned datasets: lowercased text for FastText, case-preserved text for BERT.
3. **Indexing & recommendation** (`hashtag_recommender.py`):
   - Embeds every cleaned tweet with either a pretrained **FastText** model (via `gensim`, averaging word vectors) or **BERT** (`bert-base-cased`, using the `[CLS]` token).
   - L2-normalizes embeddings and builds a **FAISS** flat inner-product index (cosine similarity) per model.
   - For a new tweet, retrieves the top-*k* nearest neighbor tweets and aggregates their hashtags by summed similarity score to produce the top-*n* recommendations.
4. **Evaluation** (`evaluate.py`) — Benchmarks both models on a held-out test set using **Precision@k**, **Recall@k**, and **Hit Rate@k**, and prints qualitative sample predictions on a fixed set of test tweets.

## Repository structure

```
.
├── dataset.py                     # Builds tweets_hashtags.csv from Sentiment140
├── preprocess_fasttext_bert.py    # Cleans tweets into FastText/BERT variants
├── hashtag_recommender.py         # Embedding, FAISS indexing, and recommendation logic
├── evaluate.py                    # Precision/Recall/Hit-Rate evaluation harness
└── README.md
```

## Requirements

```bash
pip install numpy pandas faiss-cpu gensim torch transformers
```

You'll also need:
- The [Sentiment140 dataset](http://help.sentiment140.com/for-students) (`training.1600000.processed.noemoticon.csv`)
- A pretrained FastText binary, e.g. [`cc.en.300.bin`](https://fasttext.cc/docs/en/crawl-vectors.html) (English, 300d)

## Usage

### 1. Build the raw dataset

Update the CSV path in `dataset.py` to point to your local copy of Sentiment140, then run:

```bash
python dataset.py
```

This produces `tweets_hashtags.csv` (tweet + comma-separated hashtags).

### 2. Preprocess for both models

```bash
python preprocess_fasttext_bert.py
```

Reads `tweets_test.csv` and writes `tweets_fasttext_test.csv` (lowercased) and `tweets_bert_test.csv` (case-preserved), with URLs, mentions, and hashtag tokens stripped out of the tweet text.

### 3. Build indexes and get recommendations

```python
from hashtag_recommender import (
    build_fasttext_index, load_fasttext_artifacts,
    build_bert_index, load_bert_artifacts,
    recommend,
)

# FastText
build_fasttext_index(csv_path="tweets_fasttext.csv", model_path="cc.en.300.bin")
ft, ft_idx, ft_meta = load_fasttext_artifacts("cc.en.300.bin", "fasttext_index.faiss", "fasttext_meta.json")

result = recommend("just watched the match, what a comeback!", model_type="fasttext",
                    ft_model=ft, ft_index=ft_idx, ft_meta=ft_meta)
print(result["recommendations"])
```

Running `python hashtag_recommender.py` directly will build and load both the FastText and BERT indexes end-to-end.

### 4. Evaluate

```bash
python evaluate.py
```

Prints a Precision@k / Recall@k / Hit-Rate@k table for `k = [1, 3, 5]` for each model, plus qualitative predictions on a fixed sample of test tweets.

## Notes

- Cosine similarity is used throughout (embeddings are L2-normalized, FAISS uses inner product).
- BERT embedding is GPU-accelerated automatically if `torch.cuda.is_available()`.
- `evaluate.py` skips a model's evaluation gracefully if its required CSV/index/artifact files aren't present.

## Related: a deterministic alternative

We also explored a non-embedding approach — **[TagUp](https://github.com/subrajith05/TagUp)** — which recommends hashtags using TF-IDF/YAKE keyword extraction, candidate generation, and frequency + co-occurrence ranking instead of learned embeddings and similarity search. It ended up outperforming both the FastText and BERT pipelines here, so it's worth a look if you want a lighter, fully deterministic alternative.
