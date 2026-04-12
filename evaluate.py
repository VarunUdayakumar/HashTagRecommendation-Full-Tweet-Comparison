import csv
import json
import random
import os
import sys
import torch
from hashtag_recommender import recommend as _recommend, _l2_normalize, _aggregate_hashtags
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def precision_at_k(predicted: list, ground_truth: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for tag in predicted[:k] if tag in ground_truth)
    return hits / k


def recall_at_k(predicted: list, ground_truth: set, k: int) -> float:
    if not ground_truth:
        return 0.0
    hits = sum(1 for tag in predicted[:k] if tag in ground_truth)
    return hits / len(ground_truth)


def hit_rate_at_k(predicted: list, ground_truth: set, k: int) -> float:
    return 1.0 if set(predicted[:k]) & ground_truth else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class HashtagEvaluator:
    def __init__(self, k: int = 3):
        self.k         = k
        self.precision = []
        self.recall    = []
        self.hit_rate  = []

    def add(self, predicted: list, ground_truth):
        gt_set = set(t.strip().lower() for t in ground_truth if t.strip())
        pred   = [t.strip().lower() for t in predicted]
        if not gt_set:
            return
        self.precision.append(precision_at_k(pred, gt_set, self.k))
        self.recall.append(recall_at_k(pred, gt_set, self.k))
        self.hit_rate.append(hit_rate_at_k(pred, gt_set, self.k))

    def compute(self) -> dict:
        n = len(self.precision)
        if n == 0:
            return {"k": self.k, "n_samples": 0,
                    "precision": 0.0, "recall": 0.0, "hit_rate": 0.0}
        return {
            "k":         self.k,
            "n_samples": n,
            "precision": round(sum(self.precision) / n, 4),
            "recall":    round(sum(self.recall)    / n, 4),
            "hit_rate":  round(sum(self.hit_rate)  / n, 4),
        }

    def reset(self):
        self.precision = []
        self.recall    = []
        self.hit_rate  = []


# ══════════════════════════════════════════════════════════════════════════════
# TEST SET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_test_set(csv_path: str, test_size: int = 500, seed: int = 42) -> list:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Determine tweet column and whether it is already pre-processed
        if "clean_tweet" in reader.fieldnames:
            tweet_col   = "clean_tweet"
            pre_cleaned = True
        elif "tweet" in reader.fieldnames:
            tweet_col   = "tweet"
            pre_cleaned = False
        else:
            raise KeyError(
                f"CSV must have a 'tweet' or 'clean_tweet' column. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            tweet    = row[tweet_col].strip()
            hashtags = [t.strip() for t in row["hashtags"].split(",") if t.strip()]
            if tweet and hashtags:
                rows.append({
                    "raw_tweet":   tweet,
                    "hashtags":    hashtags,
                    "pre_cleaned": pre_cleaned,
                })
    random.seed(seed)
    random.shuffle(rows)
    return rows[:test_size]


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model_type:   str,
    recommend_fn,
    test_rows:    list,
    k_values:     list = [1, 3, 5],
) -> dict:

    print(f"\n{'='*52}")
    print(f"  Model: {model_type.upper()}  |  test size: {len(test_rows)}")
    print(f"{'='*52}")

    all_results = {}

    for k in k_values:
        evaluator = HashtagEvaluator(k=k)
        for i, row in enumerate(test_rows):
            try:
                predicted = recommend_fn(row["raw_tweet"],
                                         pre_cleaned=row.get("pre_cleaned", False))
                evaluator.add(predicted, row["hashtags"])
            except Exception as e:
                print(f"  Warning: skipped row {i} — {e}")
                continue
        all_results[k] = evaluator.compute()

    # ── Print results table ───────────────────────────────────────────────────
    col_w = 10
    print(f"\n  {'Metric':<12}" + "".join(f"{'@k='+str(k):>{col_w}}" for k in k_values))
    print("  " + "-" * (12 + col_w * len(k_values)))
    for metric in ["precision", "recall", "hit_rate"]:
        row_str = f"  {metric:<12}"
        for k in k_values:
            row_str += f"{all_results[k][metric]:>{col_w}.4f}"
        print(row_str)
    print(f"\n  Samples evaluated: {all_results[k_values[0]]['n_samples']}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADERS  (graceful — skip if libraries not installed)
# ══════════════════════════════════════════════════════════════════════════════

def _load_fasttext_artifacts(ft_bin: str, faiss_index: str, meta_json: str):
    """Load FastText model + FAISS index + metadata JSON."""
    try:
        from hashtag_recommender import load_fasttext_artifacts
        return load_fasttext_artifacts(ft_bin, faiss_index, meta_json)
    except ImportError as e:
        raise RuntimeError(f"Could not import hashtag_recommender: {e}")


def _load_bert_artifacts(model_name: str, faiss_index: str, meta_json: str):
    """Load BERT tokenizer + model + FAISS index + metadata.
    Returns (tokenizer, bert_model, index, meta, device) — 5 values."""
    try:
        from hashtag_recommender import load_bert_artifacts
        return load_bert_artifacts(model_name, faiss_index, meta_json)
    except ImportError as e:
        raise RuntimeError(f"Could not import hashtag_recommender: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMEND WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_fasttext_recommend_fn(ft_model, ft_index, ft_meta, top_k: int = 10):
    from hashtag_recommender import recommend as _recommend, _get_fasttext_vector, _l2_normalize, _aggregate_hashtags
    import numpy as np

    def _fn(tweet: str, pre_cleaned: bool = False) -> list:
        vec    = _get_fasttext_vector(ft_model, tweet)
        vec    = _l2_normalize(vec.reshape(1, -1)).astype(np.float32)
        scores, indices = ft_index.search(vec, top_k)
        ranked = _aggregate_hashtags(scores[0].tolist(), indices[0].tolist(),
                                        ft_meta["hashtags"])
        return [tag for tag, _ in ranked]




def make_bert_recommend_fn(bert_tokenizer, bert_model, bert_index, bert_meta, bert_device, top_k: int = 10):
    def _fn(tweet: str, pre_cleaned: bool = False) -> list:
        inputs = bert_tokenizer(
            tweet,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(bert_device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        vec    = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        vec    = _l2_normalize(vec.reshape(1, -1)).astype(np.float32)
        scores, indices = bert_index.search(vec, top_k)
        ranked = _aggregate_hashtags(scores[0].tolist(), indices[0].tolist(),
                                        bert_meta["hashtags"])
        return [tag for tag, _ in ranked]
    return _fn


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE OUTPUT PRINTER
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_TWEETS = [
    "updated my iPhone to 3.0 and now iTunes keeps timing out on activation, so annoying",
    "really tempted by the new apple announcement but my warranty just expired two weeks ago",
    "can't believe the pens lost that game, youtube is down so i can't even watch the replay",
    "bad bad match today, rcb and kxip both need to sort themselves out before the playoffs",
    "lions gave away that game in the last five minutes, absolutely gutted",
    "eurovision has genuinely kept me sane this week, don't judge me",
    "watching masterchef and my three year old will not stop screaming, can't hear a thing",
    "they'll be knocking on your door for the census soon, whether you like it or not",
    "the situation in iran is terrifying, can't stop refreshing the news feed",
    "someone just thought of robotpickuplines as a trending topic and honestly respect that",
]

def print_sample_output(model_type: str, recommend_fn, tweets: list = SAMPLE_TWEETS):
    print(f"\n{'─'*52}")
    print(f"  Sample predictions — {model_type.upper()}")
    print(f"{'─'*52}")
    for tweet in tweets:
        try:
            tags = recommend_fn(tweet)[:5]          # show at most 5
            tag_str = "  ".join(f"#{t}" for t in tags)
            print(f"\n  Tweet : {tweet}")
            print(f"  Tags  : {tag_str}")
        except Exception as e:
            print(f"\n  Tweet : {tweet}")
            print(f"  Tags  : [ERROR — {e}]")
    print(f"\n{'─'*52}\n")



# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── FastText evaluation ───────────────────────────────────────────────────
    CSV_PATH_FAST = "tweets_fasttext_test.csv"
    FT_BIN        = "cc.en.300.bin"
    FT_FAISS      = "fasttext_index.faiss"
    FT_META       = "fasttext_meta.json"

    if not os.path.exists(CSV_PATH_FAST):
        print(f"\n  [SKIP] '{CSV_PATH_FAST}' not found.")
    elif not all(os.path.exists(p) for p in [FT_BIN, FT_FAISS, FT_META]):
        missing = [p for p in [FT_BIN, FT_FAISS, FT_META] if not os.path.exists(p)]
        print(f"\n  [SKIP] FastText — missing files: {missing}")
    else:
        test_rows_fast = build_test_set(CSV_PATH_FAST, test_size=500)
        mode = "pre-cleaned" if test_rows_fast[0]["pre_cleaned"] else "raw"
        print(f"\n  Built FastText test set: {len(test_rows_fast)} rows  (mode: {mode})")
        print("\n  Loading FastText artifacts …")
        try:
            ft, ft_idx, ft_meta = _load_fasttext_artifacts(FT_BIN, FT_FAISS, FT_META)
            ft_recommend = make_fasttext_recommend_fn(ft, ft_idx, ft_meta, top_k=10)
            print_sample_output("fasttext", ft_recommend)
            evaluate_model("fasttext", ft_recommend, test_rows_fast, k_values=[1, 3, 5])
        except Exception as e:
            print(f"  [ERROR] FastText evaluation failed: {e}")

    # ── BERT evaluation ───────────────────────────────────────────────────────
    CSV_PATH_BERT = "tweets_bert_test.csv"
    BERT_MODEL    = "bert-base-cased"
    BERT_FAISS    = "bert_index.faiss"
    BERT_META     = "bert_meta.json"

    if not os.path.exists(CSV_PATH_BERT):
        print(f"\n  [SKIP] '{CSV_PATH_BERT}' not found.")
    elif not all(os.path.exists(p) for p in [BERT_FAISS, BERT_META]):
        missing = [p for p in [BERT_FAISS, BERT_META] if not os.path.exists(p)]
        print(f"\n  [SKIP] BERT — missing files: {missing}")
    else:
        test_rows_bert = build_test_set(CSV_PATH_BERT, test_size=500)
        mode = "pre-cleaned" if test_rows_bert[0]["pre_cleaned"] else "raw"
        print(f"\n  Built BERT test set: {len(test_rows_bert)} rows  (mode: {mode})")
        print("\n  Loading BERT artifacts …")
        try:
            bert_tok, bert_mdl, bert_idx, bert_meta, bert_dev = \
                _load_bert_artifacts(BERT_MODEL, BERT_FAISS, BERT_META)
            bert_recommend = make_bert_recommend_fn(
                bert_tok, bert_mdl, bert_idx, bert_meta, bert_dev, top_k=10
            )
            print_sample_output("bert", bert_recommend)
            evaluate_model("bert", bert_recommend, test_rows_bert, k_values=[1, 3, 5])
        except Exception as e:
            print(f"  [ERROR] BERT evaluation failed: {e}")
