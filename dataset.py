import pandas as pd
import re

# ── 1. Load the raw CSV ──────────────────────────────────────────────────────
# sentiment140 has no header row; assign column names manually
col_names = ["sentiment", "id", "date", "query", "user", "text"]

df_raw = pd.read_csv(
    r"C:\Users\varun\OneDrive\Desktop\varun\NLP\training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None,
    names=col_names
)
print(f"Loaded {len(df_raw):,} rows")

# ── 2. Extract hashtags ──────────────────────────────────────────────────────
def extract_hashtags(text):
    return ", ".join(re.findall(r"#(\w+)", str(text).lower()))

df_raw["hashtags"] = df_raw["text"].apply(extract_hashtags)

# ── 3. Filter & trim ─────────────────────────────────────────────────────────
df = (
    df_raw[df_raw["hashtags"] != ""][["text", "hashtags"]]
    .rename(columns={"text": "tweet"})
    .reset_index(drop=True)
    .head(100_000)
)

print(f"Tweets with hashtags: {len(df):,}")
print(df.head(5))

# ── 4. Save to CSV ───────────────────────────────────────────────────────────
df.to_csv("tweets_hashtags.csv", index=False, encoding="utf-8")
print("Saved → tweets_hashtags.csv")