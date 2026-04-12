
# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE       = "tweets_test.csv"
OUT_FASTTEXT     = "tweets_fasttext_test.csv"
OUT_BERT         = "tweets_bert_test.csv"
MIN_CHARS        = 3      
BERT_LOWERCASE   = False     

# ── 1. Manual CSV parser ──────────────────────────────────────────────────────

def parse_csv(text):
    rows = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        fields    = []
        current   = ""
        in_quotes = False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '"':
                if in_quotes and i + 1 < len(line) and line[i + 1] == '"':
                    current += '"'   # escaped quote inside quoted field
                    i += 2
                    continue
                in_quotes = not in_quotes
            elif ch == ',' and not in_quotes:
                fields.append(current)
                current = ""
                i += 1
                continue
            else:
                current += ch
            i += 1
        fields.append(current)
        rows.append(fields)
    return rows


# ── 2. HTML entity decoder ────────────────────────────────────────────────────

HTML_ENTITIES = {
    "&amp;":   "&",
    "&lt;":    "<",
    "&gt;":    ">",
    "&quot;":  '"',
    "&#39;":   "'",
    "&apos;":  "'",
    "&nbsp;":  " ",
    "&ndash;": "-",
    "&mdash;": "--",
}

def decode_html_entities(text):
    for entity, char in HTML_ENTITIES.items():
        text = text.replace(entity, char)
    return text


# ── 3. Remove @mentions ───────────────────────────────────────────────────────

def remove_mentions(text):
    result = []
    i = 0
    while i < len(text):
        if text[i] == '@':
            i += 1
            while i < len(text) and text[i] not in (' ', '\t', '\n'):
                i += 1
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


# ── 4. Remove URLs ────────────────────────────────────────────────────────────

def remove_urls(text):
    result = []
    i = 0
    while i < len(text):
        matched = False
        for prefix in ("https://", "http://", "www."):
            plen = len(prefix)
            if text[i: i + plen].lower() == prefix:
                i += plen
                while i < len(text) and text[i] not in (' ', '\t', '\n'):
                    i += 1
                matched = True
                break
        if not matched:
            result.append(text[i])
            i += 1
    return ''.join(result)


# ── 5. Remove #hashtag tokens ─────────────────────────────────────────────────

def remove_hashtag_tokens(text):
    result = []
    i = 0
    while i < len(text):
        if text[i] == '#':
            i += 1
            while i < len(text) and text[i] not in (' ', '\t', '\n', '.', '!', '?', ','):
                i += 1
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


# ── 6. Collapse whitespace ────────────────────────────────────────────────────

def collapse_whitespace(text):
    result    = []
    prev_space = True
    for ch in text:
        if ch in (' ', '\t', '\n', '\r'):
            if not prev_space:
                result.append(' ')
            prev_space = True
        else:
            result.append(ch)
            prev_space = False
    return ''.join(result).strip()


# ── 7. Lowercase (FastText only) ──────────────────────────────────────────────

def to_lowercase(text):
    return text.lower()


# ── Core shared cleaning (steps 2-6) ─────────────────────────────────────────

def clean_base(text):
    text = decode_html_entities(text)
    text = remove_mentions(text)
    text = remove_urls(text)
    text = remove_hashtag_tokens(text)
    text = collapse_whitespace(text)
    return text


# ── Model-specific pipelines ──────────────────────────────────────────────────

def preprocess_fasttext(text):
    text = clean_base(text)
    text = to_lowercase(text)
    return text


def preprocess_bert(text):
    text = clean_base(text)
    if BERT_LOWERCASE:
        text = to_lowercase(text)
    return text


# ── CSV writer ────────────────────────────────────────────────────────────────

def write_csv(filepath, header, rows):
    def escape(field):
        if ',' in field or '"' in field or '\n' in field:
            return '"' + field.replace('"', '""') + '"'
        return field

    lines = [",".join(escape(h) for h in header)]
    for row in rows:
        lines.append(",".join(escape(str(v)) for v in row))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Read raw file
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = f.read().replace("\r\n", "\n").replace("\r", "\n")

    all_rows = parse_csv(raw)
    header   = all_rows[0]   # ['tweet', 'hashtags']
    data     = all_rows[1:]
    print(f"Loaded       : {len(data)} rows")

    fasttext_rows = []
    bert_rows     = []
    skipped       = 0

    for row in data:
        if len(row) < 2:
            skipped += 1
            continue

        raw_tweet = row[0]
        hashtags  = row[1].strip()

        ft_text   = preprocess_fasttext(raw_tweet)
        bert_text = preprocess_bert(raw_tweet)

        # Drop if cleaning leaves tweet too short
        if len(ft_text) < MIN_CHARS or len(bert_text) < MIN_CHARS:
            skipped += 1
            continue

        fasttext_rows.append((ft_text, hashtags))
        bert_rows.append((bert_text, hashtags))

    print(f"Kept         : {len(fasttext_rows)} rows")
    print(f"Skipped      : {skipped} rows")

    # Write outputs
    write_csv(OUT_FASTTEXT, ["clean_tweet", "hashtags"], fasttext_rows)
    write_csv(OUT_BERT,     ["clean_tweet", "hashtags"], bert_rows)

    print(f"\nSaved → tweets_fasttext.csv")
    print(f"Saved → tweets_bert.csv")

    # ── Sample comparison ─────────────────────────────────────────────────────
    print("\n── Sample comparison (first 6 rows) ─────────────────────────────────")
    print(f"  {'HASHTAG':<20} {'FASTTEXT':<55} BERT")
    print("  " + "─" * 110)
    for (ft, ht), (bt, _) in zip(fasttext_rows[:6], bert_rows[:6]):
        print(f"  {ht:<20} {ft:<55} {bt}")

if __name__ == "__main__":
    main()
