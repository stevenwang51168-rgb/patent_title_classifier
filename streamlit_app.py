import json
import random
import time
from typing import List

import pandas as pd
import streamlit as st
from openai import OpenAI
from pandas.api.types import CategoricalDtype

# ----------------------------
# CONFIG
# ----------------------------
BATCH_SIZE = 25          # tune: 10–50
MAX_RETRIES = 6          # tune: 5–8
BASE_DELAY = 1.0         # seconds
MAX_DELAY = 30.0         # cap backoff

client = OpenAI()

CATEGORY_ORDER = ["Chemical", "Biotechnology", "Electrical", "Mechanical", "Software"]
CATEGORY_TYPE = CategoricalDtype(categories=CATEGORY_ORDER, ordered=True)


# ----------------------------
# RATE-LIMIT / TRANSIENT RETRY
# ----------------------------
def call_openai_with_retry(create_fn, max_retries: int = MAX_RETRIES):
    """
    create_fn: a zero-arg function that performs the OpenAI request.
    Retries on rate limits and transient server/network failures with exp backoff + jitter.
    """
    for attempt in range(max_retries):
        try:
            return create_fn()
        except Exception as e:
            msg = str(e).lower()

            retryable = (
                "rate limit" in msg
                or "429" in msg
                or "timeout" in msg
                or "temporarily" in msg
                or "server error" in msg
                or "502" in msg
                or "503" in msg
                or "504" in msg
                or "connection" in msg
            )

            if not retryable or attempt == max_retries - 1:
                raise

            delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt))
            delay *= (0.75 + random.random() * 0.5)  # jitter 0.75x–1.25x
            time.sleep(delay)


# ----------------------------
# CLASSIFIERS (BATCH + FALLBACK)
# ----------------------------
def classify_patent_title_one(title: str) -> str:
    title = (title or "").strip()
    if not title:
        return "Unknown"

    prompt = (
        "You are a strict classifier.\n"
        f"Classify the patent title into exactly ONE of: {', '.join(CATEGORY_ORDER)}.\n"
        "Return ONLY the category name.\n\n"
        f"Title: {title}"
    )

    def _do_request():
        return client.responses.create(model="gpt-5.2", input=prompt)

    resp = call_openai_with_retry(_do_request)
    cat = (resp.output_text or "").strip()
    return cat if cat in CATEGORY_ORDER else "Unknown"


def classify_patent_titles_batch(titles: List[str]) -> List[str]:
    clean_titles = [(t or "").strip() for t in titles]

    prompt = (
        "You are a strict classifier.\n"
        f"Classify EACH patent title into exactly ONE of: {', '.join(CATEGORY_ORDER)}.\n"
        "Return ONLY valid JSON: a list of category strings in the SAME ORDER as the titles.\n"
        "No extra text, no markdown.\n\n"
        f"Titles:\n{json.dumps(clean_titles, ensure_ascii=False)}"
    )

    def _do_request():
        return client.responses.create(model="gpt-5.2", input=prompt)

    resp = call_openai_with_retry(_do_request)
    raw = (resp.output_text or "").strip()

    # Expect: ["Chemical", "Software", ...]
    try:
        cats = json.loads(raw)
        if not isinstance(cats, list) or len(cats) != len(clean_titles):
            raise ValueError("Bad JSON response shape")

        out = []
        for c in cats:
            c = (c or "").strip()
            out.append(c if c in CATEGORY_ORDER else "Unknown")
        return out

    except Exception:
        # If JSON parsing fails, fall back to single-title classification
        return [classify_patent_title_one(t) for t in clean_titles]


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("Patent Title Classification App")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    if "Title" not in df.columns or "Publication Number" not in df.columns:
        st.error("CSV must include 'Title' and 'Publication Number' columns.")
        st.stop()

    titles = df["Title"].tolist()
    all_cats: List[str] = []

    progress = st.progress(0)
    status = st.empty()

    with st.spinner("Classifying titles..."):
        total = len(titles)
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            status.write(f"Classifying {start + 1}-{end} of {total}...")
            batch = titles[start:end]

            batch_cats = classify_patent_titles_batch(batch)
            all_cats.extend(batch_cats)

            progress.progress(end / total)

    df["Category"] = all_cats
    status.write("Classification complete ✅")

    # publication number numeric sort key
    df["Publication Number Sort"] = (
        df["Publication Number"].astype(str).str.extract(r"(\d+)").astype(int)
    )

    # group + secondary sort
    df["Category"] = df["Category"].astype(str).str.strip().astype(CATEGORY_TYPE)
    df = (
        df.sort_values(["Category", "Publication Number Sort"])
        .drop(columns=["Publication Number Sort"])
        .reset_index(drop=True)
    )

    st.success("Done!")
    st.dataframe(df)

    out = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download classified CSV",
        data=out,
        file_name="patent_titles_classified.csv",
        mime="text/csv",
    )
