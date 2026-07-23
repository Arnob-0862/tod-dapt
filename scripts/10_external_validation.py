"""
10_external_validation.py
=========================
External validation on 50 unseen TOD papers.

Addresses circular reasoning:
  Training set (154 papers) → factors extracted → ranked
  Test set     (50 papers)  → same pipeline     → ranked
  Compare: are the top factors consistent?

Output: E:\TOD_DAPT\validation\
"""

import os, re, math, torch
import numpy as np
import pandas as pd
import fitz
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# ── Paths ──────────────────────────────────────────────────
TEST_PDF_FOLDER  = r"E:\TOD_DAPT\Test"
TOD_SCIBERT_PATH = r"E:\TOD_DAPT\model\tod_scibert"
TRAIN_RANKING    = r"E:\Level 4\Thesis\Game_Start\keyword extrract\scibert_pipeline\output\factor_ranking_report.csv"
OUTPUT_DIR       = r"E:\TOD_DAPT\validation"

TOP_K   = 15
TOP_N   = 20   # top factors to compare

# ── Q1 Aspect Queries ──────────────────────────────────────
ASPECTS = {
    "Land Use Mix":   "mixed land use transit oriented development",
    "Walkability":    "pedestrian walkability street design sidewalk",
    "Transit Access": "public transit station access bus rail frequency",
    "Density":        "residential commercial density floor area ratio",
    "Parking":        "parking reduction car dependency vehicle ownership",
    "Cycling":        "cycling bicycle infrastructure network lane",
    "Green Space":    "green space park open space environment",
    "Affordability":  "affordable housing income socioeconomic residents",
    "Station Area":   "station area development TOD zone catchment",
    "Urban Design":   "urban design placemaking street quality aesthetics",
}

# ── Q2 Sustainability variable trigger words ───────────────
Q2_TRIGGERS = [
    'factor', 'indicator', 'variable', 'criterion', 'measure',
    'index', 'metric', 'parameter', 'dimension', 'aspect',
    'contribut', 'determin', 'influenc', 'affect', 'impact',
    'sustain', 'green', 'environment', 'social', 'economic',
    'transit', 'walkab', 'densit', 'mix', 'access',
]

GARBAGE_PATTERNS = [
    r'^\s*(fig(ure)?|table|fig\.)\s*\d',
    r'this\s+(study|research|work)\s+(was|is)\s+(funded|supported)',
    r'doi\s*:?\s*10\.',
    r'issn\s*[\d\-]+',
    r'https?://',
    r'©\s*\d{4}',
    r'^\s*\d+\s*$',
    r'^\s*[A-Z][a-z]+,\s+[A-Z]\.',
]


# ── Helpers ────────────────────────────────────────────────
def is_garbage(text):
    t = text.strip()
    if len(t) < 30 or len(t) > 600:
        return True
    if sum(c.isdigit() for c in t) / max(len(t), 1) > 0.3:
        return True
    for pat in GARBAGE_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return True
    return False


def extract_sentences(pdf_path):
    try:
        doc  = fitz.open(pdf_path)
        text = "".join(page.get_text("text") + "\n" for page in doc)
        doc.close()
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        sents = sent_tokenize(text)
        return [s.strip() for s in sents if not is_garbage(s)]
    except Exception as e:
        print(f"   PDF error: {e}")
        return []


def mean_pool(token_emb, attn_mask):
    mask = attn_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed(model, tokenizer, texts, device, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc['attention_mask'])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def has_trigger(text):
    t = text.lower()
    return any(tr in t for tr in Q2_TRIGGERS)


def extract_keyphrases(sentences, top_n=10):
    if not sentences:
        return []
    try:
        vec = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=500,
            stop_words='english',
            sublinear_tf=True,
        )
        X = vec.fit_transform(sentences)
        scores = X.mean(axis=0).A1
        terms  = vec.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda x: -x[1])
        return [t for t, _ in ranked[:top_n]]
    except:
        return []


# ── Main ───────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  External Validation — 50 Unseen Papers")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    print("  Loading TOD-SciBERT...")
    tok = AutoTokenizer.from_pretrained(TOD_SCIBERT_PATH)
    mod = AutoModel.from_pretrained(TOD_SCIBERT_PATH).to(device).eval()

    # Pre-embed aspect queries
    aspect_names = list(ASPECTS.keys())
    aspect_texts = list(ASPECTS.values())
    q_embs = embed(mod, tok, aspect_texts, device)

    pdfs = sorted([f for f in os.listdir(TEST_PDF_FOLDER) if f.endswith('.pdf')])
    print(f"  Papers: {len(pdfs)}\n")

    all_factors = []   # (paper, aspect, phrase)
    q1_rows     = []
    q2_rows     = []

    for pi, fname in enumerate(pdfs, 1):
        print(f"  [{pi:2d}/{len(pdfs)}] {fname[:55]}")
        sents = extract_sentences(os.path.join(TEST_PDF_FOLDER, fname))
        if len(sents) < 5:
            print(f"         skip — only {len(sents)} sentences")
            continue

        s_embs = embed(mod, tok, sents, device)

        for ai, aspect in enumerate(aspect_names):
            # Q1: top sentences per aspect
            scores  = s_embs @ q_embs[ai]
            top_idx = np.argsort(scores)[::-1][:TOP_K]
            top_sents = [sents[i] for i in top_idx if scores[i] > 0.40]

            for s in top_sents:
                q1_rows.append({'paper': fname, 'aspect': aspect, 'sentence': s})

            # Q2: sustainability variable extraction from Q1 sentences
            q2_sents = [s for s in top_sents if has_trigger(s)]
            phrases  = extract_keyphrases(q2_sents, top_n=8)

            for ph in phrases:
                q2_rows.append({'paper': fname, 'aspect': aspect, 'variable': ph})
                all_factors.append(ph)

    # ── Factor ranking (Frequency) ─────────────────────────
    freq = Counter(all_factors)
    test_ranking = pd.DataFrame(
        freq.most_common(TOP_N),
        columns=['factor', 'test_frequency']
    )
    test_ranking['test_rank'] = range(1, len(test_ranking) + 1)

    # ── Load training ranking ──────────────────────────────
    print(f"\n  Loading training ranking: {TRAIN_RANKING}")
    try:
        train_df = pd.read_csv(TRAIN_RANKING)
        # find the factor/variable column
        factor_col = [c for c in train_df.columns
                      if 'factor' in c.lower() or 'variable' in c.lower()][0]
        rank_col   = [c for c in train_df.columns if 'rank' in c.lower()][0]
        train_top  = train_df[[factor_col, rank_col]].head(TOP_N).copy()
        train_top.columns = ['factor', 'train_rank']
        has_train = True
    except Exception as e:
        print(f"  Warning: could not load training ranking — {e}")
        has_train = False

    # ── Print test results ─────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  TOP {TOP_N} FACTORS — EXTERNAL TEST SET")
    print("=" * 55)
    for _, row in test_ranking.iterrows():
        print(f"  {int(row['test_rank']):2d}. {row['factor']:<35} ({int(row['test_frequency'])} papers)")

    # ── Overlap with training ──────────────────────────────
    if has_train:
        train_factors = set(train_top['factor'].str.lower().tolist())
        test_factors  = set(test_ranking['factor'].str.lower().tolist())
        overlap       = train_factors & test_factors
        overlap_pct   = len(overlap) / TOP_N * 100

        print(f"\n  Overlap with training top-{TOP_N}: {len(overlap)}/{TOP_N} ({overlap_pct:.1f}%)")
        print(f"  Common factors: {', '.join(sorted(overlap)[:5])}...")

    # ── Save outputs ───────────────────────────────────────
    test_ranking.to_csv(os.path.join(OUTPUT_DIR, 'test_factor_ranking.csv'), index=False)
    pd.DataFrame(q1_rows).to_csv(os.path.join(OUTPUT_DIR, 'test_q1_sentences.csv'), index=False)
    pd.DataFrame(q2_rows).to_csv(os.path.join(OUTPUT_DIR, 'test_q2_variables.csv'), index=False)

    # Summary text
    lines = [
        "External Validation Summary",
        "===========================",
        f"Test papers       : {len(pdfs)}",
        f"Total factors     : {len(all_factors)}",
        f"Unique factors    : {len(freq)}",
        "",
        f"Top-{TOP_N} Test Factors:",
        *[f"  {r['test_rank']:2.0f}. {r['factor']} ({r['test_frequency']:.0f})"
          for _, r in test_ranking.iterrows()],
    ]
    if has_train:
        lines += [
            "",
            f"Overlap with training top-{TOP_N}: {len(overlap)}/{TOP_N} ({overlap_pct:.1f}%)",
        ]

    with open(os.path.join(OUTPUT_DIR, 'validation_summary.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✅ Saved: {OUTPUT_DIR}")
    print(f"   test_factor_ranking.csv")
    print(f"   test_q1_sentences.csv")
    print(f"   test_q2_variables.csv")
    print(f"   validation_summary.txt")


if __name__ == "__main__":
    main()
