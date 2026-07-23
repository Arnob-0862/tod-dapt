"""
09c_compare_models.py
=====================
Compare SciBERT (base) vs TOD-SciBERT (DAPT) on Q1 retrieval.

Same papers, same queries, same top-15 — only the model changes.

Metrics:
  1. Avg cosine similarity (query vs top-15 sentences)
  2. Overlap@15  (same sentences retrieved by both?)
  3. Intra-list diversity (are retrieved sentences diverse?)
  4. Per-aspect improvement breakdown

Output: E:\TOD_DAPT\comparison\
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

# ── Paths ──────────────────────────────────────────────────
SCIBERT_PATH     = "allenai/scibert_scivocab_uncased"
TOD_SCIBERT_PATH = r"E:\TOD_DAPT\model\tod_scibert"
PDF_FOLDER       = r"E:\TOD_DAPT\papers"
OUTPUT_DIR       = r"E:\TOD_DAPT\comparison"
TOP_K            = 15
SAMPLE_PAPERS    = 20   # first 20 papers for speed

# ── Q1 Aspect Queries (same as pipeline) ──────────────────
ASPECTS = {
    "Land Use Mix":    "mixed land use transit oriented development",
    "Walkability":     "pedestrian walkability street design sidewalk",
    "Transit Access":  "public transit station access bus rail frequency",
    "Density":         "residential commercial density floor area ratio",
    "Parking":         "parking reduction car dependency vehicle ownership",
    "Cycling":         "cycling bicycle infrastructure network lane",
    "Green Space":     "green space park open space environment",
    "Affordability":   "affordable housing income socioeconomic residents",
    "Station Area":    "station area development TOD zone catchment",
    "Urban Design":    "urban design placemaking street quality aesthetics",
}


# ── Helpers ────────────────────────────────────────────────
def extract_sentences(pdf_path):
    try:
        doc  = fitz.open(pdf_path)
        text = "".join(page.get_text("text") + "\n" for page in doc)
        doc.close()
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        sents = sent_tokenize(text)
        return [s.strip() for s in sents if 30 <= len(s.strip()) <= 500]
    except Exception as e:
        print(f"   Error: {e}")
        return []


def mean_pool(token_emb, attn_mask):
    mask = attn_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed(model, tokenizer, texts, device, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc['attention_mask'])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def diversity(emb_matrix):
    """Intra-list diversity = 1 - avg pairwise cosine sim"""
    n = len(emb_matrix)
    if n < 2:
        return 0.0
    sims = [
        float(np.dot(emb_matrix[i], emb_matrix[j]))
        for i in range(n) for j in range(i + 1, n)
    ]
    return 1.0 - float(np.mean(sims))


# ── Main ───────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 55)
    print("  SciBERT vs TOD-SciBERT Comparison")
    print("=" * 55)
    print(f"\n  Device : {device}")

    # Load both models
    print("\n  Loading SciBERT...")
    sci_tok = AutoTokenizer.from_pretrained(SCIBERT_PATH)
    sci_mod = AutoModel.from_pretrained(SCIBERT_PATH).to(device).eval()

    print("  Loading TOD-SciBERT...")
    tod_tok = AutoTokenizer.from_pretrained(TOD_SCIBERT_PATH)
    tod_mod = AutoModel.from_pretrained(TOD_SCIBERT_PATH).to(device).eval()

    # Papers
    pdfs = sorted([f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')])[:SAMPLE_PAPERS]
    print(f"\n  Papers   : {len(pdfs)}")
    print(f"  Aspects  : {len(ASPECTS)}")
    print(f"  Top-K    : {TOP_K}")
    print(f"  Total rows: {len(pdfs) * len(ASPECTS)}\n")

    # Pre-embed aspect queries (same for all papers)
    print("  Embedding aspect queries...")
    aspect_names  = list(ASPECTS.keys())
    aspect_texts  = list(ASPECTS.values())
    sci_q_embs    = embed(sci_mod, sci_tok, aspect_texts, device)
    tod_q_embs    = embed(tod_mod, tod_tok, aspect_texts, device)

    results = []

    for pi, fname in enumerate(pdfs, 1):
        print(f"  [{pi:2d}/{len(pdfs)}] {fname[:55]}")
        sents = extract_sentences(os.path.join(PDF_FOLDER, fname))
        if len(sents) < TOP_K:
            print(f"         skip — only {len(sents)} sentences")
            continue

        sci_s = embed(sci_mod, sci_tok, sents, device)
        tod_s = embed(tod_mod, tod_tok, sents, device)

        for ai, aspect in enumerate(aspect_names):
            sci_q = sci_q_embs[ai]
            tod_q = tod_q_embs[ai]

            sci_scores = sci_s @ sci_q          # cosine (already L2-normed)
            tod_scores = tod_s @ tod_q

            sci_top = np.argsort(sci_scores)[::-1][:TOP_K]
            tod_top = np.argsort(tod_scores)[::-1][:TOP_K]

            sci_avg  = float(sci_scores[sci_top].mean())
            tod_avg  = float(tod_scores[tod_top].mean())
            overlap  = len(set(sci_top.tolist()) & set(tod_top.tolist())) / TOP_K
            sci_div  = diversity(sci_s[sci_top])
            tod_div  = diversity(tod_s[tod_top])

            results.append({
                'paper':           fname,
                'aspect':          aspect,
                'scibert_sim':     round(sci_avg, 4),
                'tod_sim':         round(tod_avg, 4),
                'sim_improvement': round(tod_avg - sci_avg, 4),
                'overlap_at_15':   round(overlap, 4),
                'scibert_div':     round(sci_div, 4),
                'tod_div':         round(tod_div, 4),
            })

    df = pd.DataFrame(results)

    # ── Print summary ────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)

    sci_mean = df['scibert_sim'].mean()
    tod_mean = df['tod_sim'].mean()
    imp_mean = df['sim_improvement'].mean()
    olp_mean = df['overlap_at_15'].mean()
    sci_div  = df['scibert_div'].mean()
    tod_div  = df['tod_div'].mean()

    print(f"\n  SciBERT avg similarity    : {sci_mean:.4f}")
    print(f"  TOD-SciBERT avg similarity: {tod_mean:.4f}")
    print(f"  Improvement               : {imp_mean:+.4f}  "
          f"({'▲ better' if imp_mean > 0 else '▼ worse'})")
    print(f"\n  Overlap@15 (shared sents) : {olp_mean:.1%}")
    print(f"  → TOD-SciBERT retrieved {(1-olp_mean):.1%} different sentences")
    print(f"\n  SciBERT diversity         : {sci_div:.4f}")
    print(f"  TOD-SciBERT diversity     : {tod_div:.4f}")

    print("\n  Per-aspect improvement (sorted):")
    aspect_summary = (
        df.groupby('aspect')['sim_improvement']
          .mean()
          .sort_values(ascending=False)
    )
    for aspect, val in aspect_summary.items():
        bar  = "▲" if val > 0 else "▼"
        print(f"    {bar} {aspect:<22} {val:+.4f}")

    # ── Save outputs ─────────────────────────────────────────
    df.to_csv(os.path.join(OUTPUT_DIR, 'comparison_scibert_vs_tod.csv'), index=False)

    aspect_df = aspect_summary.reset_index()
    aspect_df.columns = ['aspect', 'sim_improvement']
    aspect_df.to_csv(os.path.join(OUTPUT_DIR, 'aspect_improvements.csv'), index=False)

    summary_txt = "\n".join([
        "SciBERT vs TOD-SciBERT Comparison",
        "==================================",
        f"Papers tested        : {df['paper'].nunique()}",
        f"Aspects              : {len(ASPECTS)}",
        f"Top-K                : {TOP_K}",
        "",
        f"SciBERT avg sim      : {sci_mean:.4f}",
        f"TOD-SciBERT avg sim  : {tod_mean:.4f}",
        f"Improvement          : {imp_mean:+.4f}",
        "",
        f"Overlap@15           : {olp_mean:.1%}",
        f"Scibert diversity    : {sci_div:.4f}",
        f"TOD diversity        : {tod_div:.4f}",
        "",
        "Per-aspect improvement:",
        *[f"  {a}: {v:+.4f}" for a, v in aspect_summary.items()],
        "",
        "DAPT Perplexity: 6.39 (vs SciBERT baseline ~18,000)",
    ])
    with open(os.path.join(OUTPUT_DIR, 'comparison_summary.txt'), 'w') as f:
        f.write(summary_txt)

    print(f"\n✅ Saved to: {OUTPUT_DIR}")
    print(f"   comparison_scibert_vs_tod.csv")
    print(f"   aspect_improvements.csv")
    print(f"   comparison_summary.txt")


if __name__ == "__main__":
    main()
