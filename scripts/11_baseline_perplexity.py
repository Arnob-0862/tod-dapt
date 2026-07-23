"""
11_baseline_perplexity.py
=========================
SciBERT vs TOD-SciBERT perplexity comparison
on the TOD domain corpus.

Perplexity = e^(MLM loss)
Lower = model understands the text better

Output: E:\TOD_DAPT\comparison\perplexity_comparison.txt
"""

import os, math, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import Dataset
from torch.utils.data import DataLoader

# ── Paths ──────────────────────────────────────────────────
CORPUS_PATH      = r"E:\TOD_DAPT\corpus\dapt_corpus.txt"
TOD_SCIBERT_PATH = r"E:\TOD_DAPT\model\tod_scibert"
SCIBERT_PATH     = "allenai/scibert_scivocab_uncased"
OUTPUT_PATH      = r"E:\TOD_DAPT\comparison\perplexity_comparison.txt"

EVAL_SAMPLES = 5000   # use 5K sentences for speed
MAX_LENGTH   = 256
BATCH_SIZE   = 32
MLM_PROB     = 0.15


def compute_perplexity(model_path, tokenizer_path, sentences, device, label):
    print(f"\n  [{label}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model     = AutoModelForMaskedLM.from_pretrained(model_path).to(device).eval()

    print(f"  [{label}] Tokenizing {len(sentences):,} sentences...")
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_special_tokens_mask=True,
        )

    dataset   = Dataset.from_dict({"text": sentences})
    tokenized = dataset.map(tokenize_fn, batched=True, batch_size=1000,
                            remove_columns=["text"], desc="Tokenizing")

    collator  = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB
    )
    loader = DataLoader(tokenized, batch_size=BATCH_SIZE,
                        collate_fn=collator, shuffle=False)

    print(f"  [{label}] Computing MLM loss...")
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss    += outputs.loss.item()
            total_batches += 1

    avg_loss   = total_loss / total_batches
    perplexity = math.exp(avg_loss)

    print(f"  [{label}] Loss: {avg_loss:.4f}  |  Perplexity: {perplexity:.2f}")
    return avg_loss, perplexity


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("=" * 55)
    print("  Perplexity Comparison")
    print("  SciBERT  vs  TOD-SciBERT")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    # Load corpus
    print(f"\n  Loading corpus: {CORPUS_PATH}")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        all_sents = [l.strip() for l in f if l.strip()]

    # Use fixed subset for fair comparison
    import random
    random.seed(42)
    sents = random.sample(all_sents, min(EVAL_SAMPLES, len(all_sents)))
    print(f"  Eval sentences: {len(sents):,}")

    # SciBERT baseline
    sci_loss, sci_ppl = compute_perplexity(
        SCIBERT_PATH, SCIBERT_PATH, sents, device, "SciBERT"
    )

    # TOD-SciBERT
    tod_loss, tod_ppl = compute_perplexity(
        TOD_SCIBERT_PATH, TOD_SCIBERT_PATH, sents, device, "TOD-SciBERT"
    )

    # Results
    improvement = (sci_ppl - tod_ppl) / sci_ppl * 100

    print("\n" + "=" * 55)
    print("  RESULTS")
    print("=" * 55)
    print(f"\n  {'Model':<20} {'Loss':>8}  {'Perplexity':>12}")
    print(f"  {'-'*42}")
    print(f"  {'SciBERT':<20} {sci_loss:>8.4f}  {sci_ppl:>12.2f}")
    print(f"  {'TOD-SciBERT':<20} {tod_loss:>8.4f}  {tod_ppl:>12.2f}")
    print(f"  {'-'*42}")
    print(f"  Improvement: {improvement:.1f}% reduction in perplexity")
    print(f"\n  Interpretation:")
    print(f"  SciBERT was confused by TOD text ({sci_ppl:.0f} options per token)")
    print(f"  TOD-SciBERT understands TOD text ({tod_ppl:.2f} options per token)")

    # Save
    result = "\n".join([
        "Perplexity Comparison: SciBERT vs TOD-SciBERT",
        "=" * 50,
        f"Eval sentences : {len(sents):,}",
        f"MLM probability: {MLM_PROB}",
        f"Max length     : {MAX_LENGTH}",
        "",
        f"{'Model':<20} {'Loss':>8}  {'Perplexity':>12}",
        f"{'-'*42}",
        f"{'SciBERT':<20} {sci_loss:>8.4f}  {sci_ppl:>12.2f}",
        f"{'TOD-SciBERT':<20} {tod_loss:>8.4f}  {tod_ppl:>12.2f}",
        f"{'-'*42}",
        f"Perplexity reduction: {improvement:.1f}%",
        "",
        "Interpretation:",
        f"  DAPT reduced perplexity from {sci_ppl:.2f} to {tod_ppl:.2f}",
        f"  TOD-SciBERT understands TOD domain text {improvement:.1f}% better",
    ])

    with open(OUTPUT_PATH, 'w') as f:
        f.write(result)

    print(f"\n✅ Saved: {OUTPUT_PATH}")
    print(f"\n  Paper-এ লিখবে:")
    print(f"  'TOD-SciBERT achieved a perplexity of {tod_ppl:.2f},")
    print(f"   compared to SciBERT baseline of {sci_ppl:.2f},")
    print(f"   representing a {improvement:.1f}% improvement in")
    print(f"   TOD domain language understanding.'")


if __name__ == "__main__":
    main()
