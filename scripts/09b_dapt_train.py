"""
09b_dapt_train.py
=================
Domain Adaptive Pre-Training (DAPT) of SciBERT
on TOD domain corpus (~98K sentences, 2.3M words).

Unsupervised MLM — no labels, no selection bias.
SciBERT learns TOD-specific vocabulary and context.

Hardware: RTX 3060 12GB
Estimated time: ~2-3 hours (3 epochs)

Output: E:\TOD_DAPT\model\tod_scibert\
"""

import os, math
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# ── Paths ──────────────────────────────────────────────────
CORPUS_PATH = r"E:\TOD_DAPT\corpus\dapt_corpus.txt"
OUTPUT_DIR  = r"E:\TOD_DAPT\model\tod_scibert"
BASE_MODEL  = "allenai/scibert_scivocab_uncased"

# ── Hyperparameters (tuned for RTX 3060 12GB) ─────────────
MAX_LENGTH    = 256   # tokens per sequence
BATCH_SIZE    = 16    # per device
GRAD_ACCUM    = 4     # effective batch = 64
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-5
MLM_PROB      = 0.15  # standard BERT masking rate
WARMUP_RATIO  = 0.06

def main():
    print("=" * 55)
    print("  TOD-SciBERT DAPT Training")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device : {device}")
    if device == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {vram:.1f} GB")

    # ── Load corpus ───────────────────────────────────────
    print(f"\n  Loading corpus...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        sentences = [l.strip() for l in f if l.strip()]
    print(f"  Sentences: {len(sentences):,}")

    # ── Tokenizer & Model ─────────────────────────────────
    print(f"  Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    params    = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.1f}M")

    # ── Tokenize ──────────────────────────────────────────
    print("  Tokenizing dataset...")
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_special_tokens_mask=True,
        )

    dataset  = Dataset.from_dict({"text": sentences})
    tokenized = dataset.map(
        tokenize_fn, batched=True, batch_size=1000,
        remove_columns=["text"], desc="Tokenizing",
    )

    split    = tokenized.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"  Train: {len(train_ds):,}  |  Eval: {len(eval_ds):,}")

    # ── Data collator (MLM masking) ───────────────────────
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB,
    )

    # ── Training config ───────────────────────────────────
    steps_per_epoch = math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = int(total_steps * WARMUP_RATIO)

    print(f"\n  Epochs        : {NUM_EPOCHS}")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Total steps   : {total_steps}")
    print(f"  Warmup steps  : {warmup_steps}")

    args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LEARNING_RATE,
        weight_decay                = 0.01,
        warmup_steps                = warmup_steps,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        logging_steps               = 50,
        fp16                        = (device == "cuda"),
        dataloader_num_workers      = 0,
        report_to                   = "none",
        save_total_limit            = 3,  # 3 epoch = 3 checkpoint সব রাখো
    )

    trainer = Trainer(
        model              = model,
        args               = args,
        train_dataset      = train_ds,
        eval_dataset       = eval_ds,
        data_collator      = collator,
        processing_class   = tokenizer,
    )

    # ── Train ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  TRAINING STARTED — estimated 2-3 hours")
    print(f"{'='*55}\n")

    result = trainer.train()

    print(f"\n{'='*55}")
    print("  TRAINING COMPLETE")
    print(f"{'='*55}")
    runtime_min = result.metrics.get('train_runtime', 0) / 60
    print(f"  Training loss : {result.training_loss:.4f}")
    print(f"  Training time : {runtime_min:.1f} min")

    # ── Save ──────────────────────────────────────────────
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ TOD-SciBERT saved: {OUTPUT_DIR}")

    # ── Perplexity ────────────────────────────────────────
    eval_results = trainer.evaluate()
    perplexity   = math.exp(eval_results["eval_loss"])
    print(f"\n  Eval loss   : {eval_results['eval_loss']:.4f}")
    print(f"  Perplexity  : {perplexity:.2f}")
    print("  (lower = better domain adaptation)")
    print(f"\n  Next step: run 09c_compare_models.py")
    print(f"  to compare SciBERT vs TOD-SciBERT on Q1 retrieval")

if __name__ == "__main__":
    main()
