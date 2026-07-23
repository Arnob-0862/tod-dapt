# TOD-SciBERT — Project Context
> Paste this file into a new Claude Code session to continue from where you left off.

---

## What This Project Is

TOD-SciBERT is a domain-adapted language model for Transit-Oriented Development research.
Base model: SciBERT → continued MLM pre-training on 154 TOD papers → TOD-SciBERT.
Target journal: Expert Systems with Applications (Q1, IF 8.5).

---

## Completed Work

### 1. Model Training ✅
- Script: `scripts/09b_dapt_train.py`
- Trained model saved: `E:\TOD_DAPT\model\tod_scibert\` (also on Google Drive)
- Config: batch=64, lr=2e-5, epochs=3, mlm=0.15, seq_len=256, RTX 3060, 38.5 min

### 2. Evaluation ✅

**Perplexity** (`scripts/11_baseline_perplexity.py`):
- SciBERT: 10.59 | TOD-SciBERT: 6.64 | Improvement: **37.3%**

**Retrieval** (`scripts/09c_compare_models.py`):
- SciBERT: 0.7046 | TOD-SciBERT: 0.7606 | Improvement: **+7.9%**
- Tested on 10 TOD aspects (density, walkability, transit access, etc.)

**t-SNE** (`scripts/13_tsne_visualization.py`):
- Cluster spread: 16.4 → 6.1 | Improvement: **62.9% tighter**
- Output: `E:\TOD_DAPT\comparison\tsne_comparison.png`

### 3. Expert Survey ✅
- Script: `scripts/14_generate_survey.py`
- 15 TOD **statements** (not questions) — blind A/B comparison
- Survey file: `E:\TOD_DAPT\survey\expert_survey.docx`
- Answer key: `E:\TOD_DAPT\survey\answer_key.txt` (private — not on GitHub)

### 4. Paper ✅
- File: `paper/TOD_SciBERT_Paper.docx`
- All sections complete: Abstract, Intro, Related Work, Methodology, Results, Discussion, Conclusion, References
- 3 tables, 16 citations

---

## File Structure

```
E:\TOD_DAPT\
├── model\tod_scibert\        # Trained model (~500MB) — on Google Drive
├── corpus\dapt_corpus.txt    # 109,398 sentences — on Google Drive
├── scripts\
│   ├── 09b_dapt_train.py     # Training
│   ├── 09c_compare_models.py # Retrieval comparison
│   ├── 11_baseline_perplexity.py
│   ├── 13_tsne_visualization.py
│   └── 14_generate_survey.py # Survey (statements-based)
├── paper\
│   └── TOD_SciBERT_Paper.docx
├── survey\
│   ├── expert_survey.docx
│   └── answer_key.txt
├── colab\
│   └── TOD_SciBERT_Colab.ipynb  # Run everything from Google Colab
└── PROJECT_CONTEXT.md            # This file
```

---

## GitHub Repo
https://github.com/Arnob-0862/tod-dapt

## Google Drive
Folder: `TOD_DAPT/`
Contains: `model/tod_scibert/`, `corpus/dapt_corpus.txt`, survey, paper

---

## What Still Needs to Be Done

- [ ] Collect expert survey responses (send `expert_survey.docx` to TOD experts)
- [ ] Add survey results to paper Section 5 (Discussion)
- [ ] Add t-SNE figure (`tsne_comparison.png`) as Figure 1 in paper Section 4.3
- [ ] Final proofreading before submission
- [ ] Submit to Expert Systems with Applications

---

## Key Numbers for Paper

| Metric | SciBERT | TOD-SciBERT | Change |
|--------|---------|-------------|--------|
| Perplexity | 10.59 | 6.64 | -37.3% |
| Retrieval similarity | 0.7046 | 0.7606 | +7.9% |
| t-SNE cluster spread | 16.4 | 6.1 | -62.9% |
| Training time | — | 38.5 min | RTX 3060 |
| Corpus | — | 154 papers, 109,398 sentences, 2.34M words | — |

---

## How to Continue in a New Session

1. Paste this file into a new Claude Code chat
2. Say what you want to do next (e.g., "survey results add koro paper-e")
3. For running code: open `colab/TOD_SciBERT_Colab.ipynb` in Google Colab
