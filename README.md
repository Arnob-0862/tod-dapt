# TOD-SciBERT

Domain-adaptive pre-training of SciBERT on Transit-Oriented Development research literature.

## Results

| Metric | SciBERT | TOD-SciBERT | Improvement |
|--------|---------|-------------|-------------|
| Perplexity | 10.59 | 6.64 | 37.3% ↓ |
| Retrieval cosine similarity | 0.7046 | 0.7606 | +7.9% |
| t-SNE cluster spread | 16.4 | 6.1 | 62.9% tighter |

## Setup

```bash
pip install -r requirements.txt
```

## Run on Google Colab

Open `colab/TOD_SciBERT_Colab.ipynb` in Google Colab.
Upload model folder and corpus to Google Drive first.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/09b_dapt_train.py` | DAPT training |
| `scripts/11_baseline_perplexity.py` | Perplexity comparison |
| `scripts/09c_compare_models.py` | Retrieval comparison |
| `scripts/13_tsne_visualization.py` | t-SNE plot |
| `scripts/14_generate_survey.py` | Expert survey generator |

## Paper

`paper/TOD_SciBERT_Paper.docx` — ready for Expert Systems with Applications submission.
