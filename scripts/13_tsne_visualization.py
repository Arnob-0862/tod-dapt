"""
13_tsne_visualization.py
========================
t-SNE visualization: SciBERT vs TOD-SciBERT embedding space.

Shows TOD-specific terms cluster tighter in TOD-SciBERT,
proving domain adaptation worked.

Output: E:\TOD_DAPT\comparison\tsne_comparison.png
"""

import os, torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE

# ── Paths ──────────────────────────────────────────────────
SCIBERT_PATH     = "allenai/scibert_scivocab_uncased"
TOD_SCIBERT_PATH = r"E:\TOD_DAPT\model\tod_scibert"
OUTPUT_PATH      = r"E:\TOD_DAPT\comparison\tsne_comparison.png"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ── Words to visualize ─────────────────────────────────────
TOD_TERMS = [
    # TOD core
    "transit-oriented development", "TOD", "station area",
    "transit station", "rail station",
    # Mobility
    "walkability", "pedestrian", "cycling", "bicycle lane",
    "public transit", "bus rapid transit",
    # Land use
    "mixed-use", "land use mix", "density",
    "floor area ratio", "zoning",
    # Sustainability
    "sustainable development", "green space",
    "affordable housing", "urban design",
    # Environment
    "carbon emission", "energy efficiency",
    "parking reduction", "car dependency",
]

NON_TOD_TERMS = [
    # Biology/Medical
    "protein structure", "cell biology", "DNA sequence",
    "enzyme activity", "neural network",
    # Chemistry
    "chemical reaction", "molecular weight",
    "organic compound", "spectroscopy",
    # Physics
    "quantum mechanics", "electromagnetic field",
    "thermodynamics", "particle physics",
    # Computer Science
    "machine learning", "deep learning",
    "algorithm complexity", "data structure",
]

ALL_TERMS   = TOD_TERMS + NON_TOD_TERMS
LABELS      = ["TOD"] * len(TOD_TERMS) + ["Non-TOD"] * len(NON_TOD_TERMS)


def get_embeddings(model, tokenizer, texts, device):
    model.eval()
    embeddings = []
    for text in texts:
        enc = tokenizer(text, return_tensors='pt',
                        truncation=True, max_length=64,
                        padding=True).to(device)
        with torch.no_grad():
            out = model(**enc)
        # CLS token embedding
        emb = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embeddings.append(emb)
    return np.array(embeddings)


def plot_tsne(ax, embs, labels, title, tod_color, nontod_color):
    tsne  = TSNE(n_components=2, perplexity=10, random_state=42,
                 max_iter=2000, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(embs)

    tod_idx    = [i for i, l in enumerate(labels) if l == "TOD"]
    nontod_idx = [i for i, l in enumerate(labels) if l == "Non-TOD"]

    ax.scatter(coords[nontod_idx, 0], coords[nontod_idx, 1],
               c=nontod_color, s=80, alpha=0.7, zorder=2,
               edgecolors='white', linewidths=0.5)
    ax.scatter(coords[tod_idx, 0], coords[tod_idx, 1],
               c=tod_color, s=100, alpha=0.9, zorder=3,
               edgecolors='white', linewidths=0.8)

    # Label TOD terms only
    for i in tod_idx:
        ax.annotate(ALL_TERMS[i], coords[i],
                    fontsize=6.5, alpha=0.85,
                    xytext=(3, 3), textcoords='offset points')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=9)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_facecolor('#f9f9f9')

    # Compute TOD cluster compactness
    tod_coords = coords[tod_idx]
    center     = tod_coords.mean(axis=0)
    spread     = np.sqrt(((tod_coords - center) ** 2).sum(axis=1)).mean()
    ax.text(0.02, 0.02,
            f"TOD cluster spread: {spread:.1f}",
            transform=ax.transAxes,
            fontsize=8, color='gray',
            verticalalignment='bottom')
    return spread


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 55)
    print("  t-SNE Visualization")
    print("  SciBERT  vs  TOD-SciBERT")
    print("=" * 55)
    print(f"\n  Device : {device}")
    print(f"  Terms  : {len(TOD_TERMS)} TOD + {len(NON_TOD_TERMS)} Non-TOD")

    print("\n  Loading SciBERT...")
    sci_tok = AutoTokenizer.from_pretrained(SCIBERT_PATH)
    sci_mod = AutoModel.from_pretrained(SCIBERT_PATH).to(device)

    print("  Loading TOD-SciBERT...")
    tod_tok = AutoTokenizer.from_pretrained(TOD_SCIBERT_PATH)
    tod_mod = AutoModel.from_pretrained(TOD_SCIBERT_PATH).to(device)

    print("\n  Computing embeddings...")
    sci_embs = get_embeddings(sci_mod, sci_tok, ALL_TERMS, device)
    tod_embs = get_embeddings(tod_mod, tod_tok, ALL_TERMS, device)

    print("  Running t-SNE (this takes ~1 min)...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "t-SNE Visualization: TOD vs Non-TOD Term Embeddings\n"
        "SciBERT (left) vs TOD-SciBERT (right)",
        fontsize=14, fontweight='bold', y=1.02
    )

    tod_color    = '#E74C3C'   # red for TOD
    nontod_color = '#95A5A6'   # grey for non-TOD

    sci_spread = plot_tsne(axes[0], sci_embs, LABELS,
                           "SciBERT (Base Model)",
                           tod_color, nontod_color)

    tod_spread = plot_tsne(axes[1], tod_embs, LABELS,
                           "TOD-SciBERT (Domain Adapted)",
                           tod_color, nontod_color)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=tod_color,    label='TOD Terms'),
        mpatches.Patch(facecolor=nontod_color, label='Non-TOD Terms'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.04))

    improvement = (sci_spread - tod_spread) / sci_spread * 100
    fig.text(0.5, -0.08,
             f"TOD cluster spread reduced by {improvement:.1f}%  "
             f"({sci_spread:.1f} → {tod_spread:.1f})  |  "
             f"Tighter cluster = better domain adaptation",
             ha='center', fontsize=10, color='#2C3E50',
             style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()

    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  SciBERT  TOD cluster spread : {sci_spread:.2f}")
    print(f"  TOD-SciBERT cluster spread  : {tod_spread:.2f}")
    print(f"  Improvement                 : {improvement:.1f}% tighter")
    print(f"\n✅ Saved: {OUTPUT_PATH}")
    print(f"\n  Paper-এ লিখবে:")
    print(f"  'Figure X shows t-SNE visualization of term embeddings.")
    print(f"   TOD-SciBERT forms {improvement:.0f}% tighter clusters")
    print(f"   for TOD-specific terms, indicating domain adaptation.'")


if __name__ == "__main__":
    main()
