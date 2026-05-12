import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

DATA_PATH = "data/chm_variants/labels_canonical_with_splits_spatial_ensemble.csv"
OUTPUT_PATH = "LaTeX/Lamapuidu_tuvastamine/estonian/joonised/andmestiku_jaotus.png"
ENSEMBLE_THRESHOLD = 0.40  # Lõpliku ruumilis-ajalise ansambli otsustuspiir.

def create_distribution_plot():
    if not os.path.exists(DATA_PATH):
        print(f"Viga: Andmefaili ei leitud: {DATA_PATH}")
        return

    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(16, 9)) 
    
    # Ensure columns exist, split is probably 'split' and source is 'source', prediction could be 'model_prob' or 'ensemble_prob'
    prob_col = 'ensemble_prob' if 'ensemble_prob' in df.columns else 'model_prob'
    
    # Define subsets
    subsets = {
        "Kogu andmestik": df[prob_col].dropna(),
        "Treeningvalim": df[df['split'] == 'train'][prob_col].dropna(),
        "Testvalim": df[df['split'] == 'test'][prob_col].dropna(),
        "Puhver (välistatud)": df[df['split'] == 'none'][prob_col].dropna(),
        "Käsitsi märgistatud": df[df['source'] == 'manual'][prob_col].dropna(),
    }
    
    colors = ['black', '#1f77b4', '#d62728', '#7f7f7f', '#2ca02c']
    linestyles = ['-', '-', '-', '--', '-.']
    
    for (label, data), color, ls in zip(subsets.items(), colors, linestyles):
        if len(data) > 0:
            sns.kdeplot(data, ax=ax, label=label, color=color, linewidth=2, linestyle=ls, clip=(0, 1))

    ax.axvline(
        ENSEMBLE_THRESHOLD,
        color="#8b0000",
        linestyle="--",
        linewidth=2.2,
        zorder=5,
        label=f"Lõpliku ansambli piirväärtus (t = {ENSEMBLE_THRESHOLD:.2f})",
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Mudeli tõenäosus", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel("Jaotustihedus", fontsize=14, fontweight='bold', labelpad=10)
    
    ax.grid(True, linestyle=':', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    
    ax.legend(title="Andmestiku kategooria", fontsize=12, title_fontsize=13,
              loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True)
    
    sns.despine(trim=False, ax=ax)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Graafik on salvestatud: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_distribution_plot()
