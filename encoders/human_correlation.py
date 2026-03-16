"""
Human Correlation Analysis.
Compares cross-encoder and bi-encoder scores against human ratings.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Human Correlation Analysis")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with score, semantic_sim, bi_encoder_score columns")
    parser.add_argument("--output_plot", type=str, default="human_vs_model_scores.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"N samples: {len(df)}")

    print(f"Cross-encoder: mean={df.semantic_sim.mean():.4f}, std={df.semantic_sim.std():.4f}")
    print(f"Bi-encoder:    mean={df.bi_encoder_score.mean():.4f}, std={df.bi_encoder_score.std():.4f}")

    print("\nMean model scores per human rating:")
    for r in sorted(df.score.unique()):
        mask = df.score == r
        print(f"  Rating {r}: n={mask.sum()}, CE={df.semantic_sim[mask].mean():.4f}, BE={df.bi_encoder_score[mask].mean():.4f}")

    df['ce_to_rating'] = 1.0 + 4.0 * df.semantic_sim
    df['be_to_rating'] = 1.0 + 4.0 * (df.bi_encoder_score - df.bi_encoder_score.min()) / (df.bi_encoder_score.max() - df.bi_encoder_score.min())

    print("\nCorrelations with human scores:")
    print(f"  CE (probs) Pearson={pearsonr(df.score, df.semantic_sim)[0]:.4f}, Spearman={spearmanr(df.score, df.semantic_sim)[0]:.4f}")
    print(f"  BE (sims)  Pearson={pearsonr(df.score, df.bi_encoder_score)[0]:.4f}, Spearman={spearmanr(df.score, df.bi_encoder_score)[0]:.4f}")

    y_norm = (df.score - 1.0) / 4.0

    iso_ce = IsotonicRegression(out_of_bounds='clip').fit(df.semantic_sim.values, y_norm.values)
    df['ce_iso_rating'] = 1.0 + 4.0 * iso_ce.predict(df.semantic_sim.values)
    print(f"\nIsotonic calibrated CE: Pearson={pearsonr(df.score, df.ce_iso_rating)[0]:.4f}, Spearman={spearmanr(df.score, df.ce_iso_rating)[0]:.4f}")

    iso_be = IsotonicRegression(out_of_bounds='clip').fit(df.bi_encoder_score.values, y_norm.values)
    df['be_iso_rating'] = 1.0 + 4.0 * iso_be.predict(df.bi_encoder_score.values)
    print(f"Isotonic calibrated BE: Pearson={pearsonr(df.score, df.be_iso_rating)[0]:.4f}, Spearman={spearmanr(df.score, df.be_iso_rating)[0]:.4f}")

    plt.figure(figsize=(5, 4))
    plt.scatter(df.score, df.semantic_sim, alpha=0.6, label='CE probs')
    plt.scatter(df.score, df.bi_encoder_score, alpha=0.6, label='BE sim')
    plt.xlabel('Human score (1..5)')
    plt.ylabel('Model score')
    plt.legend()
    plt.title('Human score vs model scores')
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300)
    print(f"\nPlot saved to: {args.output_plot}")


if __name__ == "__main__":
    main()
