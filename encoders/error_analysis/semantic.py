"""
Semantic Error Analysis.
Detailed analysis of cross-encoder semantic similarity scores with visualizations.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def categorize_quality(score):
    if score >= 0.8:
        return 'Excellent'
    elif score >= 0.6:
        return 'Good'
    elif score >= 0.4:
        return 'Fair'
    elif score >= 0.2:
        return 'Poor'
    else:
        return 'Very Poor'


def main():
    parser = argparse.ArgumentParser(description="Semantic Error Analysis")
    parser.add_argument("--model_dir", type=str, required=True, help="Cross-encoder model directory")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with English and Sanskrit columns")
    parser.add_argument("--english_col", type=str, default="English")
    parser.add_argument("--sanskrit_col", type=str, default="Sanskrit")
    parser.add_argument("--output_prefix", type=str, default="error_analysis")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"Total rows: {len(df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    scores = []
    for i in tqdm(range(0, len(df), args.batch_size), desc="Computing similarity"):
        batch = df.iloc[i:i+args.batch_size]
        inputs = tokenizer(
            batch[args.english_col].tolist(),
            batch[args.sanskrit_col].tolist(),
            padding=True, truncation=True, max_length=256, return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        scores.extend(probs)

    df['similarity_score'] = scores

    stats = {
        'Mean': np.mean(scores), 'Median': np.median(scores),
        'Std Dev': np.std(scores), 'Min': np.min(scores), 'Max': np.max(scores),
        'Q1 (25%)': np.percentile(scores, 25), 'Q3 (75%)': np.percentile(scores, 75),
    }

    print(f"\nDistribution Statistics:")
    for key, val in stats.items():
        print(f"  {key:12s}: {val:.4f} ({val*100:.2f}%)")

    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(f'{args.output_prefix}_distribution_stats.csv', index=False)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_analysis = []
    for thresh in thresholds:
        below_count = len(df[df['similarity_score'] < thresh])
        below_pct = (below_count / len(df)) * 100
        threshold_analysis.append({
            'threshold': thresh, 'below_count': below_count,
            'below_percentage': below_pct, 'above_count': len(df) - below_count
        })

    thresh_df = pd.DataFrame(threshold_analysis)
    thresh_df.to_csv(f'{args.output_prefix}_threshold_analysis.csv', index=False)

    df['quality_category'] = df['similarity_score'].apply(categorize_quality)
    category_counts = df['quality_category'].value_counts()

    print("\nQuality Distribution:")
    for cat in ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']:
        count = category_counts.get(cat, 0)
        pct = (count / len(df)) * 100
        print(f"  {cat:12s}: {count:4d} ({pct:5.2f}%)")

    low_threshold = min(0.5, np.percentile(scores, 10))
    low_quality_df = df[df['similarity_score'] < low_threshold].copy()

    df['english_length'] = df[args.english_col].str.len()
    df['sanskrit_length'] = df[args.sanskrit_col].str.len()
    df['english_words'] = df[args.english_col].str.split().str.len()
    df['sanskrit_words'] = df[args.sanskrit_col].str.split().str.len()
    df['length_ratio'] = df['sanskrit_length'] / (df['english_length'] + 1)

    worst_samples = df.nsmallest(20, 'similarity_score')[
        [args.english_col, args.sanskrit_col, 'similarity_score', 'english_words', 'sanskrit_words', 'length_ratio']
    ]
    worst_samples.to_csv(f'{args.output_prefix}_worst_samples.csv', index=False)

    best_samples = df.nlargest(10, 'similarity_score')[
        [args.english_col, args.sanskrit_col, 'similarity_score', 'english_words', 'sanskrit_words']
    ]
    best_samples.to_csv(f'{args.output_prefix}_best_samples.csv', index=False)

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
    plt.axvline(np.median(scores), color='green', linestyle='--', label=f'Median: {np.median(scores):.3f}')
    plt.xlabel('Semantic Similarity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Semantic Similarity Scores', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{args.output_prefix}_distribution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.ylabel('')
    plt.title('Quality Category Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{args.output_prefix}_categories.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(df['english_words'], df['similarity_score'], alpha=0.3, s=10)
    plt.xlabel('English Text Length (words)', fontsize=12)
    plt.ylabel('Semantic Similarity Score', fontsize=12)
    plt.title('Similarity Score vs English Text Length', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{args.output_prefix}_score_vs_length.png', dpi=300)
    plt.close()

    df_output = df[[
        args.english_col, args.sanskrit_col, 'similarity_score', 'quality_category',
        'english_length', 'sanskrit_length', 'english_words', 'sanskrit_words', 'length_ratio'
    ]].sort_values('similarity_score', ascending=False)
    df_output.to_csv(f'{args.output_prefix}_full_results.csv', index=False)

    print(f"\nOutput files saved with prefix: {args.output_prefix}")


if __name__ == "__main__":
    main()
