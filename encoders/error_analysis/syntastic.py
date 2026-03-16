"""
Syntactic Error Analysis — Anushtubh Meter Evaluation.
Detailed error pattern analysis with visualizations for generated Sanskrit verse.
"""

import argparse
import torch
from dataclasses import dataclass
from skrutable.meter_identification import MeterIdentifier, VerseTester
from skrutable.scansion import Scanner, Verse
from skrutable.meter_patterns import anuzwuB_pAda
from typing import List, Dict, Tuple
import re
from sentence_transformers import SentenceTransformer
from datetime import datetime
from datasets import Dataset
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


valid_label_regex = re.compile(r'^anuṣṭubh \(\d,\d\: (?!asamīcīna).+?, \d,\d\: (?!asamīcīna).+?\)$')


@dataclass
class MetricsOutput:
    name: str
    meter_verses: List[Verse]
    histogram_lengths: Dict[int, int]
    histogram_labels: Dict[str, int]
    semantic_similarities: torch.Tensor


def evaluate_generated(inputs, poetry_outputs: List[str], dataset_name: str = '', model_name: str = 'sanganaka/bge-m3-sanskritFT'):
    histogram_labels, histogram_lengths, meter_verses = make_anushtup_histograms(poetry_outputs)

    semantic_model = SentenceTransformer(model_name)
    in_embs = semantic_model.encode(inputs, convert_to_tensor=True)
    out_embs = semantic_model.encode(poetry_outputs, convert_to_tensor=True)

    sims = semantic_model.similarity_pairwise(in_embs, out_embs)

    return MetricsOutput(
        name=f"{dataset_name}-{datetime.now()}",
        meter_verses=meter_verses,
        histogram_lengths=histogram_lengths,
        histogram_labels=histogram_labels,
        semantic_similarities=sims,
    )


def make_anushtup_histograms(poetry_outputs: List[str]):
    mi = MeterIdentifier()
    meter_verses = []
    for x in poetry_outputs:
        i = 0
        while i < len(x):
            try:
                meter = mi.identify_meter(x[i:], from_scheme='SLP', resplit_option='resplit_max')
            except:
                i += 1
                continue
            meter_verses.append(meter)
            break
        if i == len(x):
            meter = mi.identify_meter('')
            meter_verses.append(meter)

    histogram_lengths = {}
    histogram_labels = {}
    for x in meter_verses:
        sw = x.syllable_weights.replace('\n','')
        histogram_lengths[len(sw)] = histogram_lengths.get(len(sw), 0) + 1
        histogram_labels[x.meter_label] = histogram_labels.get(x.meter_label, 0) + 1

    return histogram_labels, histogram_lengths, meter_verses


def calculate_anushtup_percentages(
        histogram_labels: Dict[str, int],
        histogram_lengths: Dict[int, int]
    ) -> Tuple[float, float]:
    total_count = 0
    anushtup_count = 0

    for label, count in histogram_labels.items():
        total_count += count
        if valid_label_regex.match(label) is not None:
            anushtup_count += count

    assert total_count != 0
    assert total_count == sum(histogram_lengths.values())

    full_anushtup_percent = anushtup_count * 100 / total_count
    full_length_count = histogram_lengths.get(32, 0)
    partial_anushtup_percent = (full_length_count - anushtup_count) * 100 / total_count

    return full_anushtup_percent, partial_anushtup_percent


def cosine_sim_to_percentage(cos_sim: List[float], reduce=True) -> float:
    dists = ((torch.pi - cos_sim.arccos()) * 100 / torch.pi)
    if reduce:
        dists = dists.mean()
    return dists


def categorize_sample(meter_verse: Verse) -> str:
    if valid_label_regex.match(meter_verse.meter_label) is not None:
        return "Full"
    elif len(meter_verse.syllable_weights.replace('\n','')) == 32:
        return "Partial"
    else:
        return "None"


def analyze_syllable_deviation(syllable_count: int) -> Dict:
    deviation = syllable_count - 32
    deviation_pct = (deviation / 32) * 100

    if syllable_count == 32:
        category = "Perfect"
    elif 28 <= syllable_count <= 36:
        category = "Near-Perfect"
    elif 24 <= syllable_count <= 40:
        category = "Moderate"
    elif syllable_count > 0:
        category = "Poor"
    else:
        category = "Empty"

    return {'deviation': deviation, 'deviation_pct': deviation_pct, 'category': category}


def extract_error_patterns(meter_verses: List[Verse]) -> Dict:
    error_patterns = {
        'too_short': 0, 'too_long': 0,
        'correct_length_wrong_pattern': 0, 'empty': 0, 'unknown': 0
    }
    syllable_errors = []

    for verse in meter_verses:
        syllable_count = len(verse.syllable_weights.replace('\n',''))

        if syllable_count == 0:
            error_patterns['empty'] += 1
        elif syllable_count < 32:
            error_patterns['too_short'] += 1
            syllable_errors.append(syllable_count)
        elif syllable_count > 32:
            error_patterns['too_long'] += 1
            syllable_errors.append(syllable_count)
        elif syllable_count == 32 and valid_label_regex.match(verse.meter_label) is None:
            error_patterns['correct_length_wrong_pattern'] += 1
        elif valid_label_regex.match(verse.meter_label) is None:
            error_patterns['unknown'] += 1
            syllable_errors.append(syllable_count)

    return error_patterns, syllable_errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Syntactic Error Analysis - Anushtubh Meter")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with English and Sanskrit columns")
    parser.add_argument("--english_col", type=str, default="English")
    parser.add_argument("--sanskrit_col", type=str, default="Sanskrit")
    parser.add_argument("--output_prefix", type=str, default="syntactic_error_analysis")
    parser.add_argument("--semantic_model", type=str, default="sanganaka/bge-m3-sanskritFT", help="SentenceTransformer model for semantic similarity")
    args = parser.parse_args()

    OUTPUT_PREFIX = args.output_prefix

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples")

    assert args.english_col in df.columns
    assert args.sanskrit_col in df.columns

    english_inputs = df[args.english_col].tolist()
    sanskrit_outputs = df[args.sanskrit_col].tolist()

    metrics = evaluate_generated(english_inputs, sanskrit_outputs, dataset_name='syntactic_analysis', model_name=args.semantic_model)

    full_anushtup_percent, partial_anushtup_percent = calculate_anushtup_percentages(
        metrics.histogram_labels, metrics.histogram_lengths
    )
    none_percent = 100 - full_anushtup_percent - partial_anushtup_percent
    semantic_sim = cosine_sim_to_percentage(metrics.semantic_similarities)

    total_samples = len(sanskrit_outputs)
    full_count = int(total_samples * full_anushtup_percent / 100)
    partial_count = int(total_samples * partial_anushtup_percent / 100)
    none_count = total_samples - full_count - partial_count

    print(f"\nFull Anushtubh:    {full_count:4d} ({full_anushtup_percent:5.2f}%)")
    print(f"Partial Anushtubh: {partial_count:4d} ({partial_anushtup_percent:5.2f}%)")
    print(f"None:              {none_count:4d} ({none_percent:5.2f}%)")
    print(f"Semantic Similarity: {semantic_sim:.2f}%")

    anushtup_type = []
    syllable_counts = []
    meter_labels = []
    syllable_deviations = []

    for verse in metrics.meter_verses:
        anushtup_type.append(categorize_sample(verse))
        syllable_count = len(verse.syllable_weights.replace('\n',''))
        syllable_counts.append(syllable_count)
        meter_labels.append(verse.meter_label)
        syllable_deviations.append(analyze_syllable_deviation(syllable_count))

    df['anushtup_type'] = anushtup_type
    df['syllable_count'] = syllable_counts
    df['meter_label'] = meter_labels
    df['syllable_deviation'] = [d['deviation'] for d in syllable_deviations]
    df['syllable_deviation_pct'] = [d['deviation_pct'] for d in syllable_deviations]
    df['syllable_category'] = [d['category'] for d in syllable_deviations]
    df['semantic_sim_%'] = cosine_sim_to_percentage(metrics.semantic_similarities, reduce=False).cpu()
    df['sanskrit_length'] = df[args.sanskrit_col].str.len()
    df['sanskrit_words'] = df[args.sanskrit_col].str.split().str.len()
    df['english_length'] = df[args.english_col].str.len()
    df['english_words'] = df[args.english_col].str.split().str.len()

    error_patterns, syllable_errors = extract_error_patterns(metrics.meter_verses)

    print(f"\nError Patterns:")
    for error_type, count in error_patterns.items():
        percentage = count * 100 / total_samples
        print(f"  {error_type.replace('_', ' ').title():30s}: {count:4d} ({percentage:5.2f}%)")

    category_counts = df['syllable_category'].value_counts()

    failed_df = df[df['anushtup_type'] != 'Full'].copy()
    success_df = df[df['anushtup_type'] == 'Full'].copy()

    print(f"\nFailed: {len(failed_df)} ({len(failed_df)/len(df)*100:.2f}%) | Success: {len(success_df)} ({len(success_df)/len(df)*100:.2f}%)")

    plt.figure(figsize=(8, 8))
    type_counts = df['anushtup_type'].value_counts()
    colors = {'Full': '#2ecc71', 'Partial': '#f39c12', 'None': '#e74c3c'}
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
            colors=[colors.get(x, '#95a5a6') for x in type_counts.index], startangle=90)
    plt.title('Anushtubh Type Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}_type_distribution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(syllable_counts, bins=range(0, max(syllable_counts)+2), edgecolor='black', alpha=0.7)
    plt.axvline(32, color='red', linestyle='--', linewidth=2, label='Target: 32 syllables')
    plt.axvline(np.mean(syllable_counts), color='green', linestyle='--', label=f'Mean: {np.mean(syllable_counts):.1f}')
    plt.xlabel('Syllable Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Syllable Counts', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}_syllable_distribution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['syllable_count'], df['semantic_sim_%'],
                         c=df['anushtup_type'].map({'Full': 0, 'Partial': 1, 'None': 2}),
                         cmap='RdYlGn_r', alpha=0.5, s=20)
    plt.axvline(32, color='red', linestyle='--', alpha=0.5, label='Target: 32 syllables')
    plt.xlabel('Syllable Count', fontsize=12)
    plt.ylabel('Semantic Similarity (%)', fontsize=12)
    plt.title('Syllable Count vs Semantic Similarity', fontsize=14)
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Full', 'Partial', 'None'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}_syllable_vs_semantic.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    error_df = pd.DataFrame(list(error_patterns.items()), columns=['Error Type', 'Count'])
    error_df['Percentage'] = error_df['Count'] * 100 / total_samples
    error_df = error_df.sort_values('Count', ascending=True)
    plt.barh(error_df['Error Type'], error_df['Count'])
    plt.xlabel('Number of Samples', fontsize=12)
    plt.title('Error Pattern Distribution', fontsize=14)
    for i, (idx, row) in enumerate(error_df.iterrows()):
        plt.text(row['Count'], i, f" {row['Count']} ({row['Percentage']:.1f}%)", va='center', fontsize=10)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}_error_patterns.png', dpi=300)
    plt.close()

    df_sorted = df.sort_values('syllable_deviation', key=abs, ascending=False)
    df_sorted.to_csv(f'{OUTPUT_PREFIX}_full_results.csv', index=False)
    failed_df.to_csv(f'{OUTPUT_PREFIX}_failed_samples.csv', index=False)
    success_df.to_csv(f'{OUTPUT_PREFIX}_success_samples.csv', index=False)

    summary_stats = {
        'Total Samples': total_samples,
        'Full Anushtubh %': full_anushtup_percent,
        'Partial Anushtubh %': partial_anushtup_percent,
        'None %': none_percent,
        'Mean Syllable Count': np.mean(syllable_counts),
        'Mean Semantic Similarity %': float(semantic_sim),
        'Too Short': error_patterns['too_short'],
        'Too Long': error_patterns['too_long'],
        'Correct Length Wrong Pattern': error_patterns['correct_length_wrong_pattern'],
    }
    stats_df = pd.DataFrame([summary_stats])
    stats_df.to_csv(f'{OUTPUT_PREFIX}_summary_statistics.csv', index=False)

    print(f"\nOutput files saved with prefix: {OUTPUT_PREFIX}")
