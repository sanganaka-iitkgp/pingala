"""
Syntactic Evaluation — Anushtubh Meter Conformance.
Evaluates generated Sanskrit text for meter pattern correctness and semantic similarity.
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


valid_label_regex = re.compile(r'^anuṣṭubh \(\d,\d\: (?!asamīcīna).+?, \d,\d\: (?!asamīcīna).+?\)$')


@dataclass
class MetricsOutput:
    name: str
    meter_verses: List[Verse]
    histogram_lengths: Dict[int, int]
    histogram_labels: Dict[str, int]
    semantic_similarities: torch.Tensor


def evaluate_generated(inputs, poetry_outputs: List[str], dataset_name: str = ''):
    histogram_labels, histogram_lengths, meter_verses = make_anushtup_histograms(poetry_outputs)

    semantic_model = SentenceTransformer('sanganaka/bge-m3-sanskritFT')
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
                meter = mi.identify_meter(x[i:], from_scheme='DEV', resplit_option='resplit_max')
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

    assert total_count != 0, "histogram_labels has a total count of 0."
    assert total_count == sum(histogram_lengths.values()), "total counts of the two histograms are different"

    full_anushtup_percent = anushtup_count * 100 / total_count
    full_length_count = histogram_lengths.get(32, 0)
    partial_anushtup_percent = (full_length_count - anushtup_count) * 100 / total_count

    return full_anushtup_percent, partial_anushtup_percent


def cosine_sim_to_percentage(cos_sim: List[float], reduce=True) -> float:
    dists = ((torch.pi - cos_sim.arccos()) * 100 / torch.pi)
    if reduce:
        dists = dists.mean()
    return dists


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Syntactic Evaluation - Anushtubh Meter")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with English and Sanskrit columns")
    parser.add_argument("--english_col", type=str, default="English")
    parser.add_argument("--sanskrit_col", type=str, default="Sanskrit")
    parser.add_argument("--output_csv", type=str, default="test_evaluation_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples")

    assert args.english_col in df.columns, f"CSV must have '{args.english_col}' column"
    assert args.sanskrit_col in df.columns, f"CSV must have '{args.sanskrit_col}' column"

    english_inputs = df[args.english_col].tolist()
    sanskrit_outputs = df[args.sanskrit_col].tolist()

    metrics = evaluate_generated(english_inputs, sanskrit_outputs, dataset_name='evaluation')

    full_anushtup_percent, partial_anushtup_percent = calculate_anushtup_percentages(
        metrics.histogram_labels,
        metrics.histogram_lengths
    )
    none_percent = 100 - full_anushtup_percent - partial_anushtup_percent
    semantic_sim = cosine_sim_to_percentage(metrics.semantic_similarities)

    print(f"\nFull Anushtubh:    {full_anushtup_percent:.2f}%")
    print(f"Partial Anushtubh: {partial_anushtup_percent:.2f}%")
    print(f"None:              {none_percent:.2f}%")
    print(f"Semantic Similarity: {semantic_sim:.2f}%")

    print(f"\nSyllable Length Histogram:")
    for length in sorted(metrics.histogram_lengths.keys()):
        count = metrics.histogram_lengths[length]
        percentage = count * 100 / len(sanskrit_outputs)
        print(f"  Length {length:2d}: {count:4d} ({percentage:5.2f}%)")

    print(f"\nMeter Labels (top 10):")
    sorted_labels = sorted(metrics.histogram_labels.items(), key=lambda x: x[1], reverse=True)
    for i, (label, count) in enumerate(sorted_labels[:10]):
        percentage = count * 100 / len(sanskrit_outputs)
        print(f"  {i+1:2d}. {label[:50]:50s}: {count:4d} ({percentage:5.2f}%)")

    anushtup_type = []
    for x in metrics.meter_verses:
        a_t = "None"
        if valid_label_regex.match(x.meter_label) is not None:
            a_t = "Full"
        elif len(x.syllable_weights.replace('\n','')) == 32:
            a_t = "Partial"
        anushtup_type.append(a_t)

    results_df = df.copy()
    results_df['anushtup_type'] = anushtup_type
    results_df['semantic_sim_%'] = cosine_sim_to_percentage(metrics.semantic_similarities, reduce=False).cpu()
    results_df['meter_label'] = [x.meter_label for x in metrics.meter_verses]
    results_df['syllable_count'] = [len(x.syllable_weights.replace('\n','')) for x in metrics.meter_verses]

    results_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to: {args.output_csv}")
