"""
Semantic Similarity Evaluation using Cross-Encoder.
Computes similarity scores between English inputs and generated Sanskrit outputs.
"""

import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    parser = argparse.ArgumentParser(description="Semantic Similarity Evaluation")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to cross-encoder model directory")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV with English and Sanskrit columns")
    parser.add_argument("--english_col", type=str, default="English", help="English column name")
    parser.add_argument("--sanskrit_col", type=str, default="Sanskrit_generated", help="Sanskrit column name")
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

    for i in tqdm(range(0, len(df), args.batch_size), desc="Computing semantic similarity"):
        batch = df.iloc[i:i+args.batch_size]

        inputs = tokenizer(
            batch[args.english_col].tolist(),
            batch[args.sanskrit_col].tolist(),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        scores.extend(probs)

    avg_similarity = sum(scores) / len(scores)
    print(f"\nAverage Semantic Similarity: {avg_similarity:.4f} ({avg_similarity*100:.2f}%)")


if __name__ == "__main__":
    main()
