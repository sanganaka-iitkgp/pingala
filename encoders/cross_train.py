"""
Cross-Encoder Training — FAISS-Optimized Hard Negative Mining
Trains a cross-encoder for English-Sanskrit semantic similarity.
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
import torch.nn.functional as F
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class Config:
    english_col = "english"
    sanskrit_col = "sanskrit"

    dev_split = 0.02
    test_split = 0.01

    easy_neg_ratio = 1
    hard_neg_ratio = 2
    mining_model = "sanganaka/bge-m3-sanskritFT"

    model_name = "microsoft/infoxlm-large"
    max_length = 256

    num_epochs = 8
    batch_size = 16
    gradient_accumulation_steps = 2
    learning_rate = 2e-5
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0

    use_class_weights = True
    pos_weight = 6.0

    use_focal_loss = True
    focal_alpha = 0.3
    focal_gamma = 3.0

    use_margin_loss = True
    margin_weight = 0.5
    target_margin = 0.8

    output_dir = "cross_encoder_optimized_high_positive"

    fp16 = torch.cuda.is_available()


config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"


class AdvancedLossTrainer(Trainer):
    def __init__(self,
                 pos_weight=6.0,
                 use_focal=True,
                 focal_alpha=0.3,
                 focal_gamma=3.0,
                 use_margin=True,
                 margin_weight=0.5,
                 target_margin=0.8,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_margin = use_margin
        self.margin_weight = margin_weight
        self.target_margin = target_margin

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        probs = F.softmax(logits, dim=-1)
        pos_probs = probs[:, 1]

        if self.use_focal:
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_term = (1 - pt) ** self.focal_gamma

            alpha_t = torch.where(
                labels == 1,
                torch.tensor(self.focal_alpha * self.pos_weight, device=logits.device),
                torch.tensor(1.0 - self.focal_alpha, device=logits.device)
            )

            focal_loss = (alpha_t * focal_term * ce_loss).mean()
        else:
            weight = torch.tensor([1.0, self.pos_weight], device=logits.device)
            focal_loss = F.cross_entropy(logits, labels, weight=weight)

        margin_loss = torch.tensor(0.0, device=logits.device)
        if self.use_margin and labels.sum() > 0 and (labels == 0).sum() > 0:
            pos_mask = labels == 1
            neg_mask = labels == 0

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_mean = pos_probs[pos_mask].mean()
                neg_mean = pos_probs[neg_mask].mean()
                current_margin = pos_mean - neg_mean
                margin_loss = F.relu(self.target_margin - current_margin)

        total_loss = focal_loss + self.margin_weight * margin_loss

        return (total_loss, outputs) if return_outputs else total_loss


def load_data(csv_path: str):
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df[[config.english_col, config.sanskrit_col]].dropna().reset_index(drop=True)
    df.columns = ['English', 'Sanskrit']
    df['English'] = df['English'].str.strip()
    df['Sanskrit'] = df['Sanskrit'].str.strip()

    df['Sanskrit'] = df['Sanskrit'].apply(lambda x: transliterate(x, sanscript.IAST, sanscript.DEVANAGARI))

    print(f"Loaded {len(df):,} parallel pairs")
    return df


def create_negatives_curriculum(df: pd.DataFrame):
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS is required. Install with: pip install faiss-gpu")

    english = df['English'].tolist()
    sanskrit = df['Sanskrit'].tolist()
    N = len(english)

    rows = []

    for i in tqdm(range(N), desc="Easy negatives"):
        en = english[i]
        sa_correct = sanskrit[i]

        rows.append({'English': en, 'Sanskrit': sa_correct, 'label': 1, 'type': 'positive'})

        for _ in range(config.easy_neg_ratio):
            rand_idx = random.randint(0, N-1)
            while rand_idx == i:
                rand_idx = random.randint(0, N-1)
            rows.append({'English': en, 'Sanskrit': sanskrit[rand_idx], 'label': 0, 'type': 'easy_neg'})

    if config.hard_neg_ratio > 0:
        embedder = SentenceTransformer(config.mining_model, device=device)

        emb_en = embedder.encode(english, batch_size=128, show_progress_bar=True,
                                convert_to_numpy=True, normalize_embeddings=True)

        emb_sa = embedder.encode(sanskrit, batch_size=128, show_progress_bar=True,
                                convert_to_numpy=True, normalize_embeddings=True)

        dim = emb_sa.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb_sa.astype('float32'))

        K = 50
        batch_size = 4096

        for start in tqdm(range(0, N, batch_size), desc="FAISS hard negative mining"):
            end = min(start + batch_size, N)
            query_batch = emb_en[start:end].astype('float32')
            scores, indices = index.search(query_batch, K)

            for i in range(len(query_batch)):
                global_i = start + i
                en = english[global_i]

                candidates = [int(idx) for idx in indices[i] if int(idx) != global_i]

                if len(candidates) < config.hard_neg_ratio:
                    while len(candidates) < config.hard_neg_ratio:
                        rand_idx = random.randint(0, N-1)
                        if rand_idx != global_i and rand_idx not in candidates:
                            candidates.append(rand_idx)

                selected = random.sample(candidates[:30], min(config.hard_neg_ratio, len(candidates)))

                for neg_idx in selected:
                    rows.append({
                        'English': en,
                        'Sanskrit': sanskrit[neg_idx],
                        'label': 0,
                        'type': 'hard_neg'
                    })

        del embedder, emb_en, emb_sa, index
        torch.cuda.empty_cache()

    df_full = pd.DataFrame(rows)
    df_full = df_full.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"Dataset: {len(df_full):,} total | Pos: {(df_full['label']==1).sum():,} | Easy neg: {(df_full['type']=='easy_neg').sum():,} | Hard neg: {(df_full['type']=='hard_neg').sum():,}")
    return df_full


def split_data(df: pd.DataFrame):
    n = len(df)
    test_size = int(n * config.test_split)
    dev_size = int(n * config.dev_split)

    df_test = df.iloc[:test_size].reset_index(drop=True)
    df_dev = df.iloc[test_size:test_size+dev_size].reset_index(drop=True)
    df_train = df.iloc[test_size+dev_size:].reset_index(drop=True)

    print(f"Train: {len(df_train):,} | Dev: {len(df_dev):,} | Test: {len(df_test):,}")
    return df_train, df_dev, df_test


def create_datasets(df_train, df_dev, df_test, tokenizer):
    def tokenize_df(df):
        ds = Dataset.from_pandas(df[['English', 'Sanskrit', 'label']])

        def tokenize_fn(batch):
            return tokenizer(
                batch['English'],
                batch['Sanskrit'],
                padding='max_length',
                truncation=True,
                max_length=config.max_length
            )

        ds = ds.map(tokenize_fn, batched=True, remove_columns=['English', 'Sanskrit'])
        ds = ds.rename_column('label', 'labels')
        ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        return ds

    train_ds = tokenize_df(df_train)
    dev_ds = tokenize_df(df_dev)
    test_ds = tokenize_df(df_test)

    return train_ds, dev_ds, test_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    acc = accuracy_score(labels, preds)

    if preds.sum() == 0:
        f1 = prec = 0.0
    else:
        f1 = f1_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)

    rec = recall_score(labels, preds, zero_division=0)

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_acc = (preds[pos_mask] == labels[pos_mask]).mean() if pos_mask.sum() > 0 else 0
    neg_acc = (preds[neg_mask] == labels[neg_mask]).mean() if neg_mask.sum() > 0 else 0

    pos_prob_avg = probs[pos_mask].mean() if pos_mask.sum() > 0 else 0
    neg_prob_avg = probs[neg_mask].mean() if neg_mask.sum() > 0 else 0

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
        'pos_accuracy': pos_acc,
        'neg_accuracy': neg_acc,
        'pos_prob_avg': pos_prob_avg,
        'neg_prob_avg': neg_prob_avg,
        'prob_gap': pos_prob_avg - neg_prob_avg
    }


def train_cross_encoder(train_ds, dev_ds):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="prob_gap",
        greater_is_better=True,
        fp16=config.fp16,
        save_total_limit=2,
        seed=SEED,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = AdvancedLossTrainer(
        pos_weight=config.pos_weight,
        use_focal=config.use_focal_loss,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        use_margin=config.use_margin_loss,
        margin_weight=config.margin_weight,
        target_margin=config.target_margin,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Model saved to {config.output_dir}")

    return model, tokenizer


def evaluate_model(model, tokenizer, df_test):
    device = next(model.parameters()).device
    model.eval()

    df_pos = df_test[df_test['label'] == 1].reset_index(drop=True)

    if len(df_pos) > 5000:
        df_pos = df_pos.sample(5000, random_state=SEED).reset_index(drop=True)

    def get_scores(df_pairs, desc="Scoring"):
        scores = []
        for i in tqdm(range(0, len(df_pairs), 32), desc=desc):
            batch = df_pairs.iloc[i:i+32]
            inputs = tokenizer(
                batch['English'].tolist(),
                batch['Sanskrit'].tolist(),
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            scores.extend(probs)

        return np.array(scores)

    own_scores = get_scores(df_pos, "Own scores")

    cross_pairs = []
    all_sanskrit = df_pos['Sanskrit'].tolist()

    for i, row in df_pos.iterrows():
        en = row['English']
        correct_sa = row['Sanskrit']
        other_sa = random.choice([s for s in all_sanskrit if s != correct_sa])
        cross_pairs.append({'English': en, 'Sanskrit': other_sa})

    df_cross = pd.DataFrame(cross_pairs)
    cross_scores = get_scores(df_cross, "Cross scores")

    own_mean = own_scores.mean()
    cross_mean = cross_scores.mean()
    diff = own_mean - cross_mean

    print(f"\nOwn Translation:   {own_mean*100:.2f}%")
    print(f"Cross Translation: {cross_mean*100:.2f}%")
    print(f"Differentiation:   {diff*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Cross-Encoder Training (FAISS Hard Negative Mining)")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV with English and Sanskrit columns")
    parser.add_argument("--mining_model", type=str, default="sanganaka/bge-m3-sanskritFT", help="Bi-encoder for hard negative mining")
    parser.add_argument("--model_name", type=str, default="microsoft/infoxlm-large", help="Cross-encoder base model")
    args = parser.parse_args()

    config.mining_model = args.mining_model
    config.model_name = args.model_name

    try:
        df = load_data(args.input_csv)
        df_full = create_negatives_curriculum(df)
        df_train, df_dev, df_test = split_data(df_full)

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        train_ds, dev_ds, test_ds = create_datasets(df_train, df_dev, df_test, tokenizer)

        model, tokenizer = train_cross_encoder(train_ds, dev_ds)
        evaluate_model(model, tokenizer, df_test)

        print(f"Complete. Model saved to: {config.output_dir}")

    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
