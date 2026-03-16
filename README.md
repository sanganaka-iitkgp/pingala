# NeuroSymbolic Sanskrit Poetry Generation

Code for the paper: **"NeuroSymbolic Anushtup: Integrating Constraint Decoding with Neural Translation for Sanskrit Metrical Poetry"**.

This repository contains the full pipeline for generating Sanskrit verse in the **Anushtubh** (śloka) meter from English input — using constrained beam search decoding, fine-tuned language models, and automatic evaluation tools.

---

## Overview

Sanskrit Anushtubh meter requires exactly **32 syllables** arranged as 4 padas of 8 syllables each, with strict constraints on syllable weight (laghu/guru) at positions 5, 6, and 7 of every pada. Satisfying these constraints during neural generation is the core challenge this codebase addresses.

Our approach combines:
- **Constrained decoding** via a custom `LogitsProcessor` that enforces meter at every generation step
- **Fine-tuned language models**: NLLB (seq2seq) and Phi-4 (causal LM with LoRA)
- **Two-stage fallback**: Constrained generation first; simple translation fallback when constraints cannot be satisfied
- **Cross-encoder evaluation** — trained on English-Sanskrit parallel data with hard negative mining

---

## Repository Structure

```
hub/
│
├── nllb_pingala.py              # NLLB-based constrained generation (Devanagari output)
├── phi_pingala.py               # Phi-4 + LoRA constrained generation (Devanagari output)
│
├── slp/
│   └── phi_pingala.py           # Phi-4 constrained generation (SLP1 transliteration output)
│   └── CD/
│       └── phi_CD.py            # Phi-4 constraint-decoding only — no two-stage fallback (SLP1)
│
├── encoders/
│   ├── cross_train.py           # Cross-encoder training with FAISS hard negative mining
│   ├── semantic.py              # Semantic similarity inference using trained cross-encoder
│   ├── syntactic.py             # Syntactic evaluation: Anushtubh meter conformance + semantic sim
│   ├── human_correlation.py     # Correlation of model scores against human ratings
│   │
│   └── error_analysis/
│       ├── symentic.py          # Semantic error analysis with distribution plots
│       └── syntastic.py         # Syntactic error analysis: syllable deviation, meter patterns
│
└── requirements.txt
```

---

## Models

All models are hosted on Hugging Face and loaded automatically at runtime.

| Script | Model |
|--------|-------|
| `nllb_pingala.py` | [`sanganaka/nllb-200-distilled-1.3B-FT_anushtubh`](https://huggingface.co/sanganaka/nllb-200-distilled-1.3B-FT_anushtubh) |
| `phi_pingala.py` | [`unsloth/phi-4`](https://huggingface.co/unsloth/phi-4) + [`barca-boy/lora_model_phi-4_anushtup_final`](https://huggingface.co/barca-boy/lora_model_phi-4_anushtup_final) |
| `slp/phi_pingala.py`, `slp/CD/phi_CD.py` | [`manojbalaji1/phi4-slp1`](https://huggingface.co/manojbalaji1/phi4-slp1) |
| `encoders/cross_train.py` | [`microsoft/infoxlm-large`](https://huggingface.co/microsoft/infoxlm-large) (fine-tuned) |
| `encoders/syntactic.py`, `encoders/error_analysis/syntastic.py` | [`sanganaka/bge-m3-sanskritFT`](https://huggingface.co/sanganaka/bge-m3-sanskritFT) |
| `encoders/semantic.py` | Custom trained cross-encoder (output of `cross_train.py`) |

---

## Input/Output Format

All generation scripts expect an input CSV with at minimum an **`English`** column:

```csv
English
"Let the wise man speak the truth."
"The river flows towards the ocean."
```

Output CSV will have an **`English`** and **`Sanskrit`** column with the generated verse.

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** If you do not have a GPU with CUDA support, replace `faiss-gpu` with `faiss-cpu` in `requirements.txt`.

---

## Usage

### 1. NLLB Generation (Devanagari)

```bash
python nllb_pingala.py \
    --input_csv input.csv \
    --output_csv output.csv \
    --model_name sanganaka/nllb-200-distilled-1.3B-FT_anushtubh \
    --num_beams 25 \
    --batch_size 16
```

### 2. Phi-4 Generation (Devanagari, Two-Stage)

```bash
python phi_pingala.py \
    --input_csv input.csv \
    --output_csv output.csv \
    --base_model unsloth/phi-4 \
    --lora_adapter barca-boy/lora_model_phi-4_anushtup_final
```

### 3. Phi-4 Generation in SLP1 (Two-Stage)

```bash
python slp/phi_pingala.py \
    --input_csv input.csv \
    --output_csv output.csv \
    --model_name manojbalaji1/phi4-slp1
```

### 4. Phi-4 SLP1 — Constraint Decoding Only

```bash
python slp/CD/phi_CD.py \
    --input_csv input.csv \
    --output_csv output_cd.csv \
    --model_name manojbalaji1/phi4-slp1
```

---

## Evaluation

### Syntactic Evaluation (Anushtubh Meter Conformance)

Measures what percentage of generated verses satisfy the full Anushtubh meter pattern.

```bash
python encoders/syntactic.py \
    --input_csv output.csv \
    --english_col English \
    --sanskrit_col Sanskrit \
    --output_csv eval_results.csv \
    --semantic_model sanganaka/bge-m3-sanskritFT
```

**Output columns added to CSV:** `anushtup_type` (Full/Partial/None), `syllable_count`, `meter_label`, `semantic_sim_%`

### Semantic Evaluation (Cross-Encoder)

```bash
python encoders/semantic.py \
    --model_dir path/to/trained_cross_encoder \
    --input_csv output.csv \
    --english_col English \
    --sanskrit_col Sanskrit
```

### Error Analysis

**Semantic error analysis** (score distribution, worst/best samples, plots):

```bash
python encoders/error_analysis/symentic.py \
    --model_dir path/to/trained_cross_encoder \
    --input_csv output.csv \
    --output_prefix semantic_analysis
```

**Syntactic error analysis** (syllable deviation, error patterns, plots):

```bash
python encoders/error_analysis/syntastic.py \
    --input_csv output.csv \
    --output_prefix syntactic_analysis \
    --semantic_model sanganaka/bge-m3-sanskritFT
```

### Human Correlation

```bash
python encoders/human_correlation.py \
    --input_csv annotated.csv \
    --output_plot correlation.png
```

Input CSV must have columns: `score` (1–5 human rating), `semantic_sim` (cross-encoder probability), `bi_encoder_score`.

---

## Training the Cross-Encoder

```bash
python encoders/cross_train.py \
    --input_csv train_fixed.csv \
    --mining_model sanganaka/bge-m3-sanskritFT \
    --model_name microsoft/infoxlm-large
```

Input CSV must have `english` and `sanskrit` columns. The script will:
1. Create easy negatives (random mismatched pairs) and hard negatives (FAISS similarity search)
2. Train with Focal Loss + Margin Loss
3. Save the trained model to `./cross_encoder_optimized_high_positive/`

---

## Key Concepts

### Constrained Decoding (`AnushtupConstrainedLogitsProcessor`)

At each generation step, the logits processor:
1. Scans the current partial sequence using `skrutable` to get the syllable weight pattern
2. For each candidate next token (top-K), extends and re-scans the sequence
3. Sets logits of tokens that violate meter constraints to `-inf`
4. Forces EOS when 32 syllables are complete; prevents EOS before 32 syllables

### Two-Stage Generation

1. **Stage 1** — Constrained beam search: enforces full Anushtubh meter during decoding
2. **Stage 2** — If no valid verse is found, falls back to simple translation (no constraint) to ensure the output field is never empty

### Anushtubh Rules (Positions 5-6-7 of each pada)

| Pada | Position 5 | Position 6 | Position 7 |
|------|-----------|-----------|-----------|
| Odd (1, 3) | laghu (l) | guru (g) | guru (g) |
| Even (2, 4) | laghu (l) | guru (g) | laghu (l) |

---

## System Requirements

- Python ≥ 3.10
- CUDA GPU with ≥ 16 GB VRAM recommended (for Phi-4 8-bit inference)
- `faiss-gpu` for hard negative mining (or `faiss-cpu` if no GPU)

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{anushtup2025,
  title   = {NeuroSymbolic Anushtup: Integrating Constraint Decoding with Neural Translation for Sanskrit Metrical Poetry},
  year    = {2025}
}
```

---

## License

MIT License
