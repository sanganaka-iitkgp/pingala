# PINGALA: Prosody-Aware Decoding for Sanskrit Poetry Generation

Official code for the paper **"PINGALA: Prosody-Aware Decoding for Sanskrit Poetry Generation"**.

PINGALA is a model-agnostic, prosody-aware constrained decoding framework that enforces Anuṣṭubh metrical validity at inference time — without requiring architectural changes or retraining. It works by biasing token selection toward candidates that satisfy syllable-level constraints at pāda boundaries, ensuring generated Sanskrit verse adheres to the strict 32-syllable, 4-pāda structure of the Anuṣṭubh meter.

---

## Repository Structure

```
├── nllb_pingala.py                    # NLLB + PINGALA (Devanagari)
├── phi_pingala.py                     # Phi-4 + LoRA + PINGALA (Devanagari)
│
├── slp/
│   ├── phi_pingala.py                 # Phi-4 + PINGALA (SLP1)
│   └── CD/
│       └── phi_CD.py                  # Phi-4 + Constraint Decoding only (SLP1)
│
├── encoders/
│   ├── cross_train.py                 # Cross-encoder training (FAISS hard negatives)
│   ├── semantic.py                    # Semantic similarity evaluation
│   ├── syntactic.py                   # Syntactic evaluation (meter conformance)
│   ├── human_correlation.py           # Human correlation analysis
│   └── error_analysis/
│       ├── symentic.py                # Semantic error analysis
│       └── syntastic.py               # Syntactic error analysis
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

> Replace `faiss-gpu` with `faiss-cpu` in `requirements.txt` if you don't have a CUDA GPU.

---

## Usage

### Generation

**NLLB + PINGALA (Devanagari output):**
```bash
python nllb_pingala.py --input_csv input.csv --output_csv output.csv
```

**Phi-4 + PINGALA (Devanagari output):**
```bash
python phi_pingala.py --input_csv input.csv --output_csv output.csv
```

**Phi-4 + PINGALA (SLP1 output):**
```bash
python slp/phi_pingala.py --input_csv input.csv --output_csv output.csv
```

**Constraint Decoding only — no pāda rewards (SLP1):**
```bash
python slp/CD/phi_CD.py --input_csv input.csv --output_csv output.csv
```

All generation scripts expect a CSV with an `English` column and produce an output CSV with `English` and `Sanskrit` columns.

### Evaluation

**Syntactic (meter conformance):**
```bash
python encoders/syntactic.py --input_csv output.csv
```

**Semantic (cross-encoder):**
```bash
python encoders/semantic.py --model_dir <cross_encoder_dir> --input_csv output.csv
```

### Training the Cross-Encoder

```bash
python encoders/cross_train.py --input_csv train.csv
```

---

## Models

| Component | Model |
|-----------|-------|
| NLLB translation | `sanganaka/nllb-200-distilled-1.3B-FT_anushtubh` |
| Phi-4 base | `unsloth/phi-4` |
| Phi-4 LoRA adapter | `barca-boy/lora_model_phi-4_anushtup_final` |
| Phi-4 SLP1 | `manojbalaji1/phi4-slp1` |
| Cross-encoder base | `microsoft/infoxlm-large` |
| Bi-encoder (eval) | `sanganaka/bge-m3-sanskritFT` |

---

## Citation

```bibtex
@inproceedings{pingala2025,
  title   = {PINGALA: Prosody-Aware Decoding for Sanskrit Poetry Generation},
  year    = {2025}
}
```
