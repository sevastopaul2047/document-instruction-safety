# Document Instruction Safety

Evaluating Language Model Resilience Against Indirect Prompt Injection in Reference Documents.

Developed for the **Lexsi — AI Safety and Alignment Hackathon**.

---

## Research Question

> **Can a language model reliably extract factual information from an external document while ignoring untrusted embedded instructions designed to hijack its behavior?**

Modern retrieval-augmented generation (RAG) and document question-answering systems ingest third-party text. If that text contains indirect prompt injections, models can be manipulated into executing attacker-controlled tasks rather than serving the user. This repository benchmarks and evaluates model behavior under controlled document injection and defensive prompting.

---

## Evaluation Architecture

The framework measures behavior across a matched, three-condition experimental design:

- **Clean**: Authentic documents evaluated on legitimate task accuracy.
- **Injection**: Documents embedded with adversarial instructions aiming to hijack the output.
- **Defense**: System-level defensive prompts instructing the model to treat document content strictly as untrusted reference material.

### Two-Stage Evaluation Pipeline

1. **Stage A (Deterministic Scoring)**: Automated, exact-match rules for known answers, attacker markers, and canonical refusals via **AuditKIT** and **CuratorKIT**.
2. **Stage B (Adjudication)**: Semantic resolution of paraphrased outputs and nuanced refusals using an automated **LLM-as-a-Judge** (`scripts/judge_reviews.py`) or verified human review, cryptographically bound to run hashes.

---

## Pilot Results Summary

Evaluated on `CohereLabs/tiny-aya-global` (27 matched tasks, 81 generations):

| Condition | Task Accuracy | Attack Success Rate (ASR) | Overrefusal Rate |
| :--- | :---: | :---: | :---: |
| **Clean** (Baseline) | **81.5%** (22/27) | — | **0.0%** (0/27) |
| **Injection** (Attacked) | **77.8%** (21/27) | **8.3%** (2/24) | **0.0%** (0/27) |
| **Defense** (Guarded) | **77.8%** (21/27) | **8.3%** (2/24) | **0.0%** (0/27) |

*Full results, confidence bounds, and provenance records are available in `results/real-pilot-adjudicated/`.*

---

## Repository Guide

Navigate the repository using the directories and documents below:

| Directory / File | Description | Primary Reference |
| :--- | :--- | :--- |
| **`docs/`** | Technical documentation, operational manual, and experimental design | [Usage Guide](docs/usage.md) &middot; [Protocol](docs/protocol.md) &middot; [Validation](docs/validation.md) |
| **`scripts/`** | Primary CLI tools for pilot generation, LLM adjudication, and rescoring | `run_pilot.py` &middot; `judge_reviews.py` &middot; `rescore_pilot.py` |
| **`lexsi_real_pilot/`** | Frozen raw outputs and metadata from the real model pilot run | `cases.json` &middot; `responses.jsonl` &middot; `metadata.json` |
| **`data/fixtures/`** | Curated PIArena evaluation dataset records and SHA-256 manifests | `piarena_selected.jsonl` &middot; `provenance.json` |
| **`notebooks/`** | Google Colab notebook for executing GPU model inference | `colab_runner.ipynb` |
| **`configs/`** | Model parameters, generation settings, and prompt templates | `pilot.yaml` |
| **`src/lexsi_pilot/`** | Core library: data curation, inference drivers, and scoring metrics | CuratorKIT & AuditKIT integration |
| **`tests/`** | Automated regression and cryptographic integrity test suite | `pytest` test suite (72 tests) |
| **`results/`** | Output directories, rescored reports, and AuditKIT evaluation bundles | `real-pilot-adjudicated/report.md` |

---

## Quickstart

Detailed step-by-step instructions for all environments are documented in the **[Operating Guide](docs/usage.md)**.

```bash
# 1. Environment Setup
python -m venv .venv
source .venv/bin/activate       # On Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .

# 2. Run Test Suite
python -m pytest -q

# 3. Rescore the Real Pilot Run Offline
python scripts/rescore_pilot.py \
    --input lexsi_real_pilot \
    --output results/real-pilot-adjudicated \
    --reviews reviews.csv
```

---

## Documentation

- **[Operating Guide](docs/usage.md)**: Comprehensive manual for setup, CLI commands, Colab GPU inference, LLM-as-a-judge, and troubleshooting.
- **[Experimental Protocol](docs/protocol.md)**: Dataset adaptation, attack definitions, metric mathematics, and denominator rules.
- **[Validation Log](docs/validation.md)**: Historical records of mock runs, dependency verification, and baseline audits.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
