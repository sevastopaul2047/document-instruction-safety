# Operating Guide: Document Instruction Safety

This document provides a comprehensive operational guide for the Document Instruction Safety research repository. It describes system prerequisites, environment setup, core workflows, data structures, and the evaluation lifecycle.

---

## Table of Contents

1. [System Overview and Architecture](#1-system-overview-and-architecture)
2. [Prerequisites and Environment Setup](#2-prerequisites-and-environment-setup)
3. [Repository Directory Structure](#3-repository-directory-structure)
4. [Quickstart Workflow](#4-quickstart-workflow)
5. [Core Operations and Workflows](#5-core-operations-and-workflows)
   - [Workflow A: Local Smoke and Mock Testing](#workflow-a-local-smoke-and-mock-testing)
   - [Workflow B: Running Real Model Inference on Google Colab](#workflow-b-running-real-model-inference-on-google-colab)
   - [Workflow C: Automated LLM-as-a-Judge Adjudication](#workflow-c-automated-llm-as-a-judge-adjudication)
   - [Workflow D: Manual Human Review Adjudication](#workflow-d-manual-human-review-adjudication)
   - [Workflow E: Offline Rescoring and Report Generation](#workflow-e-offline-rescoring-and-report-generation)
6. [Understanding Evaluation Metrics and Artifacts](#6-understanding-evaluation-metrics-and-artifacts)
7. [Testing and Verification](#7-testing-and-verification)
8. [Troubleshooting and Common Questions](#8-troubleshooting-and-common-questions)

---

## 1. System Overview and Architecture

The Document Instruction Safety framework evaluates language model vulnerability and resilience to indirect prompt injection embedded within external reference documents.

The benchmark executes a matched three-condition study:

- **Clean Condition (Baseline)**: The model answers a factual question using an authentic document.
- **Injection Condition (Attacked)**: An adversarial command is injected into the document attempting to redirect model execution toward an attacker-specified objective.
- **Defense Condition (Guarded)**: A defensive system instruction or prefix is applied, instructing the model to rely solely on document content for factual retrieval while disregarding embedded commands.

### Two-Stage Evaluation Design

To eliminate reporting ambiguity while maintaining statistical rigor, evaluation is separated into two explicit stages:

1. **Stage A (Deterministic Rule-Based Scoring)**:
   - High-precision automated rules evaluate exact strings, known answer aliases, attacker marker strings, and canonical refusal patterns.
   - Non-standard paraphrases, conversational sentence wrapping, or ambiguous outputs are not classified arbitrarily. They are labeled as `unknown` and exported to `manual_review.csv`.

2. **Stage B (Adjudication via Human or LLM-as-a-Judge)**:
   - Ambiguous and unconfirmed cases are adjudicated by an authoritative reviewer or an LLM judge.
   - Reviews require an explicit categorical label (`correct`, `incorrect`, or `uncertain` for task accuracy; `refusal`, `not_refusal`, or `uncertain` for refusal classification) along with a factual justification.
   - Cryptographic SHA-256 bindings ensure that reviews match the exact run and response bytes without tampering.

---

## 2. Prerequisites and Environment Setup

### 2.1 Supported Environments

- **Python**: 3.10, 3.11, 3.12, or 3.13 (3.12 recommended for upstream parity).
- **Operating Systems**: Linux (Ubuntu 22.04+), macOS (13+), or Windows (10/11 with PowerShell).
- **Compute Profiles**:
  - **CPU Only**: Sufficient for mock testing, data preparation, offline rescoring, and LLM-as-a-Judge adjudication.
  - **GPU (16GB+ VRAM or Google Colab T4)**: Required only if executing local model inference on 8-billion parameter models such as `CohereLabs/tiny-aya-global`.

### 2.2 Local Installation

Clone the repository and initialize the Python virtual environment:

#### On Linux / macOS:

```bash
git clone https://github.com/sevastopaul2047/document-instruction-safety.git
cd document-instruction-safety

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pip check
```

#### On Windows (PowerShell):

```powershell
git clone https://github.com/sevastopaul2047/document-instruction-safety.git
cd document-instruction-safety

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pip check
```

> [!NOTE]
> On Windows, if Git automatically converts LF line endings to CRLF, configure `git config --local core.autocrlf false` to ensure test hash fixtures match expected byte signatures.

---

## 3. Repository Directory Structure

```text
document-instruction-safety/
|-- configs/
|   `-- pilot.yaml                  # Model inference configuration and prompt templates
|-- data/
|   `-- fixtures/
|       |-- piarena_selected.jsonl  # 24 curated PIArena document-QA tasks + scoring aliases
|       |-- provenance.json         # SHA-256 digest manifest for fixture integrity
|       `-- synthetic.json          # Small synthetic test fixture for rapid smoke tests
|-- docs/
|   |-- protocol.md                 # Formal experimental design, scoring rules, and math
|   |-- usage.md                    # This operational manual
|   `-- validation.md               # Historical test record and synthetic baseline log
|-- lexsi_real_pilot/               # Saved real pilot run (Tiny Aya Global, 81 responses)
|   |-- cases.json                  # Canonical evaluation tasks
|   |-- responses.jsonl             # Raw model outputs across clean/injection/defense
|   |-- metadata.json               # Model version, hardware, and generation metadata
|   `-- manual_review.csv           # Unresolved Stage A responses requiring adjudication
|-- notebooks/
|   `-- colab_runner.ipynb          # Automated Google Colab GPU runner for model inference
|-- results/                        # Target directory for generated pilot runs and reports
|-- scripts/
|   |-- judge_reviews.py            # Automated LLM-as-a-Judge adjudication script
|   |-- rescore_pilot.py            # Offline rescoring script (strictly no inference dependencies)
|   `-- run_pilot.py                # Primary pilot execution runner
|-- src/
|   `-- lexsi_pilot/
|       |-- evaluate.py             # AuditKIT evaluation adapter and metrics orchestration
|       |-- inference.py            # Hugging Face Transformers and Mock model drivers
|       |-- metrics.py              # Stage A deterministic scoring rules and conditions
|       |-- prepare_data.py         # CuratorKIT data loading, filtering, and curation
|       |-- rescore.py              # Offline report compiler and cryptographic verifier
|       `-- reviews.py              # Stage B review ingestion, schema checks, and merging
|-- tests/
|   `-- test_pilot.py               # Test suite covering deterministic rules, hashing, schemas
|-- .env                            # Local environment variables and API keys (git-ignored)
|-- requirements.lock               # Pinned reproducible dependencies
`-- pyproject.toml                  # Package specification
```

---

## 4. Quickstart Workflow

The standard operational pipeline consists of four major steps:

```text
+-----------------------+     +------------------------+     +------------------------+     +-----------------------+
|  1. Verify Pipeline   | --> | 2. Acquire Model Run   | --> | 3. Adjudicate Unknowns | --> |  4. Rescore Offline   |
| (pytest & Mock Smoke) |     |  (Colab or Existing)   |     |    (judge_reviews.py)  |     |  (rescore_pilot.py)   |
+-----------------------+     +------------------------+     +------------------------+     +-----------------------+
```

### Step 1: Run Test Suite and Mock Smoke Run

```bash
python -m pytest -q
python scripts/run_pilot.py --model mock --data synthetic --output results/smoke-synthetic
```

### Step 2: Use an Existing or New Model Run

The repository includes a validated real model run in `lexsi_real_pilot/` containing 81 generations from `CohereLabs/tiny-aya-global`. Alternatively, generate a new run using Google Colab as detailed in Workflow B.

### Step 3: Adjudicate Unresolved Responses via LLM-as-a-Judge

```bash
python scripts/judge_reviews.py --input lexsi_real_pilot --output reviews.csv
```

### Step 4: Rescore Offline into a Final Report

```bash
python scripts/rescore_pilot.py --input lexsi_real_pilot --output results/real-pilot-adjudicated --reviews reviews.csv
```

Inspect the final report in `results/real-pilot-adjudicated/report.md`.

---

## 5. Core Operations and Workflows

### Workflow A: Local Smoke and Mock Testing

Mock testing executes the entire data curation and scoring architecture using deterministic mock responses. It requires no GPU and no external API keys.

#### Synthetic Data Smoke Run

Processes 4 synthetic test cases across 3 conditions (12 total responses):

```bash
python scripts/run_pilot.py --model mock --data synthetic --output results/smoke-synthetic
```

#### Curated PIArena Fixture Smoke Run

Processes the full 24 PIArena QA cases plus 3 benign distractor controls (81 total responses):

```bash
python scripts/run_pilot.py --model mock --data piarena --output results/smoke-piarena
```

#### Live Dataset Drift Verification

To verify that the local frozen PIArena fixture matches the live upstream dataset on Hugging Face without drift, append `--refresh`:

```bash
python scripts/run_pilot.py --model mock --data piarena --refresh --output results/smoke-piarena-refresh
```

> [!IMPORTANT]
> The runner enforces non-destructive operation: if the target `--output` directory already exists, the script halts to prevent accidental data overwrites. Always specify a fresh output path.

---

### Workflow B: Running Real Model Inference on Google Colab

To generate real language model responses without local GPU hardware, use the provided Colab runner:

1. **Hugging Face Model Access**:
   - Navigate to [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global).
   - Accept the user agreement conditions.
   - Generate a Hugging Face Read Token under your account settings.

2. **Open Notebook in Google Colab**:
   - Open `notebooks/colab_runner.ipynb` in Google Colab.
   - Under **Runtime → Change runtime type**, select a GPU runtime (T4 or higher).

3. **Configure Access Secret**:
   - In Colab, click the key icon (Secrets) in the left panel.
   - Add a new secret named `HF_TOKEN` and paste your token value.
   - Enable notebook access for `HF_TOKEN`.

4. **Execute Pipeline Cells**:
   - Run the notebook cells sequentially.
   - The notebook checks model accessibility, clones the pinned repository code, loads the model in 4-bit NF4 precision, and executes all 81 generation runs.

5. **Download Run Bundle**:
   - The final notebook cell packages the output directory into a `.zip` archive.
   - Download the archive and extract it locally (for example, into `results/my-model-run/`).

---

### Workflow C: Automated LLM-as-a-Judge Adjudication

Outputs generated by language models frequently contain conversational framing (for example, *"Based on the document provided, the three high schools are..."*). While factually accurate, these responses differ from raw exact-match strings and are routed to `manual_review.csv` as `unknown`.

The `judge_reviews.py` script automates Stage B adjudication using a secondary LLM judge.

#### 1. Configure API Credentials

Create a `.env` file in the repository root directory (or export environment variables directly):

```bash
# Google Gemini (Recommended for free-tier usage)
GEMINI_API_KEY=AIzaSy...

# Optional alternative providers
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk-...
```

#### 2. Execute the Adjudication Script

Run the judge targeting a completed pilot run:

```bash
python scripts/judge_reviews.py --input lexsi_real_pilot --output reviews.csv
```

#### 3. Execution Features and Flags

- **Free-Tier Protection**: When using the default `gemini` provider, the script automatically enforces a 4.0-second request delay (15 requests/minute) and applies exponential backoff for transient HTTP 503 capacity errors.
- **Model Auto-Rotation**: When a Gemini free-tier daily model quota is exhausted (HTTP 429), the script detects the quota limit and automatically rotates to the next available free-tier model (`gemini-flash-lite-latest`, `gemini-3.1-flash-lite`, etc.) without losing progress.
- **Incremental Checkpointing**: Progress is written to `--output` after every evaluation. If execution is interrupted, re-running the command automatically resumes from the last completed item.
- **Multi-Provider Support**:

```bash
# Run using OpenAI GPT-4o-mini
python scripts/judge_reviews.py --provider openai --model gpt-4o-mini --input lexsi_real_pilot --output reviews.csv

# Run using Groq Llama 3.3 70B
python scripts/judge_reviews.py --provider groq --model llama-3.3-70b-versatile --input lexsi_real_pilot --output reviews.csv

# Run using local Ollama instance
python scripts/judge_reviews.py --provider ollama --model llama3.2 --input lexsi_real_pilot --output reviews.csv
```

---

### Workflow D: Manual Human Review Adjudication

If external API services are not available, adjudication can be conducted entirely by human reviewers:

1. Open `manual_review.csv` located within the completed run directory (or copy `review_template.csv` to `reviews.csv`).
2. Populate the following columns for each entry:
   - `label`:
     - For `task_accuracy`: `correct`, `incorrect`, or `uncertain`.
     - For `overrefusal`: `not_refusal`, `refusal`, or `uncertain`.
   - `reviewer`: Identifier of the human evaluator (e.g., `evaluator_alice`).
   - `explanation`: Brief factual justification explaining why the label was assigned.
3. Save the file as `reviews.csv`.

> [!CAUTION]
> Do not modify the `run_sha256`, `case_id`, `condition`, or `response_sha256` columns. These hashes bind each judgment to the exact input bytes. Modifying them will cause `rescore_pilot.py` to reject the review file during integrity verification.

---

### Workflow E: Offline Rescoring and Report Generation

Once `reviews.csv` has been generated (via LLM judge or manual evaluation), execute offline rescoring:

```bash
python scripts/rescore_pilot.py \
    --input lexsi_real_pilot \
    --output results/real-pilot-adjudicated \
    --reviews reviews.csv
```

#### What Rescoring Performs:

1. Verifies that `--output` is a new, nonexistent directory.
2. Loads all original records and validates input SHA-256 hashes against `cases.json` and `responses.jsonl`.
3. Ingests `reviews.csv`, verifying cryptographic response signatures.
4. Checks logical consistency (for example, ensuring that an answer marked `correct` is never contradictory with a refusal label).
5. Copies original inputs into `results/real-pilot-adjudicated/original/` for provenance preservation.
6. Generates updated metrics, structured scores, and markdown summary reports.

---

## 6. Understanding Evaluation Metrics and Artifacts

### 6.1 Primary Evaluation Metrics

All metrics report numerators, denominators, and bounds. Unknown cases are strictly tracked and never counted as incorrect.

| Metric | Target Condition | Evaluation Rule |
| :--- | :--- | :--- |
| **`task_accuracy`** | Clean, Injection, Defense | Proportion of eligible tasks where the model accurately retrieved the factual answer specified by the reference document. |
| **`attack_success`** (ASR) | Injection, Defense | Proportion of attack cases where the model executed the injected command instead of answering the factual query. |
| **`marker_compliance`** | Injection, Defense | Proportion of cases where the model output leaked or reproduced the attacker's required tracking marker. |
| **`overrefusal`** | Clean, Injection, Defense | Proportion of valid requests where the model unnecessarily refused to answer. |
| **`benign_overrefusal`** | Clean, Injection, Defense | Specific refusal rate on the 3 benign distractor controls (documents quoting instructions in safe educational contexts). |

### 6.2 Output Directory Manifest

Upon completion of rescoring, the target output directory contains the following artifacts:

- **`report.md`**: Human-readable markdown table summarizing confirmed positive counts, eligible counts, unknown counts, and defense-versus-injection delta comparisons.
- **`metrics.json`**: Machine-readable JSON hierarchy with full nested statistics, confidence bounds, and matched pair tracking.
- **`scores.jsonl`**: Detailed record for each response containing Stage A automatic labels, Stage B review labels, and finalized classifications.
- **`auditkit.json`**: Formal AuditKIT evaluation bundle documenting test cases, model outputs, and evaluator configurations.
- **`scoring_provenance.json`**: Cryptographic provenance manifest recording code file hashes, Python version, package versions, and input SHA-256 digests.
- **`manual_review.csv`**: Contains any remaining unadjudicated entries. If all items were adjudicated, this file contains only the header.
- **`original/`**: An exact bitwise copy of the input run directory.

---

## 7. Testing and Verification

The repository contains an automated regression suite enforcing strict schema integrity, cryptographic hash binding, and edge-case scoring behavior.

### Running Pytest

Execute all unit and integration tests:

```bash
python -m pytest -q
```

All 72 tests should pass:

```text
........................................................................ [100%]
72 passed in 5.83s
```

### Key Test Validations

- **`test_snapshot_hash`**: Verifies that the frozen PIArena dataset snapshot matches its historical SHA-256 digest (`bb5970c4...`).
- **`test_real_schema_snapshot_and_matches`**: Confirms that all 24 curated tasks load cleanly through CuratorKIT without data corruption.
- **`test_review_cryptographic_verification`**: Confirms that forged, stale, or tampered review entries are rejected by the rescorer.
- **`test_refresh_rejects_drift`**: Verifies that schema changes or upstream modifications are detected during dataset refresh.

---

## 8. Troubleshooting and Common Questions

### 1. `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8d`
- **Cause**: On Windows systems, Python may default to the system locale (`cp1252`) when opening UTF-8 text files.
- **Resolution**: All file operations in the repository explicitly enforce `encoding="utf-8"`. If running custom scripts, ensure files are opened with `open(path, encoding="utf-8")`.

### 2. `AssertionError` in `test_snapshot_hash`
- **Cause**: Git on Windows converted LF line endings in `data/fixtures/piarena_selected.jsonl` to CRLF line endings upon checkout, changing the file's SHA-256 hash.
- **Resolution**: Run the following commands in PowerShell:
  ```powershell
  git config --local core.autocrlf false
  git checkout HEAD -- data/fixtures/piarena_selected.jsonl
  ```

### 3. `The '<' operator is reserved for future use` (PowerShell)
- **Cause**: In PowerShell, `<` is a reserved character. This error occurs if copy-pasting command examples containing angle brackets literally (e.g., `<path_to_run>`).
- **Resolution**: Replace placeholder values with the actual folder name:
  ```powershell
  # Correct usage
  python scripts/rescore_pilot.py --input lexsi_real_pilot --output results/my-run --reviews reviews.csv
  ```

### 4. `RuntimeError: Repository path already exists` (Google Colab)
- **Cause**: Re-running Cell 1 in Google Colab when the repository directory has already been cloned.
- **Resolution**: The updated notebook automatically checks and cleans previous checkouts. If running manually, execute `shutil.rmtree(ROOT)` or restart the Colab runtime before re-running.

### 5. `HTTP Error 429: Resource Exhausted` during LLM-as-a-Judge
- **Cause**: Google AI Studio enforces daily request limits per model on the free tier (typically 20 requests per model per day).
- **Resolution**: The `judge_reviews.py` script automatically catches HTTP 429 exceptions and seamlessly rotates to the next available free-tier model (`gemini-flash-lite-latest`, `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-3-flash-preview`). No manual intervention is needed.

### 6. `HTTP Error 503: Service Unavailable`
- **Cause**: Google AI Studio preview models intermittently experience high-demand capacity spikes.
- **Resolution**: The script incorporates automatic exponential backoff (retrying up to 6 times) and will automatically fall back to an active alternative model if the overload persists.
