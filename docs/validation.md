# Validation record

Completed 2026-11-13 in Codex Cloud, Python **3.12.14**, CPU only.
No model weights were downloaded and no real Tiny Aya inference was attempted.

## Installed and checked

- CuratorKIT **0.1.21**, Git `ed45ae3489bfb6ee12c969e5d07e0309b758e829`.
- AuditKIT **1.1.11**, Git `556eb4852d515391294203232ca3c90bdd48251d`.
- A second, fresh virtual environment installed `requirements.lock` directly from
  these immutable Git revisions, plus the local editable project. `uv pip check`
  reported all 21 installed packages compatible.
- The GPU requirements resolved together (65 packages). Transformers **5.21.1**
  was installed without dependencies into an inspection directory; its Cohere2
  auto-model registrations were checked. This is source/resolver validation,
  not a GPU-stack installation or runtime compatibility claim.

## Automated verification

`python -m pytest -q`: **29 passed**. Coverage includes exact answers, known wrong
answers, successful attacks, strict refusals, ambiguous paraphrases/marker
quotations, truncation/empty outputs, inference exceptions, schema and snapshot
integrity, CuratorKIT duplicate removal without attack changes, all matched
conditions, AuditKIT unknown exclusion/directions/synthetic labels, benign ASR
exclusion, matched-baseline denominators, and machine-readable exports.

`ruff check src scripts tests` and `git diff --check` passed. All notebook code
cells parse as Python, contain no saved outputs, and invoke the Python pipeline
rather than duplicating model/scoring logic. The notebook has not run in Colab.

## End-to-end smoke runs — SYNTHETIC MODEL OUTPUTS

Both commands completed through real CuratorKIT and AuditKIT operations:

```bash
python scripts/run_pilot.py --model mock --data synthetic --output <new-directory>
python scripts/run_pilot.py --model mock --data piarena --refresh --output <new-directory>
```

The first produced 12 responses for four synthetic fixtures. The second verified
all 24 frozen records against the live PIArena viewer and produced 81 responses
for 24 adapted real QA tasks plus three benign controls. Both saved raw outputs,
per-case results, metrics, manual-review CSV, run metadata, and AuditKIT reports
and provenance. Existing output directories were protected from overwrite.

For the 27-case smoke run only, the **scripted mock** produced the following
arithmetic checks. These numbers are expected fixture behavior, **not findings
about Tiny Aya or evidence that the defense works on a language model**:

| Condition | Exact task answers / known labels | Exact attack outputs / attack trials | Unnecessary refusals / known labels |
|---|---:|---:|---:|
| Clean | 27/27 | not tested | 0/27 |
| Injection | 3/27 | 24/24 | 0/27 |
| Defense | 27/27 | 0/24 | 0/27 |

Edge-case tests separately exercise wrong/unknown/refusal/error results so the
smoke model's scripted success does not conceal those paths. Full generated
artifacts remain outside Git; rerun the commands to reproduce synthetic checks.

## Access observations and remaining validation

PIArena viewer split discovery and first-row schema inspection succeeded over
HTTPS. `default` / `dolly_closed_qa` is the actual configuration/split. The frozen
selection is untruncated and preserves every original field. The Hub dataset
metadata endpoint returned 401, so the snapshot uses a content hash and viewer
row indices rather than an invented dataset revision.

Tiny Aya's tokenizer configuration and model card requests returned **HTTP 401,
restricted model access**. The user must accept/request the model's conditions
with the token's Hugging Face account and supply a read token in Colab Secrets.
The actual tokenizer, system-role behavior, gated model architecture, model
loading, NF4 memory use, output quality, and GPU execution remain untested.

Run `notebooks/colab_runner.ipynb` on a GPU as described in README. Review the
resulting unknowns and raw responses before interpreting accuracy or reporting
security results. The controlled attack adaptation and small convenience sample
limit the claims even after real inference. Fine-tuning and the broader hackathon
submission are outside this task.
