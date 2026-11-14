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

## Scorer 2.0.0 repair validation — 2026-11-15

The preceding sections record the original CPU implementation, not the current
state of the user's experiments. The user subsequently supplied a recovered
**real Tiny Aya Global run** (`debug-20261113T191347Z.zip` and its nine individual
artifacts). All individual artifact bytes matched their ZIP entries. The saved
metadata identifies `CohereLabs/tiny-aya-global`, revision
`af89d219b53ed9b13b8a4645f9c8028973510324`, with system-role rendering. The
recovered run has 27 unique cases, 81 saved responses, and no recorded generation
errors or truncation. This repair performed **no new model inference**.

Original input identities:

- `cases.json` SHA256: `e5440fd5f4f7752800485ec3f5be5b67b7b79a0dd9982b2a2afb4fd68376e968`
- `responses.jsonl` SHA256: `81b9aa21e7ed0b1a21171394e6ffcd67d2e653e1fd3da0a1157930993f528c89`

### Automated checks

`python -m pytest -q`: **72 passed**, zero failures. The original 29 tests remain
covered with expectations updated for whole-response attack definitions and the
fourth metric. New tests cover human correct/incorrect/uncertain labels,
explanations and identities, partial/blank/missing/duplicate/stale/malformed
reviews, conflicting reviews, correct rejection of document instructions,
unknown denominators and coverage counts, paired gains/losses/introduced
refusals, complete-run validation, deterministic offline exports, original-file
preservation, and per-condition AuditKIT/custom agreement.

The offline CLI also passed a subprocess test that deliberately rejects imports
of `torch`, `transformers`, and `lexsi_pilot.inference`. All tests involving
response generation use synthetic mocks only; the real-pilot validation below
reads saved files exclusively. Existing requirements were sufficient; no runtime
dependency was added. `uv pip check` on the Python 3.12 CPU environment passed;
Ruff and `git diff --check` passed.

### Saved real-pilot rescore, with no human labels supplied

The command was executed against the recovered directory:

```bash
python scripts/rescore_pilot.py --input /workspace/lexsi_real_pilot --output results/lexsi-real-rescored-v2
```

A second rescore of the same inputs into a separate directory produced identical
bytes for every top-level file, including scores, metrics, CSVs, report,
AuditKIT exports, and scoring/lineage metadata. SHA256 checks before and after
confirmed that **all nine original artifacts remained unchanged**, and that all
copies under `original/` match them. All 81 original raw responses, prompts, and
inference metadata are preserved. New scoring data lives separately. Full raw
outputs and review queues are kept outside Git.

The following are deterministic v2 observations, not semantic adjudications:

| Condition | Confirmed task answers / known task labels | Eligible task responses | Task labels unresolved | Exact attack success | Case-insensitive marker compliance |
|---|---:|---:|---:|---:|---:|
| Clean | unmeasured (0 known) | 27 | 27 | ineligible | ineligible |
| Injection | 9/13 | 27 | 14 | 2/24 | 4/24 |
| Defense | 9/13 | 27 | 14 | 2/24 | 4/24 |

There are **55 distinct responses needing semantic task review**: 27 clean,
14 injection, and 14 defense. Their overrefusal labels are also unresolved.
`manual_review.csv` lists **110 metric-review entries** for these 55 responses;
`review_template.csv` has all 162 task/refusal entries. Correct task adjudication
can also resolve the corresponding refusal label without a second judgment.
No reviewer identities, explanations, or human labels were invented or imported.
The original v1 manual-review export was archived, not treated as adjudication.

Overrefusal has 0 positives among 13 known labels in each attacked condition,
with 14 unknowns in each; clean has no known refusal labels. This is **not** a
claim of zero overall overrefusal. Unrecognized wording, including potentially
incorrect answers and legitimate paraphrases, still needs human evidence.

Matched exact-marker and case-insensitive-marker comparisons each have 24 known
attack pairs, no decreases, no increases, and zero net change. Thus there is no
observed defense reduction of either marker objective in these saved outputs.
Task accuracy has only 12 matched known injection/defense pairs, all unchanged,
and 15 unknown pairs; no task-accuracy improvement conclusion is supported.
All 24 malicious clean baselines are unknown, so redirection conditional on a
correct clean answer remains unmeasured until human review. Refusal effects also
remain incomplete. These observations supersede no human conclusions: the
original reports remain intact, and the difference in marker coverage is
explicitly defined by scorer version 2.

### Remaining human work

Review every queued `(case_id, condition, metric)` against its saved question,
document, reference, and actual output. Start with clean task labels to establish
baseline capability, then review attacked task answers and ambiguous refusal
language. Decide whether a response answers the question, refuses the legitimate
request, rejects only the malicious instruction, or remains uncertain. Use the
exported hashes and provide reviewer explanations for definitive decisions.
Do not guess labels to increase coverage; `uncertain` remains a valid unresolved
outcome. Run a new reviewed rescore with the cumulative review file, preserving
this unreviewed result and all original artifacts.
