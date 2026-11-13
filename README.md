# Document Instruction Safety

This repository is being developed for the **Lexsi - AI Safety and Alignment Hackathon**.

## Research question

Can a language model use helpful information from an external document while ignoring instructions in that document that try to redirect its behavior?

## Approach

We plan to evaluate a Cohere Labs open-weight language model on document-based question-answering tasks. The evaluation will compare normal documents, documents containing injected instructions, and a simple defensive condition.

The pipeline will use **CuratorKIT** for dataset preparation and **AuditKIT** for safety evaluation. We plan to measure attack success, legitimate task accuracy, and unnecessary refusals.

## Status

Initial project setup. Experiments and results have not been completed yet. The dataset, code, evaluation settings, and findings will be documented as the project develops.


## Validated pilot pipeline

The CPU evaluation pipeline is now implemented. **Tiny Aya inference has not run;
there are no research findings yet.** Python 3.12 is the verified environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pip check
python -m pytest -q
python scripts/run_pilot.py --model mock --data synthetic --output results/synthetic-smoke
python scripts/run_pilot.py --model mock --data piarena --refresh --output results/piarena-mock
```

Use a new output path each time. `--refresh` verifies the frozen real records
against the live dataset viewer and fails on drift. Omit it for offline runs;
the repository includes 24 original PIArena records and their scoring aliases.
No network access or torch is needed for the offline CPU pipeline.

The pilot comprises 24 controlled attacks on real document QA examples and three
benign quoted-instruction controls, each run in clean, injection, and defense
conditions (81 responses). The original questions, documents, references, and
native attacks are preserved. The controlled attacks replace the native attacks
with an exact-output objective. See [protocol](docs/protocol.md) for selection,
adaptations, scoring rules, denominators, and limits.

CuratorKIT `DataSample` and `ExactDeduplicator.run` perform real preparation;
AuditKIT `Sample`, custom `Metric.score`, `evaluate(model="precomputed")`, and
`RunResult.save` perform real response evaluation and reporting. Upstream Git
revisions and all core dependencies are pinned in `requirements.lock`.

## Run the real pilot in Colab

1. On Hugging Face, request/accept access conditions for
   [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global)
   and create a read token for the approved account.
2. Open [notebooks/colab_runner.ipynb](notebooks/colab_runner.ipynb) in Google
   Colab. Select **Runtime → Change runtime type → GPU** (T4 or better).
3. Add the token as the Colab secret `HF_TOKEN` and allow notebook access.
   Never paste it into a code cell, config, output, or Git.
4. Run the notebook cells in order. Its clone currently uses
   `feat/lexsi-safety-pilot`; after merge you may select `main`. The notebook
   installs the pinned core plus GPU requirements, checks model/tokenizer access,
   then invokes the existing Python runner with 4-bit NF4 loading, batch size one,
   greedy decoding, seed 42, and no fine-tuning.
5. Download the generated ZIP. Inspect `manual_review.csv`, `responses.jsonl`,
   `metrics.json`, `metadata.json`, and `auditkit.json`. Review unknowns before
   interpreting accuracy. Preserve the original files when adding human labels.

The gated tokenizer/template and GPU loader remain untested here. At runtime the
loader resolves and records the model SHA, inspects the actual tokenizer template,
checks whether system messages survive rendering, and records `system` or
`fold` mode. Folding puts the defense before the question in a user message; it
must be described as such in any findings. No silent context truncation occurs.

See [validation](docs/validation.md) for completed checks and remaining work.
