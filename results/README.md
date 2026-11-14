Run artifacts are ignored by Git. Choose a new directory for every run.
CPU smoke summaries are documented in `docs/validation.md`; all mock outputs
are synthetic pipeline checks, never findings about Tiny Aya.

Offline rescoring writes only to a new directory. `responses.jsonl` and
`metadata.json` retain original bytes; v2 labels live in `scores.jsonl` and the
new metrics/report files. `original/` retains the complete source run. Do not
commit private pilot outputs or mistake mock fixtures for real model findings.
