"""Generate once, then let real AuditKIT evaluate and persist precomputed answers."""

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import auditkit as ak
from auditkit.runspec import RunConfig

from .metrics import PilotMetric, aggregate, labels
from .prepare_data import conditions, digest


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def run(cases, model, output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, samples = [], []
    write_json(output / "cases.json", cases)
    # Flush every raw response so an interrupted GPU run retains completed generations.
    with (output / "responses.jsonl").open("w") as stream:
        for case in cases:
            for condition in conditions(case):
                try:
                    response = model.generate(case, condition)
                except Exception as exc:
                    response = dict(
                        raw_output="",
                        error=f"{type(exc).__name__}: {exc}",
                        truncated=False,
                        prompt=condition["messages"],
                    )
                row = dict(
                    condition,
                    **response,
                    synthetic=model.synthetic,
                    **labels(
                        case,
                        response["raw_output"],
                        response.get("error"),
                        response.get("truncated", False),
                    ),
                )
                if condition["condition"] == "clean":
                    row["attack_success"] = None
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                samples.append(
                    ak.Sample(
                        id=case["id"] + "/" + condition["condition"],
                        input=json.dumps(condition["messages"], ensure_ascii=False),
                        target=case["expected"],
                        actual_output=response["raw_output"],
                        metadata=dict(
                            case=case,
                            synthetic=model.synthetic,
                            condition=condition["condition"],
                            error=response.get("error"),
                            truncated=response.get("truncated", False),
                        ),
                    )
                )
    previous_cache = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str((output / ".cache").resolve())
    try:
        audited = ak.evaluate(
            samples,
            model="precomputed",
            scorers=[
                PilotMetric(n)
                for n in ("task_accuracy", "attack_success", "overrefusal")
            ],
            tags=["synthetic_mock" if model.synthetic else "real_unreviewed"],
            config=RunConfig(seed=config["seed"], track_performance=False),
            cache=False,
        )
    finally:
        if previous_cache is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = previous_cache
    if audited.errors:
        write_json(output / "auditkit_errors.json", audited.errors)
        fatal = [
            e
            for e in audited.errors
            if not e["error"].startswith("unknown result, not scored")
        ]
        if fatal:
            raise RuntimeError("AuditKIT scoring errors; inspect auditkit_errors.json")
    audited.metadata["execution"] = dict(
        synthetic=model.synthetic, model=model.metadata
    )
    audited.save(str(output / "auditkit.json"))
    summary = dict(
        synthetic=model.synthetic,
        research_findings=not model.synthetic,
        note="Synthetic mock validation only; no Tiny Aya findings"
        if model.synthetic
        else "Unreviewed real outputs; inspect manual_review.csv before interpreting",
        cases=len(cases),
        responses=len(rows),
        metrics=aggregate(rows),
    )
    write_json(output / "metrics.json", summary)
    versions = {}
    for package in (
        "auditkit",
        "curatorkit",
        "transformers",
        "torch",
        "accelerate",
        "bitsandbytes",
        "huggingface-hub",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    write_json(
        output / "metadata.json",
        dict(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            python=platform.python_version(),
            packages=versions,
            git_commit=commit,
            case_sha256=digest(cases),
            model=model.metadata,
            config=config,
            synthetic=model.synthetic,
        ),
    )
    fields = [
        "case_id",
        "condition",
        "kind",
        "synthetic",
        "raw_output",
        "task_accuracy",
        "attack_success",
        "overrefusal",
        "review_reason",
        "error",
        "truncated",
    ]
    for filename, selected in [
        ("results.csv", rows),
        ("manual_review.csv", [r for r in rows if r["review_reason"]]),
    ]:
        with (output / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
    return summary
