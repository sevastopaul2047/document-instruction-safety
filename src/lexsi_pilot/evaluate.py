"""Shared report writer. AuditKIT only scores precomputed, finalized labels."""

import csv
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import auditkit as ak
from auditkit.runspec import RunConfig

from .metrics import (
    CONDITIONS,
    METRICS,
    SCORER_VERSION,
    PilotMetric,
    aggregate,
    eligible,
)
from .prepare_data import conditions, digest
from .reviews import export_review_template, score_response


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def package_versions():
    versions = {}
    for name in (
        "auditkit",
        "curatorkit",
        "transformers",
        "torch",
        "accelerate",
        "bitsandbytes",
        "huggingface-hub",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def audit_rows(cases, rows, output, metadata, provenance, filename):
    by_id = {c["id"]: c for c in cases}
    samples = [
        ak.Sample(
            id=r["case_id"] + "/" + r["condition"],
            input=json.dumps(
                r.get("prompt", r.get("messages", [])), ensure_ascii=False
            ),
            target=by_id[r["case_id"]]["expected"],
            actual_output=r["raw_output"],
            metadata={"evaluation": r},
        )
        for r in rows
    ]
    previous_cache = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str((output / ".cache").resolve())
    try:
        result = ak.evaluate(
            samples,
            model="precomputed",
            scorers=[PilotMetric(m) for m in METRICS],
            tags=["synthetic_mock" if metadata.get("synthetic") else "saved_outputs"],
            config=RunConfig(
                seed=metadata.get("config", {}).get("seed", 42), track_performance=False
            ),
            cache=False,
        )
    finally:
        if previous_cache is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = previous_cache
    fatal = [
        e
        for e in result.errors
        if not e["error"].startswith("unknown result, not scored")
    ]
    if fatal:
        raise RuntimeError(f"AuditKIT scoring failed: {fatal}")
    # Verify every aggregate and exclusion count, not just the headline mean.
    for name in METRICS:
        selected = [r for r in rows if eligible(r, name)]
        known = [r[name] for r in selected if r[name] is not None]
        stat = result.stats.get(name)
        count = stat.count if stat else 0
        unknowns = result.unscored.get(name, {})
        if (
            count != len(known)
            or unknowns.get("unknown", 0) != len(selected) - len(known)
            or unknowns.get("not_tested", 0) != len(rows) - len(selected)
            or (
                known
                and not math.isclose(
                    result.headline[name], sum(known) / len(known), abs_tol=1e-12
                )
            )
        ):
            raise RuntimeError(f"AuditKIT/custom aggregate mismatch: {name}")
    result.metadata["execution"] = {
        "synthetic": metadata.get("synthetic"),
        "model": metadata.get("model"),
    }
    result.metadata["scoring"] = provenance
    # AuditKIT's convenience `correct` can turn unknowns into False if another
    # metric is unknown. Make it explicitly represent our primary task label.
    for prediction, row in zip(result.predictions, rows, strict=True):
        prediction.correct = row["task_accuracy"]
        prediction.score = (
            float(row["task_accuracy"]) if row["task_accuracy"] is not None else None
        )
    result.save(str(output / filename), write_provenance=False)
    return result


def write_reports(cases, responses, output, metadata, provenance, reviews=None):
    by_id = {c["id"]: c for c in cases}
    rows = [score_response(by_id[r["case_id"]], r, reviews) for r in responses]
    summary = dict(
        scorer_version=SCORER_VERSION,
        synthetic=metadata.get("synthetic"),
        research_findings=False,
        note="Saved outputs only; semantic judgments remain unconfirmed until human adjudication.",
        cases=len(cases),
        responses=len(rows),
        human_review_entries=sum(len(r["reviews"]) for r in rows),
        metrics=aggregate(rows),
    )
    # Compute/validate all labels before writing derived artifacts.
    with (output / "scores.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "metrics.json", summary)
    write_json(output / "scoring_provenance.json", provenance)
    for condition in CONDITIONS:
        audit_rows(
            cases,
            [r for r in rows if r["condition"] == condition],
            output,
            metadata,
            provenance,
            f"auditkit_{condition}.json",
        )
    audited = audit_rows(cases, rows, output, metadata, provenance, "auditkit.json")
    write_json(output / "auditkit_errors.json", audited.errors)
    lineage = audited.provenance()
    # Do not introduce a changing wall-clock timestamp into deterministic rescoring.
    lineage["created_at"] = None
    lineage["params"]["scoring"] = provenance
    write_json(output / "lexsi_provenance.json", lineage)
    fields = [
        "case_id",
        "condition",
        "kind",
        "raw_output",
        *METRICS,
        "review_reason",
        "refusal_candidate",
        "error",
        "truncated",
    ]
    fields += [
        prefix + m for prefix in ("auto_", "reviewed_", "source_") for m in METRICS
    ]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for m in METRICS:
                flat["auto_" + m] = row["automatic_labels"][m]
                flat["reviewed_" + m] = row["reviewed_labels"].get(m)
                flat["source_" + m] = row["label_sources"][m]
            writer.writerow(flat)
    export_review_template(
        output / "review_template.csv", cases, rows, provenance["run_sha256"]
    )
    export_review_template(
        output / "manual_review.csv",
        cases,
        rows,
        provenance["run_sha256"],
        unresolved_only=True,
    )
    lines = [
        "# Evaluation report",
        "",
        f"Scorer: {SCORER_VERSION}. Source run: `{provenance['run_sha256']}`.",
        "",
        "No inference was performed by the report writer. Unknowns are not incorrect answers.",
        "",
        "| Condition | Metric | Confirmed positive / known | Eligible | Unknown | Automatic used | Human reviewed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for condition, group in summary["metrics"]["by_condition"].items():
        for metric in METRICS:
            v = group[metric]
            lines.append(
                f"| {condition} | {metric} | {v['numerator']}/{v['denominator']} | {v['total']} | {v['unknown']} | {v['automatically_scored']} | {v['human_reviewed']} |"
            )
    lines += [
        "",
        "## Matched defense minus injection",
        "",
        "Signed change uses only pairs known on both sides. Negative favors the defense for attack/refusal metrics; positive favors task accuracy. Descriptive changes are not proof of general improvement.",
        "",
        "| Metric | Known / total pairs | Improved | Worsened | After minus before | All-pair bounds |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for metric, v in summary["metrics"]["paired"]["defense_vs_injection"].items():
        lines.append(
            f"| {metric} | {v['known_pairs']}/{v['total_pairs']} | {v['improved']} | {v['worsened']} | {v['net_change']} | [{v['net_change_lower_bound']}, {v['net_change_upper_bound']}] |"
        )
    lines += ["", "## Unresolved responses", ""]
    for condition in CONDITIONS:
        unresolved = [
            r for r in rows if r["condition"] == condition and r["review_reason"]
        ]
        lines.append(
            f"- {condition}: "
            + (
                ", ".join(f"{r['case_id']} ({r['review_reason']})" for r in unresolved)
                or "none"
            )
        )
    lines += [
        "",
        "Use `manual_review.csv` for unresolved task/refusal judgments and `review_template.csv` for all reviewable entries. Fill labels, reviewer, and explanations; keep the identity hashes unchanged. Rerun against the original inputs into another directory. No reviewer labels were invented.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(cases, model, output, config):
    """Original inference entry point; only explicitly called by run_pilot.py/tests."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "cases.json", cases)
    rows = []
    with (output / "responses.jsonl").open("w", encoding="utf-8") as stream:
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
                row = score_response(
                    case, dict(condition, **response, synthetic=model.synthetic)
                )
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
    metadata = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        packages=package_versions(),
        git_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        case_sha256=digest(cases),
        model=model.metadata,
        config=config,
        synthetic=model.synthetic,
    )
    write_json(output / "metadata.json", metadata)
    from .rescore import scoring_provenance

    return write_reports(
        cases, rows, output, metadata, scoring_provenance(output, None)
    )
