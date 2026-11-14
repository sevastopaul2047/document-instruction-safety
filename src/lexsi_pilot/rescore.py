"""Offline rescoring: reads saved artifacts, never imports an inference backend."""

import argparse
import hashlib
import json
import logging
import platform
import shutil
from pathlib import Path

from .evaluate import package_versions, write_reports
from .metrics import CONDITIONS, SCORER_VERSION, matched_rows
from .reviews import import_reviews, score_response


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(source):
    return {
        p.relative_to(source).as_posix(): file_hash(p)
        for p in sorted(source.rglob("*"))
        if p.is_file()
    }


def run_hash(source):
    return hashlib.sha256(
        json.dumps(
            {
                name: file_hash(source / name)
                for name in ("cases.json", "responses.jsonl")
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def scoring_provenance(source, reviews_path):
    return dict(
        scorer_version=SCORER_VERSION,
        run_sha256=run_hash(source),
        input_hashes=source_hashes(source),
        reviews_sha256=file_hash(reviews_path) if reviews_path else None,
        scoring_code_sha256={
            p.name: file_hash(p)
            for p in sorted(Path(__file__).parent.glob("*.py"))
            if p.name != "inference.py"
        },
        python=platform.python_version(),
        packages={
            k: v
            for k, v in package_versions().items()
            if k in ("auditkit", "curatorkit")
        },
        inference_performed=False,
    )


def load_run(source):
    cases = json.loads((source / "cases.json").read_text(encoding="utf-8"))
    responses = [
        json.loads(line)
        for line in (source / "responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not isinstance(cases, list) or not cases or not responses:
        raise ValueError("Empty or invalid completed run")
    by_id = {}
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or not case["id"]
            or case["id"] in by_id
        ):
            raise ValueError("Missing or duplicate case ID")
        for key in ("question", "document", "expected", "marker"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Missing case field: {key}")
        if (
            case.get("kind") not in ("attack", "benign")
            or not isinstance(case.get("aliases"), list)
            or not all(isinstance(a, str) and a.strip() for a in case["aliases"])
        ):
            raise ValueError("Invalid case kind/aliases")
        if not isinstance(case.get("wrong_aliases", []), list) or not all(
            isinstance(a, str) for a in case.get("wrong_aliases", [])
        ):
            raise ValueError("Invalid wrong aliases")
        by_id[case["id"]] = case
    for response in responses:
        if not isinstance(response, dict) or response.get("case_id") not in by_id:
            raise ValueError("Unknown response case ID")
        if (
            response.get("condition") not in CONDITIONS
            or response.get("kind") != by_id[response["case_id"]]["kind"]
        ):
            raise ValueError("Invalid response condition/kind")
        if (
            not isinstance(response.get("raw_output"), str)
            or type(response.get("truncated")) is not bool
            or "error" not in response
        ):
            raise ValueError("Missing or invalid raw output/error/truncation status")
        if response["error"] is not None and not isinstance(response["error"], str):
            raise ValueError("Invalid generation error")
    groups = matched_rows(responses)
    if set(groups) != set(by_id):
        raise ValueError("Missing responses for a case")
    # Canonical scoring order; original files retain their exact bytes and order.
    responses = [groups[c["id"]][condition] for c in cases for condition in CONDITIONS]
    metadata = (
        json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if (source / "metadata.json").exists()
        else {}
    )
    flags = {r.get("synthetic") for r in responses}
    if len(flags) > 1 or any(f is not None and type(f) is not bool for f in flags):
        raise ValueError("Inconsistent synthetic response flags")
    flag = next(iter(flags))
    if (
        "synthetic" in metadata
        and flag is not None
        and metadata["synthetic"] is not flag
    ):
        raise ValueError("Metadata/response synthetic flags disagree")
    # Only reporting metadata is enriched; the saved original metadata is unchanged.
    metadata = dict(metadata, synthetic=metadata.get("synthetic", flag))
    return cases, responses, metadata


def rescore(source, output, reviews_path=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (
        output.exists()
        or source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Output must be a new directory separate from the source run")
    if not source.is_dir() or any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Source must be a regular run directory without symlinks")
    cases, responses, metadata = load_run(source)
    provenance = scoring_provenance(source, reviews_path)
    reviews = import_reviews(reviews_path, cases, responses, provenance["run_sha256"])
    # Validate contradictions before creating output files.
    by_id = {c["id"]: c for c in cases}
    for response in responses:
        score_response(by_id[response["case_id"]], response, reviews)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, output / "original")
    for name in ("cases.json", "responses.jsonl", "metadata.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, output / name)
    if reviews_path:
        shutil.copyfile(reviews_path, output / "reviews_imported.csv")
    result = write_reports(cases, responses, output, metadata, provenance, reviews)
    if source_hashes(source) != provenance["input_hashes"]:
        raise RuntimeError("Source run changed during rescoring")
    if source_hashes(output / "original") != provenance["input_hashes"]:
        raise RuntimeError("Original artifact copies do not match the source manifest")
    if (
        reviews_path
        and file_hash(output / "reviews_imported.csv") != provenance["reviews_sha256"]
    ):
        raise RuntimeError("Review file changed during rescoring")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Rescore saved pilot outputs offline; never run inference"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Completed run directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New, separate directory"
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        help="Filled review template CSV (optional; partial allowed)",
    )
    args = parser.parse_args()
    # Expected unknown-score notices are preserved in JSON, not repeated hundreds of times.
    logging.getLogger("auditkit.runner").setLevel(logging.ERROR)
    try:
        result = rescore(args.input, args.output, args.reviews)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
