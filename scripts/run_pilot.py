#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml

from lexsi_pilot.evaluate import run
from lexsi_pilot.inference import MockModel, TransformersModel
from lexsi_pilot.prepare_data import ROOT, build_cases, curate, load_selection


def main():
    parser = argparse.ArgumentParser(
        description="Generate a new pilot (mock outputs are synthetic). For saved outputs use scripts/rescore_pilot.py."
    )
    parser.add_argument("--model", choices=["mock", "transformers"], default="mock")
    parser.add_argument("--data", choices=["piarena", "synthetic"], default="piarena")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Verify frozen rows against live PIArena viewer; fail on drift",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing runs are never overwritten",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.model == "transformers" and args.data == "synthetic":
        parser.error("Use PIArena data for the real pilot")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new run path")
    cases = (
        build_cases(load_selection(refresh=args.refresh))
        if args.data == "piarena"
        else json.loads((ROOT / "data/fixtures/synthetic.json").read_text())
    )
    cases = curate(cases)
    model = MockModel() if args.model == "mock" else TransformersModel(config)
    print(json.dumps(run(cases, model, args.output, config), indent=2))


if __name__ == "__main__":
    main()
