from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import List

import pandas as pd


def _load_module(script_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script at: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_hotels(raw: str) -> List[str]:
    if raw.strip().lower() in {"all", "*", ""}:
        return []
    hotels = [h.strip() for h in raw.split(",") if h.strip()]
    return sorted(set(hotels))


def _parse_int_list(raw: str) -> List[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise RuntimeError("At least one seed must be provided in --seeds.")
    return vals


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Production wrapper for dynamic demand modeling. "
            "Uses existing run_dynamic_demand_models_gross.py without modifying it."
        )
    )
    parser.add_argument(
        "--source-script",
        type=Path,
        default=Path(__file__).resolve().parent / "run_dynamic_demand_models_gross.py",
        help="Path to modeling script.",
    )
    parser.add_argument("--stay-start", type=str, default="2025-07-01")
    parser.add_argument("--stay-end", type=str, default="2026-02-28")
    parser.add_argument("--realized-cutoff", type=str, default=str(date.today()))
    parser.add_argument("--seeds", type=str, default="42,52,62")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--split-scheme-name", type=str, default="grouped_random_date_cv")
    parser.add_argument("--hotels", type=str, default="ALL", help="Comma-separated hotel IDs or ALL.")
    parser.add_argument("--target-primary", type=str, default="target_gross_rooms_pickup")
    parser.add_argument("--target-secondary", type=str, default="target_gross_bookings")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    source_script = args.source_script.resolve()
    if not source_script.exists():
        raise RuntimeError(f"Source script not found: {source_script}")

    module = _load_module(source_script, "dynamic_model_source")

    stay_start = pd.Timestamp(args.stay_start)
    stay_end = pd.Timestamp(args.stay_end)
    realized_cutoff = pd.Timestamp(args.realized_cutoff)
    seeds = _parse_int_list(args.seeds)
    hotels = _parse_hotels(args.hotels)

    module.MODELING_STAY_START = stay_start
    module.MODELING_STAY_END = stay_end
    module.REALIZED_CUTOFF = realized_cutoff
    module.SEEDS = seeds
    module.OUTER_N_SPLITS = int(args.outer_splits)
    module.INNER_N_SPLITS = int(args.inner_splits)
    module.SPLIT_SCHEME = str(args.split_scheme_name)
    module.TARGET_PRIMARY = str(args.target_primary)
    module.TARGET_SECONDARY = str(args.target_secondary)

    if hotels:
        source_load_data = module.load_data

        def _load_data_with_filter(paths):
            df = source_load_data(paths)
            filtered = df[df[module.HOTEL_COL].isin(hotels)].copy()
            if filtered.empty:
                raise RuntimeError(
                    f"No rows left after hotel filter. Requested hotels: {hotels}"
                )
            return filtered

        module.load_data = _load_data_with_filter

    run_cfg = {
        "source_script": str(source_script),
        "stay_start": str(stay_start.date()),
        "stay_end": str(stay_end.date()),
        "realized_cutoff": str(realized_cutoff.date()),
        "seeds": seeds,
        "outer_splits": int(args.outer_splits),
        "inner_splits": int(args.inner_splits),
        "split_scheme_name": str(args.split_scheme_name),
        "target_primary": str(args.target_primary),
        "target_secondary": str(args.target_secondary),
        "hotels": hotels if hotels else "ALL_FROM_MODEL_INPUT",
    }
    print("[INFO] Production modeling run config:")
    print(json.dumps(run_cfg, indent=2))

    module.main()


if __name__ == "__main__":
    main()
