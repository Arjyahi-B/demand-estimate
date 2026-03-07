from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import List

import pandas as pd


# =========================
# EDIT THIS CONFIG BLOCK
# =========================
SOURCE_SCRIPT = Path(__file__).resolve().parent / "run_dynamic_demand_models_gross.py"

# Use [] to model all hotels found in model input. Otherwise list hotel IDs.
HOTEL_IDS: List[str] = []

STAY_START = "2025-07-01"
STAY_END = "2026-02-28"
REALIZED_CUTOFF = "2026-03-06"
SEEDS = [42, 52, 62]
OUTER_SPLITS = 5
INNER_SPLITS = 3
SPLIT_SCHEME_NAME = "grouped_random_date_cv"
TARGET_PRIMARY = "target_gross_rooms_pickup"
TARGET_SECONDARY = "target_gross_bookings"


def _load_module(script_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script at: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    source_script = SOURCE_SCRIPT.resolve()
    if not source_script.exists():
        raise RuntimeError(f"Source script not found: {source_script}")

    module = _load_module(source_script, "dynamic_model_source_template")

    module.MODELING_STAY_START = pd.Timestamp(STAY_START)
    module.MODELING_STAY_END = pd.Timestamp(STAY_END)
    module.REALIZED_CUTOFF = pd.Timestamp(REALIZED_CUTOFF)
    module.SEEDS = [int(s) for s in SEEDS]
    module.OUTER_N_SPLITS = int(OUTER_SPLITS)
    module.INNER_N_SPLITS = int(INNER_SPLITS)
    module.SPLIT_SCHEME = str(SPLIT_SCHEME_NAME)
    module.TARGET_PRIMARY = str(TARGET_PRIMARY)
    module.TARGET_SECONDARY = str(TARGET_SECONDARY)

    hotel_ids = sorted(set(HOTEL_IDS))
    if hotel_ids:
        source_load_data = module.load_data

        def _load_data_with_filter(paths):
            df = source_load_data(paths)
            filtered = df[df[module.HOTEL_COL].isin(hotel_ids)].copy()
            if filtered.empty:
                raise RuntimeError(
                    f"No rows left after hotel filter. Requested hotels: {hotel_ids}"
                )
            return filtered

        module.load_data = _load_data_with_filter

    run_cfg = {
        "source_script": str(source_script),
        "stay_start": STAY_START,
        "stay_end": STAY_END,
        "realized_cutoff": REALIZED_CUTOFF,
        "seeds": module.SEEDS,
        "outer_splits": module.OUTER_N_SPLITS,
        "inner_splits": module.INNER_N_SPLITS,
        "split_scheme_name": module.SPLIT_SCHEME,
        "target_primary": module.TARGET_PRIMARY,
        "target_secondary": module.TARGET_SECONDARY,
        "hotel_ids": hotel_ids if hotel_ids else "ALL_FROM_MODEL_INPUT",
    }
    print("[INFO] Template modeling run config:")
    print(json.dumps(run_cfg, indent=2))

    module.main()


if __name__ == "__main__":
    main()
