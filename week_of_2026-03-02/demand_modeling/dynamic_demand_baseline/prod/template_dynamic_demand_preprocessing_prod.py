from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pandas as pd


# =========================
# EDIT THIS CONFIG BLOCK
# =========================
SOURCE_SCRIPT = Path(__file__).resolve().parents[1] / "run_dynamic_demand_baseline.py"
DB_CONFIG_FILE = Path(__file__).resolve().parents[1] / "db_config.py"

# Use [] for all active hotels in DB, or list explicit hotel IDs.
HOTEL_IDS: List[str] = []

STAY_START = "2025-07-01"
STAY_END = "2026-02-28"
DECISION_START = "2025-07-01"
PRICE_CREATED_START = "2025-07-01"
REALIZED_CUTOFF = "2026-03-06"


def _load_module(script_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script at: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_db_config_py(config_path: Path) -> Dict[str, Any]:
    cfg_module = _load_module(config_path, "prod_db_config_template")
    cfg = getattr(cfg_module, "DB_CONFIG", None)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"{config_path} must define DB_CONFIG as a dict.")
    required = {"host", "port", "database", "user", "password"}
    missing = sorted(required - set(cfg.keys()))
    if missing:
        raise RuntimeError(f"Missing DB_CONFIG keys in {config_path}: {missing}")
    return cfg


def main() -> None:
    source_script = SOURCE_SCRIPT.resolve()
    if not source_script.exists():
        raise RuntimeError(f"Source script not found: {source_script}")

    module = _load_module(source_script, "dynamic_preprocess_source_template")
    db_cfg = _load_db_config_py(DB_CONFIG_FILE.resolve())

    module.PRICE_CREATED_AT_START = pd.Timestamp(PRICE_CREATED_START)
    module.DECISION_DATE_START = pd.Timestamp(DECISION_START)
    module.MODELING_STAY_END = pd.Timestamp(STAY_END)
    module.REALIZED_CUTOFF = pd.Timestamp(REALIZED_CUTOFF)
    module.TARGET_HOTELS = sorted(set(HOTEL_IDS))
    module.load_db_config = lambda _base: db_cfg

    run_cfg = {
        "source_script": str(source_script),
        "db_config_file": str(DB_CONFIG_FILE.resolve()),
        "hotel_ids": module.TARGET_HOTELS if module.TARGET_HOTELS else "ALL_ACTIVE_HOTELS",
        "stay_start": STAY_START,
        "stay_end": STAY_END,
        "decision_start": DECISION_START,
        "price_created_start": PRICE_CREATED_START,
        "realized_cutoff": REALIZED_CUTOFF,
    }
    print("[INFO] Template preprocessing run config:")
    print(json.dumps(run_cfg, indent=2))

    module.main()


if __name__ == "__main__":
    main()
