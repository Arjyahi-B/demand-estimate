from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

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


def _load_db_config_py(config_path: Path) -> Dict[str, Any]:
    cfg_module = _load_module(config_path, "prod_db_config")
    cfg = getattr(cfg_module, "DB_CONFIG", None)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"{config_path} must define DB_CONFIG as a dict.")
    required = {"host", "port", "database", "user", "password"}
    missing = sorted(required - set(cfg.keys()))
    if missing:
        raise RuntimeError(f"Missing DB_CONFIG keys in {config_path}: {missing}")
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Production wrapper for dynamic demand preprocessing. "
            "Uses existing run_dynamic_demand_baseline.py without modifying it."
        )
    )
    parser.add_argument(
        "--source-script",
        type=Path,
        default=Path(__file__).resolve().parent / "run_dynamic_demand_baseline.py",
        help="Path to baseline preprocessing script.",
    )
    parser.add_argument(
        "--db-config",
        type=Path,
        default=None,
        help="Path to Python file defining DB_CONFIG dict. If omitted, source script default loading is used.",
    )
    parser.add_argument(
        "--hotels",
        type=str,
        default="ALL",
        help="Comma-separated hotel IDs. Use ALL for all active hotels.",
    )
    parser.add_argument("--stay-start", type=str, default="2025-07-01")
    parser.add_argument("--stay-end", type=str, default="2026-02-28")
    parser.add_argument("--decision-start", type=str, default=None)
    parser.add_argument("--price-created-start", type=str, default=None)
    parser.add_argument("--realized-cutoff", type=str, default=str(date.today()))
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    source_script = args.source_script.resolve()
    if not source_script.exists():
        raise RuntimeError(f"Source script not found: {source_script}")

    module = _load_module(source_script, "dynamic_preprocess_source")

    stay_start = pd.Timestamp(args.stay_start)
    stay_end = pd.Timestamp(args.stay_end)
    decision_start = pd.Timestamp(args.decision_start) if args.decision_start else stay_start
    price_created_start = (
        pd.Timestamp(args.price_created_start) if args.price_created_start else stay_start
    )
    realized_cutoff = pd.Timestamp(args.realized_cutoff)
    hotels = _parse_hotels(args.hotels)

    module.PRICE_CREATED_AT_START = price_created_start
    module.DECISION_DATE_START = decision_start
    module.MODELING_STAY_END = stay_end
    module.REALIZED_CUTOFF = realized_cutoff
    module.TARGET_HOTELS = hotels

    db_config_path = args.db_config.resolve() if args.db_config else None
    if db_config_path is not None:
        if not db_config_path.exists():
            raise RuntimeError(f"DB config file not found: {db_config_path}")
        db_cfg = _load_db_config_py(db_config_path)
        module.load_db_config = lambda _base: db_cfg
    else:
        db_cfg = "source-script-default"

    run_cfg = {
        "source_script": str(source_script),
        "db_config": str(db_config_path) if db_config_path else "source-script-default",
        "target_hotels": hotels if hotels else "ALL_ACTIVE_HOTELS",
        "stay_start": str(stay_start.date()),
        "stay_end": str(stay_end.date()),
        "decision_start": str(decision_start.date()),
        "price_created_start": str(price_created_start.date()),
        "realized_cutoff": str(realized_cutoff.date()),
    }
    print("[INFO] Production preprocessing run config:")
    print(json.dumps(run_cfg, indent=2))
    if db_cfg != "source-script-default":
        print("[INFO] DB config loaded from external template path.")

    module.main()


if __name__ == "__main__":
    main()
