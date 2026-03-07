# Dynamic Demand Prod Runbook (Ground Truth)

This document is the execution reference for Dan and production engineers.
It explains what each script does, when to use it, what inputs it expects, and what outputs it writes.

## 1) What this package is

The `prod/` folder is a production execution layer around two core scripts in the parent folder:

- `../run_dynamic_demand_baseline.py` (preprocessing, feature engineering, QA exports)
- `../run_dynamic_demand_models_gross.py` (model training/evaluation/explainability exports)

The prod layer provides two execution styles:
- Wrapper CLI style (parameterized command-line runs)
- Template style (edit config block in Python file, then run)

## 2) Folder dependencies

The following files must exist together under:
`week_of_2026-03-02/demand_modeling/dynamic_demand_baseline/`

- Core scripts:
  - `run_dynamic_demand_baseline.py`
  - `run_dynamic_demand_models_gross.py`
- Dependency file:
  - `requirements.txt`
- DB config template:
  - `db_config_template.py` (or `prod/db_config_template.py`)
- Prod scripts:
  - `prod/run_dynamic_demand_preprocessing_prod.py`
  - `prod/run_dynamic_demand_modeling_prod.py`
  - `prod/template_dynamic_demand_preprocessing_prod.py`
  - `prod/template_dynamic_demand_modeling_prod.py`
  - `prod/db_config_template.py`

## 3) Prerequisites

- Python environment with dependencies from `requirements.txt`
- Network access to Postgres
- Valid DB credentials

Install deps:
```powershell
pip install -r week_of_2026-03-02/demand_modeling/dynamic_demand_baseline/requirements.txt
```

## 4) DB config

Create a credentials file from template.

Template file:
- `prod/db_config_template.py`

Create:
- `prod/db_config.py` (or any path passed to `--db-config`)

Expected object:
```python
DB_CONFIG = {
    "host": "...",
    "port": 5432,
    "database": "...",
    "user": "...",
    "password": "...",
    "sslmode": "require",
}
```

Security:
- Do not commit real credentials.
- Use secret store in production if available.

## 5) Script catalog (what each script does)

### A) Core script: `../run_dynamic_demand_baseline.py`

Purpose:
- Connect to Postgres.
- Build hotel-day panel using DBA0..DBA7 logic.
- Construct demand labels and pricing features.
- Apply preprocessing transforms (including clipping/log features already in core logic).
- Write panel/model-input/QA outputs.

Inputs:
- Postgres tables (`hotels`, `inventories`, `reservations`, `rate_amounts`, `ota_rates`)
- Config constants in script (dates, clipping settings)
- Optional runtime hotel filter via `--hotels`

Hotel scope behavior:
- Default: all active hotels (`TARGET_HOTELS = []`)
- Override at runtime:
  - single hotel: `--hotels Anvil_Hotel`
  - multiple hotels: `--hotels "Anvil_Hotel,Ozarker_Lodge"`
  - all: `--hotels ALL`

Outputs:
- `output/01_long_ls_panel/`
- `output/02_model_input/`
- `output/06_reporting/`

---

### B) Core script: `../run_dynamic_demand_models_gross.py`

Purpose:
- Load model input produced by preprocessing.
- Train GLM/XGB/baseline models.
- Run grouped random date CV.
- Compute portfolio and hotelwise metrics.
- Generate SHAP and coefficient outputs.

Inputs:
- `output/02_model_input/all_hotels_model_input_dynamic_baseline.csv`
- Modeling constants in script (date window, CV seeds/folds, target, params)
- Optional runtime hotel filter via `--hotels`

Hotel scope behavior:
- Default: all hotels present in model input file
- Override at runtime:
  - single hotel: `--hotels Anvil_Hotel`
  - multiple hotels: `--hotels "Anvil_Hotel,Ozarker_Lodge"`
  - all: `--hotels ALL`

Outputs:
- `output/07_model_results/`

---

### C) Wrapper script: `run_dynamic_demand_preprocessing_prod.py`

Purpose:
- Parameterized wrapper for core preprocessing script.
- Lets engineering pass dates/hotels/db-config at runtime without editing core code.

What it does internally:
- Loads `../run_dynamic_demand_baseline.py`
- Overrides key globals (`PRICE_CREATED_AT_START`, `DECISION_DATE_START`, `MODELING_STAY_END`, `REALIZED_CUTOFF`, `TARGET_HOTELS`)
- Injects DB config from `--db-config` if provided
- Calls source `main()`

When to use:
- Scheduled/automated runs
- Repeatable operations across environments

---

### D) Wrapper script: `run_dynamic_demand_modeling_prod.py`

Purpose:
- Parameterized wrapper for core modeling script.
- Lets engineering set dates/targets/seeds/folds/hotel filter via CLI.

What it does internally:
- Loads `../run_dynamic_demand_models_gross.py`
- Overrides key globals (`MODELING_STAY_START`, `MODELING_STAY_END`, `REALIZED_CUTOFF`, `SEEDS`, split settings, targets)
- Optionally wraps `load_data()` to filter hotel IDs
- Calls source `main()`

When to use:
- Scheduled/automated runs
- Explicit experiment configuration without editing source

---

### E) Template script: `template_dynamic_demand_preprocessing_prod.py`

Purpose:
- Editable config-block version of preprocessing wrapper.
- Engineers edit constants at top, then run.

When to use:
- Teams preferring file-based config over CLI args
- Manual operations and ad hoc reruns

---

### F) Template script: `template_dynamic_demand_modeling_prod.py`

Purpose:
- Editable config-block version of modeling wrapper.
- Engineers edit constants at top, then run.

When to use:
- Same as above for modeling stage

---

### G) Template file: `db_config_template.py`

Purpose:
- Credential schema template for DB connection.

## 6) Recommended execution mode

For production operations, use **Wrapper CLI mode**.
Use templates only when teams prefer editing config blocks in source files.

Do not run wrappers and templates simultaneously for the same job; pick one mode.

## 7) End-to-end run sequence

Run order is mandatory:
1. Preprocessing
2. Modeling

### 7.1 Wrapper mode examples

All active hotels preprocessing:
```powershell
python week_of_2026-03-02/demand_modeling/dynamic_demand_baseline/prod/run_dynamic_demand_preprocessing_prod.py `
  --db-config C:\path\to\db_config.py `
  --hotels ALL `
  --stay-start 2025-07-01 `
  --stay-end 2026-02-28 `
  --decision-start 2025-07-01 `
  --price-created-start 2025-07-01 `
  --realized-cutoff 2026-03-06
```

Single hotel preprocessing:
```powershell
python week_of_2026-03-02/demand_modeling/dynamic_demand_baseline/prod/run_dynamic_demand_preprocessing_prod.py `
  --db-config C:\path\to\db_config.py `
  --hotels Anvil_Hotel `
  --stay-start 2025-07-01 `
  --stay-end 2026-02-28
```

Single hotel modeling:
```powershell
python week_of_2026-03-02/demand_modeling/dynamic_demand_baseline/prod/run_dynamic_demand_modeling_prod.py `
  --hotels Anvil_Hotel `
  --stay-start 2025-07-01 `
  --stay-end 2026-02-28 `
  --realized-cutoff 2026-03-06 `
  --seeds "42,52,62" `
  --outer-splits 5 `
  --inner-splits 3 `
  --split-scheme-name grouped_random_date_cv `
  --target-primary target_gross_rooms_pickup `
  --target-secondary target_gross_bookings
```

All hotels modeling:
```powershell
python week_of_2026-03-02/demand_modeling/dynamic_demand_baseline/prod/run_dynamic_demand_modeling_prod.py `
  --hotels ALL `
  --stay-start 2025-07-01 `
  --stay-end 2026-02-28 `
  --realized-cutoff 2026-03-06 `
  --seeds "42,52,62" `
  --outer-splits 5 `
  --inner-splits 3
```

### 7.2 Template mode examples

Preprocessing template:
```powershell
python week_of_2026-03-02/demand_modeling/dynamic_demand_baseline/prod/template_dynamic_demand_preprocessing_prod.py
```

Modeling template:
```powershell
python week_of_2026-03-02/demand_modeling/dynamic_demand_baseline/prod/template_dynamic_demand_modeling_prod.py
```

## 8) Output map

Preprocessing outputs:
- `output/01_long_ls_panel/` (LS panel datasets)
- `output/02_model_input/` (model-input datasets)
- `output/06_reporting/` (QA and reporting support tables)

Modeling outputs:
- `output/07_model_results/` (portfolio metrics, fold metrics, SHAP, GLM coefficients, diagnostics)

## 9) Common run failures and fixes

1. DB config not found:
- Ensure `db_config.py` exists and contains `DB_CONFIG`.
- Or pass explicit `--db-config` path in wrapper mode.

2. No rows after hotel/date filtering:
- Confirm hotel IDs match `hotels.global_id`.
- Confirm date window overlaps available data.

3. Modeling fails with missing model input:
- Preprocessing must run first successfully.
- Verify `output/02_model_input/all_hotels_model_input_dynamic_baseline.csv` exists.

4. Empty results for single hotel:
- Hotel may not have enough rows for minimum fold thresholds in modeling script.

## 10) Notes for maintainers

- Core scripts now support single-hotel and multi-hotel runs via `--hotels`.
- Wrappers/templates exist to avoid editing core code for each run.
- Keep date windows aligned between preprocessing and modeling.
