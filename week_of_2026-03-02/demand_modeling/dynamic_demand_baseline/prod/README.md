<<<<<<< HEAD
# Dynamic Demand Prod Folder (Standalone Runbook)

This `prod/` folder is now **fully runnable by itself**.
You can share this folder only.

## 1) What is included

### Core executables (standalone copies)
- `run_dynamic_demand_baseline.py`
- `run_dynamic_demand_models_gross.py`

These are the main scripts that do preprocessing and modeling.

### Wrapper scripts (CLI parameterized)
- `run_dynamic_demand_preprocessing_prod.py`
- `run_dynamic_demand_modeling_prod.py`

Use these if you prefer passing config via command line flags.

### Template scripts (config-block style)
- `template_dynamic_demand_preprocessing_prod.py`
- `template_dynamic_demand_modeling_prod.py`

Use these if you prefer editing constants in Python.

### Config and dependencies
- `db_config_template.py`
- `requirements.txt`

## 2) What each script does

### A) `run_dynamic_demand_baseline.py` (core preprocessing)
Purpose:
- Connects to Postgres.
- Pulls hotel/inventory/reservation/rate/compset data.
- Builds DBA0..DBA7 hotel-day panel.
- Creates demand targets and engineered features.
- Applies clipping/log transforms in current pipeline logic.
- Writes preprocessing outputs.

Hotel scope:
- Default: all active hotels (`TARGET_HOTELS = []` in file).
- Runtime override supported:
  - `--hotels ALL`
  - `--hotels Anvil_Hotel`
  - `--hotels "Anvil_Hotel,Ozarker_Lodge"`

Outputs:
- `output/01_long_ls_panel/`
- `output/02_model_input/`
- `output/06_reporting/`

---

### B) `run_dynamic_demand_models_gross.py` (core modeling)
Purpose:
- Loads model input from preprocessing.
- Runs baseline, GLM, and XGBoost modeling.
- Executes grouped random date CV.
- Produces metrics, diagnostics, SHAP, coefficients.

Hotel scope:
- Default: all hotels present in model input.
- Runtime override supported:
  - `--hotels ALL`
  - `--hotels Anvil_Hotel`
  - `--hotels "Anvil_Hotel,Ozarker_Lodge"`

Outputs:
- `output/07_model_results/`

---

### C) `run_dynamic_demand_preprocessing_prod.py` (wrapper)
Purpose:
- Runs preprocessing core script with CLI-supplied parameters (dates/hotels/db-config/source path).
- Good for automation/scheduling.

Default source script:
- `./run_dynamic_demand_baseline.py` (inside this same folder)

---

### D) `run_dynamic_demand_modeling_prod.py` (wrapper)
Purpose:
- Runs modeling core script with CLI-supplied parameters (dates/seeds/folds/targets/hotels/source path).
- Good for automation/scheduling.

Default source script:
- `./run_dynamic_demand_models_gross.py` (inside this same folder)

---

### E) `template_dynamic_demand_preprocessing_prod.py` (template)
Purpose:
- Same as preprocessing wrapper, but config is edited at top of file.
- Good for manual operations.

Default source script:
- `./run_dynamic_demand_baseline.py`

Default DB config file:
- `./db_config.py`

---

### F) `template_dynamic_demand_modeling_prod.py` (template)
Purpose:
- Same as modeling wrapper, but config is edited at top of file.
- Good for manual operations.

Default source script:
- `./run_dynamic_demand_models_gross.py`

## 3) Prerequisites

- Python installed
- Network access to Postgres
- Valid DB credentials

Install dependencies from this folder:
```powershell
pip install -r requirements.txt
```

## 4) DB config setup

1. Copy:
- `db_config_template.py` -> `db_config.py`

2. Fill credentials:
=======
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
>>>>>>> 9c4e6b400dbce8e3b0496c59908e3dc2ca739ae6
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
<<<<<<< HEAD
- Never commit real credentials.

## 5) Recommended mode

Use **wrapper CLI mode** in production.

Use template mode only if your team prefers editing Python config blocks.

Run either wrappers or templates for a given job, not both.

## 6) End-to-end run sequence

Run order:
1. Preprocessing
2. Modeling

## 7) Commands (wrapper mode)

### 7.1 All hotels preprocessing
=======
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
>>>>>>> 9c4e6b400dbce8e3b0496c59908e3dc2ca739ae6
```powershell
python run_dynamic_demand_preprocessing_prod.py `
  --db-config .\db_config.py `
  --hotels ALL `
  --stay-start 2025-07-01 `
  --stay-end 2026-02-28 `
  --decision-start 2025-07-01 `
  --price-created-start 2025-07-01 `
  --realized-cutoff 2026-03-06
```

<<<<<<< HEAD
### 7.2 Single hotel preprocessing
```powershell
python run_dynamic_demand_preprocessing_prod.py `
  --db-config .\db_config.py `
  --hotels Anvil_Hotel `
  --stay-start 2025-07-01 `
  --stay-end 2026-02-28
```

### 7.3 All hotels modeling
```powershell
python run_dynamic_demand_modeling_prod.py `
  --hotels ALL `
=======
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
>>>>>>> 9c4e6b400dbce8e3b0496c59908e3dc2ca739ae6
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

<<<<<<< HEAD
### 7.4 Single hotel modeling
=======
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
>>>>>>> 9c4e6b400dbce8e3b0496c59908e3dc2ca739ae6
```powershell
python run_dynamic_demand_modeling_prod.py `
  --hotels Anvil_Hotel `
  --stay-start 2025-07-01 `
  --stay-end 2026-02-28 `
  --realized-cutoff 2026-03-06
```

<<<<<<< HEAD
## 8) Commands (template mode)

### 8.1 Preprocessing template
1. Edit config block in `template_dynamic_demand_preprocessing_prod.py`
2. Run:
=======
Modeling template:
>>>>>>> 9c4e6b400dbce8e3b0496c59908e3dc2ca739ae6
```powershell
python template_dynamic_demand_preprocessing_prod.py
```

<<<<<<< HEAD
### 8.2 Modeling template
1. Edit config block in `template_dynamic_demand_modeling_prod.py`
2. Run:
```powershell
python template_dynamic_demand_modeling_prod.py
```

## 9) Outputs

Preprocessing:
- `output/01_long_ls_panel/`
- `output/02_model_input/`
- `output/06_reporting/`

Modeling:
- `output/07_model_results/`

## 10) Common failures and fixes

1. `DB config not found`:
- Ensure `db_config.py` exists in this folder.
- Or pass `--db-config` path in wrapper mode.

2. `No rows after filtering`:
- Validate hotel IDs exactly match `hotels.global_id`.
- Confirm date range has data.

3. Modeling cannot find model-input file:
- Run preprocessing first.
- Confirm `output/02_model_input/all_hotels_model_input_dynamic_baseline.csv` exists.

4. Single-hotel modeling gives no folds:
- Hotel may not meet minimum fold train/test row thresholds.
=======
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
>>>>>>> 9c4e6b400dbce8e3b0496c59908e3dc2ca739ae6
