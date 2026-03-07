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

### 7.4 Single hotel modeling
```powershell
python run_dynamic_demand_modeling_prod.py `
  --hotels Anvil_Hotel `
  --stay-start 2025-07-01 `
  --stay-end 2026-02-28 `
  --realized-cutoff 2026-03-06
```

## 8) Commands (template mode)

### 8.1 Preprocessing template
1. Edit config block in `template_dynamic_demand_preprocessing_prod.py`
2. Run:
```powershell
python template_dynamic_demand_preprocessing_prod.py
```

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
