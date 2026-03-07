# Dynamic Demand Prod Folder (Standalone Runbook)

This `prod/` folder is fully runnable by itself and can be shared as-is.

## 1) What is included

### Core executables (standalone copies)
- `run_dynamic_demand_baseline.py`
- `run_dynamic_demand_models_gross.py`

### Wrapper scripts (CLI parameterized)
- `run_dynamic_demand_preprocessing_prod.py`
- `run_dynamic_demand_modeling_prod.py`

### Template scripts (config-block style)
- `template_dynamic_demand_preprocessing_prod.py`
- `template_dynamic_demand_modeling_prod.py`

### Config and dependencies
- `db_config_template.py`
- `requirements.txt`

## 2) What each script does

### A) `run_dynamic_demand_baseline.py` (core preprocessing)
Purpose:
- Connects to Postgres.
- Pulls hotels, inventory, reservations, own price, compset price.
- Builds DBA0..DBA7 hotel-day panel.
- Creates targets and engineered features.
- Writes preprocessing outputs.

Hotel scope:
- Default: all active hotels (`TARGET_HOTELS = []`).
- Override via CLI:
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
- Loads model input.
- Trains baseline, GLM, and XGBoost models.
- Runs grouped random date CV.
- Produces performance metrics, SHAP outputs, GLM coefficients, diagnostics.

Hotel scope:
- Default: all hotels present in model input.
- Override via CLI:
  - `--hotels ALL`
  - `--hotels Anvil_Hotel`
  - `--hotels "Anvil_Hotel,Ozarker_Lodge"`

Outputs:
- `output/07_model_results/`

---

### C) `run_dynamic_demand_preprocessing_prod.py` (wrapper)
Purpose:
- Runs preprocessing core script with CLI-supplied params (dates/hotels/db-config/source path).
- Best for automation and scheduled jobs.

Default source:
- `./run_dynamic_demand_baseline.py`

---

### D) `run_dynamic_demand_modeling_prod.py` (wrapper)
Purpose:
- Runs modeling core script with CLI-supplied params (dates/seeds/folds/targets/hotels/source path).
- Best for automation and scheduled jobs.

Default source:
- `./run_dynamic_demand_models_gross.py`

---

### E) `template_dynamic_demand_preprocessing_prod.py` (template)
Purpose:
- Same as preprocessing wrapper, but configuration is edited in-file at top.

Default source:
- `./run_dynamic_demand_baseline.py`

Default DB config path:
- `./db_config.py`

---

### F) `template_dynamic_demand_modeling_prod.py` (template)
Purpose:
- Same as modeling wrapper, but configuration is edited in-file at top.

Default source:
- `./run_dynamic_demand_models_gross.py`

## 3) Prerequisites

- Python installed
- Network access to Postgres
- Valid DB credentials

Install dependencies (from inside `prod/`):
```powershell
pip install -r requirements.txt
```

## 4) DB config setup

1. Copy:
- `db_config_template.py` -> `db_config.py`

2. Fill:
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

Use wrapper CLI mode for production.

Use template mode only when you prefer in-file config editing.

## 6) End-to-end run order

1. Preprocessing
2. Modeling

## 7) Wrapper mode commands

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
Change dates in wrapper mode if required to override dates in core files. 

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

## 8) Template mode commands

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

## 9) Output map

Preprocessing:
- `output/01_long_ls_panel/`
- `output/02_model_input/`
- `output/06_reporting/`

Modeling:
- `output/07_model_results/`

## 10) Common failures and fixes

1. DB config not found:
- Ensure `db_config.py` exists in this folder.
- Or pass explicit `--db-config` path.

2. No rows after hotel/date filtering:
- Validate hotel IDs against `hotels.global_id`.
- Check data availability for date window.

3. Modeling missing model-input file:
- Run preprocessing first.
- Confirm `output/02_model_input/all_hotels_model_input_dynamic_baseline.csv` exists.

4. No folds for single hotel:
- Hotel may not meet minimum fold train/test thresholds.
