# Prod Templates and Wrappers (Core Pipeline Untouched)

This folder is a production handoff layer around your current scripts:
- preprocessing source: `../run_dynamic_demand_baseline.py`
- modeling source: `../run_dynamic_demand_models_gross.py`

No changes are required in your current pipeline files.

## Files in this folder
- `run_dynamic_demand_preprocessing_prod.py`
- `run_dynamic_demand_modeling_prod.py`
- `template_dynamic_demand_preprocessing_prod.py`
- `template_dynamic_demand_modeling_prod.py`

## Two ways to run
1. CLI wrapper mode (best for scheduled jobs / automation).
2. Template mode (best for prod engineers who prefer editing one config block in Python).

## Reproducibility contract
- Date window is explicit.
- Hotels are explicit (or ALL).
- Seeds/folds are explicit.
- Target columns are explicit.
- Source scripts remain unchanged.

## DB config requirement
Preprocessing needs a Python file that defines:

```python
DB_CONFIG = {
  "host": "...",
  "port": 5432,
  "database": "...",
  "user": "...",
  "password": "...",
  "sslmode": "require"
}
```

You can use `prod/db_config_template.py`, fill values, and save as `prod/db_config.py`.
You can also reuse `../db_config_template.py` and save it as `../db_config.py`.

## Option 1: CLI wrapper mode
### Preprocessing
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

### Modeling
```powershell
python week_of_2026-03-02/demand_modeling/dynamic_demand_baseline/prod/run_dynamic_demand_modeling_prod.py `
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

## Option 2: Template mode
### Preprocessing template
Edit config block at top of:
- `template_dynamic_demand_preprocessing_prod.py`

Then run:
```powershell
python week_of_2026-03-02/demand_modeling/dynamic_demand_baseline/prod/template_dynamic_demand_preprocessing_prod.py
```

### Modeling template
Edit config block at top of:
- `template_dynamic_demand_modeling_prod.py`

Then run:
```powershell
python week_of_2026-03-02/demand_modeling/dynamic_demand_baseline/prod/template_dynamic_demand_modeling_prod.py
```

## What each step writes
Preprocessing outputs:
- `output/01_long_ls_panel`
- `output/02_model_input`
- `output/06_reporting`

Modeling outputs:
- `output/07_model_results`

## Handoff recommendation for Dan/prod
Send this `prod/` folder plus:
- `run_dynamic_demand_baseline.py`
- `run_dynamic_demand_models_gross.py`
- `prod/db_config_template.py`
- `requirements.txt`

This keeps implementation logic fixed while letting prod control runtime parameters.
