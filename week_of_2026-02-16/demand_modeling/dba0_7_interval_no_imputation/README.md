# DBA0-7 Interval Pipeline (Direct Postgres, No Imputation)

This pipeline reads data directly from Postgres tables and builds DBA0..7 demand datasets with no statistical imputation.

## Scope

- Target: bookings demand (`gross_bookings`)
- Window: DBA 0..7 only
- Time cutoff: use records from **July 1, 2025 onward** for time-based reliability
- Own price: reconstructed from `rate_amounts.created_at/updated_at` validity intervals
- Competitor price: reconstructed from `ota_rates.created_at/updated_at` validity intervals
- Own base-rate rule: minimum price across room types per `(stay_date, decision_date)`
- No imputation is performed

## Source Tables (Direct DB Reads)

- `reservations` (demand labels)
- `rate_amounts` (own prices)
- `ota_rates` (competitor prices)
- `inventories` (inventory/occupancy)
- `hotels` (active hotel scope + competitor mapping)

## How Interval Reconstruction Works

For both own and competitor prices:
1. Take each record's `created_at` as start date.
2. Take `updated_at` (or fallback) as end date.
3. Expand to daily values in overlap with `[stay_date-7, stay_date]`.
4. Join to the DBA scaffold by exact `(stay_date, decision_date)`.
5. Apply cutoff `created_at >= 2025-07-01`.

This avoids model-based imputation and uses the operational "price stayed same until updated" logic.

Demand rows are aligned with `decision_date >= 2025-07-01` to keep the panel consistent with the same reliable period.

Interpretation of cutoff:
- If your concern is specifically price reliability, the core cutoff is on price `created_at`.
- We also align demand rows by `decision_date >= 2025-07-01` so labels and price period are consistent.

## Run

From repository root:

```powershell
python "week_of_2026-02-16\demand_modeling\dba0_7_interval_no_imputation\run_dba0_7_interval_pipeline.py"
```

DB config loading order:
1. `nandan_notebooks/configs/db_config.py` (`DB_CONFIG` dict), else
2. environment variables:
   - `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, optional `DB_SSLMODE`

Replication note:
- Replace `DB_CONFIG` with your own database credentials when teammates replicate.
- Do not commit personal or production secrets to GitHub.

## Outputs

- `output/01_long/*_long_dba0_7_interval.csv`
- `output/01_long/all_hotels_long_dba0_7_interval.csv`
- `output/02_preprocessed/*_wide_dba0_7_interval.csv`
- `output/02_preprocessed/all_hotels_wide_dba0_7_interval.csv`
- `output/03_features/*_features_dba0_7_interval.csv`
- `output/03_features/all_hotels_features_dba0_7_interval.csv`
- `output/06_reporting/preprocessing_summary_hotelwise.csv`
- `output/06_reporting/pipeline_summary.md`
