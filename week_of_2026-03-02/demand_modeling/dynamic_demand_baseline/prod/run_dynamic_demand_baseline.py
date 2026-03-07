from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import psycopg2

# Global modeling window + scope: DBA0-7 lead times and date cutoffs.
DBA_DAYS = list(range(0, 8))
# Leave empty to run all active hotels from the hotels table.
# To run a subset, populate with explicit global_id values, e.g.:
# TARGET_HOTELS = ["Anvil_Hotel", "Ozarker_Lodge"]
TARGET_HOTELS: List[str] = []
PRICE_CREATED_AT_START = pd.Timestamp("2025-07-01")
DECISION_DATE_START = pd.Timestamp("2025-07-01")
# Freeze modeling extraction window for aligned pipeline comparisons.
MODELING_STAY_END = pd.Timestamp("2026-02-28")
# Keep realized cutoff as "today" for metadata/report traceability.
REALIZED_CUTOFF = pd.Timestamp(date.today())
WOY_PERIOD = 53.0
OUTLIER_LOWER_PCT = 1.0
OUTLIER_UPPER_PCT = 99.0

# Output path container used across preprocessing and reporting writes.
@dataclass
class Paths:
    base: Path
    out_root: Path
    out_long: Path
    out_model: Path
    out_rep: Path

# Create output folders (long panel, model input, reporting) and return resolved paths.
def build_paths() -> Paths:
    base = Path(__file__).resolve().parent
    out_root = base / "output"
    out_long = out_root / "01_long_ls_panel"
    out_model = out_root / "02_model_input"
    out_rep = out_root / "06_reporting"
    for p in [out_long, out_model, out_rep]:
        p.mkdir(parents=True, exist_ok=True)
    return Paths(base, out_root, out_long, out_model, out_rep)

# Normalize arbitrary IDs/names into safe lowercase tokens for column naming.
def sanitize_token(s: str) -> str:
    tok = re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_").lower()
    return tok or "unknown"

# Load DB credentials from local file first, otherwise from environment variables
def load_db_config(base: Path) -> Dict[str, str]:
    local_cfg = base / "db_config.py"
    if local_cfg.exists():
        spec = importlib.util.spec_from_file_location("local_db_config", local_cfg)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        if hasattr(module, "DB_CONFIG") and isinstance(module.DB_CONFIG, dict):
            return module.DB_CONFIG

    env_keys = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    if all(k in os.environ for k in env_keys):
        return {
            "host": os.environ["DB_HOST"],
            "port": os.environ["DB_PORT"],
            "database": os.environ["DB_NAME"],
            "user": os.environ["DB_USER"],
            "password": os.environ["DB_PASSWORD"],
            "sslmode": os.environ.get("DB_SSLMODE", "require"),
        }

    raise RuntimeError(
        "DB config not found. Add local db_config.py with DB_CONFIG dict or set DB_* env vars."
    )


def parse_hotels_arg(raw: str) -> List[str]:
    token = str(raw or "").strip()
    if not token or token.upper() in {"ALL", "*"}:
        return []
    return sorted({h.strip() for h in token.split(",") if h.strip()})

# Open PostgreSQL connection using loaded DB config.
def get_db_connection(db_config: Dict[str, str]):
    return psycopg2.connect(
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
        sslmode=db_config.get("sslmode", "require"),
    )

# Pull active hotels and parse competitor mapping from hotel_settings JSON.
def get_active_hotels_and_comp_map(conn, target_hotels: Sequence[str]) -> tuple[List[str], Dict[str, List[str]]]:
    q = """
    SELECT global_id, hotel_settings
    FROM hotels
    WHERE LOWER(COALESCE(status, '')) = 'active'
    ORDER BY global_id;
    """
    rows = pd.read_sql(q, conn)
    if rows.empty:
        raise RuntimeError("No active hotels found in hotels table.")

    active = set(rows["global_id"].astype(str).tolist())
    hotels = [h for h in target_hotels if h in active]
    if not hotels:
        hotels = sorted(active)

    comp_map: Dict[str, List[str]] = {}
    subset = rows[rows["global_id"].isin(hotels)].copy()
    for _, r in subset.iterrows():
        h = str(r["global_id"])
        comps: List[str] = []
        raw = r["hotel_settings"]
        if raw is not None and str(raw).strip():
            try:
                settings = raw if isinstance(raw, dict) else json.loads(raw)
            except Exception:
                settings = {}
            for c in settings.get("competitors", []):
                if isinstance(c, dict):
                    cid = c.get("hotel_id") or c.get("name")
                    if cid:
                        comps.append(str(cid))
        comp_map[h] = sorted(set(comps))

    return hotels, comp_map

# Extract stay-date level inventory (total rooms) for scaffold creation
def extract_inventory_stays(conn, hotel_id: str, stay_date_start: pd.Timestamp, stay_date_end: pd.Timestamp) -> pd.DataFrame:
    q = """
    SELECT
        stay_date::date AS stay_date,
        SUM(total_qty) AS total_rooms
    FROM inventories
    WHERE hotel_id = %s
      AND is_active = TRUE
      AND total_qty > 0
      AND stay_date::date >= %s
      AND stay_date::date <= %s
    GROUP BY stay_date::date
    ORDER BY stay_date::date;
    """
    df = pd.read_sql(q, conn, params=(hotel_id, stay_date_start.date(), stay_date_end.date()))
    if df.empty:
        return df
    df["stay_date"] = pd.to_datetime(df["stay_date"])
    df["total_rooms"] = pd.to_numeric(df["total_rooms"], errors="coerce")
    return df

# Build leakage-safe scaffold: one row per stay_date x DBA day (decision_date derived from stay_date).
def build_inventory_scaffold(inv_stays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sd in inv_stays["stay_date"].tolist():
        for dba in DBA_DAYS:
            rows.append(
                {
                    "stay_date": sd,
                    "decision_date": sd - pd.Timedelta(days=dba),
                    "lead_time": dba,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["stay_date", "decision_date"]).reset_index(drop=True)

# Extract daily demand labels from reservations, deduplicated at reservation_id + stay_date.
def extract_demand_daily_labels(
    conn,
    hotel_id: str,
    decision_date_start: pd.Timestamp,
    stay_date_start: pd.Timestamp,
    stay_date_end: pd.Timestamp,
) -> pd.DataFrame:
    q = """
    SELECT
        stay_date::date AS stay_date,
        DATE(booked_at) AS decision_date,
        COUNT(*) AS gross_bookings,
        SUM(COALESCE(number_of_rooms, 0)) AS gross_rooms_pickup,
        SUM(
            CASE
                WHEN canceled_at IS NULL AND status_code NOT IN ('Canceled', 'NoShow')
                THEN 1 ELSE 0
            END
        ) AS net_bookings,
        SUM(
            CASE
                WHEN canceled_at IS NULL AND status_code NOT IN ('Canceled', 'NoShow')
                THEN COALESCE(number_of_rooms, 0) ELSE 0
            END
        ) AS net_rooms_pickup
    FROM (
        SELECT DISTINCT ON (reservation_id, stay_date)
            reservation_id,
            stay_date,
            booked_at,
            number_of_rooms,
            price,
            canceled_at,
            status_code,
            pms_updated_at
        FROM reservations
        WHERE hotel_id = %s
          AND booked_at IS NOT NULL
          AND stay_date::date >= %s
          AND stay_date::date <= %s
        ORDER BY reservation_id, stay_date, pms_updated_at DESC
    ) dedup
    WHERE COALESCE(price, 0) > 0
    GROUP BY stay_date::date, DATE(booked_at)
    ORDER BY stay_date::date, DATE(booked_at);
    """
    df = pd.read_sql(
        q,
        conn,
        params=(hotel_id, stay_date_start.date(), stay_date_end.date()),
    )
    if df.empty:
        return df

    df["stay_date"] = pd.to_datetime(df["stay_date"])
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["gross_bookings"] = pd.to_numeric(df["gross_bookings"], errors="coerce").fillna(0.0)
    df["gross_rooms_pickup"] = pd.to_numeric(df["gross_rooms_pickup"], errors="coerce").fillna(0.0)
    df["net_bookings"] = pd.to_numeric(df["net_bookings"], errors="coerce").fillna(0.0)
    df["net_rooms_pickup"] = pd.to_numeric(df["net_rooms_pickup"], errors="coerce").fillna(0.0)
    # Compute lead_time in days and keep only DBA0..DBA7 rows.
    df["lead_time"] = (df["stay_date"] - df["decision_date"]).dt.days
    return df[df["lead_time"].isin(DBA_DAYS)].copy()

# Reconstruct own-price daily intervals from rate_amounts and aggregate min/avg price by day.
def extract_own_price_interval_daily(
    conn,
    hotel_id: str,
    created_at_start: pd.Timestamp,
    stay_date_start: pd.Timestamp,
    stay_date_end: pd.Timestamp,
) -> pd.DataFrame:
    q = """
    SELECT
        stay_date::date AS stay_date,
        DATE(created_at) AS observed_date,
        MIN(price::float) AS bar_price
    FROM rate_amounts
    WHERE hotel_id = %s
      AND stay_date IS NOT NULL
      AND created_at IS NOT NULL
      AND price IS NOT NULL
      AND price > 0
      AND stay_date::date >= %s
      AND stay_date::date <= %s
      AND DATE(created_at) >= (%s::date - INTERVAL '90 day')
    GROUP BY stay_date::date, DATE(created_at)
    ORDER BY stay_date::date, DATE(created_at);
    """
    obs = pd.read_sql(
        q,
        conn,
        params=(hotel_id, stay_date_start.date(), stay_date_end.date(), created_at_start.date()),
    )
    if obs.empty:
        return pd.DataFrame(
            columns=["stay_date", "decision_date", "own_price_min", "own_price_avg", "own_price_bar"]
        )

    obs["stay_date"] = pd.to_datetime(obs["stay_date"])
    obs["observed_date"] = pd.to_datetime(obs["observed_date"])
    obs["bar_price"] = pd.to_numeric(obs["bar_price"], errors="coerce")

    # LOCF by stay_date: for each DBA day, use latest observed BAR at or before decision_date.
    rows: List[Dict[str, object]] = []
    for stay_date, g in obs.groupby("stay_date"):
        g = g.sort_values("observed_date").drop_duplicates("observed_date", keep="last")
        for dba in DBA_DAYS:
            decision_date = stay_date - pd.Timedelta(days=dba)
            valid = g[g["observed_date"] <= decision_date]
            if valid.empty:
                continue
            price = float(valid.iloc[-1]["bar_price"])
            rows.append(
                {
                    "stay_date": stay_date,
                    "decision_date": decision_date,
                    # Keep all own-price variants aligned to BAR for backward compatibility.
                    "own_price_min": price,
                    "own_price_avg": price,
                    "own_price_bar": price,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["stay_date", "decision_date", "own_price_min", "own_price_avg", "own_price_bar"]
        )
    df["stay_date"] = pd.to_datetime(df["stay_date"])
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["own_price_min"] = pd.to_numeric(df["own_price_min"], errors="coerce")
    df["own_price_avg"] = pd.to_numeric(df["own_price_avg"], errors="coerce")
    df["own_price_bar"] = pd.to_numeric(df["own_price_bar"], errors="coerce")
    df = df.sort_values(["stay_date", "decision_date"]).reset_index(drop=True)
    return df

# Reconstruct competitor daily interval prices from ota_rates for all mapped competitors.
def extract_comp_prices_interval_daily(
    conn,
    competitor_ids: Sequence[str],
    created_at_start: pd.Timestamp,
    stay_date_start: pd.Timestamp,
    stay_date_end: pd.Timestamp,
) -> pd.DataFrame:
    if not competitor_ids:
        return pd.DataFrame(columns=["competitor_id", "stay_date", "decision_date", "comp_price_raw"])

    q = """
    SELECT
        competitor_id,
        stay_date,
        observed_date,
        comp_price_obs
    FROM (
        SELECT
            hotel_id::text AS competitor_id,
            stay_date::date AS stay_date,
            DATE(created_at) AS observed_date,
            MIN(price::float) AS comp_price_obs
        FROM ota_rates
        WHERE hotel_id = ANY(%s)
          AND stay_date IS NOT NULL
          AND created_at IS NOT NULL
          AND price IS NOT NULL
          AND price > 0
          AND stay_date::date >= %s
          AND stay_date::date <= %s
          AND DATE(created_at) >= (%s::date - INTERVAL '90 day')
        GROUP BY hotel_id::text, stay_date::date, DATE(created_at)
    ) s
    ORDER BY competitor_id, stay_date, observed_date;
    """
    obs = pd.read_sql(
        q,
        conn,
        params=(list(competitor_ids), stay_date_start.date(), stay_date_end.date(), created_at_start.date()),
    )
    if obs.empty:
        return pd.DataFrame(columns=["competitor_id", "stay_date", "decision_date", "comp_price_raw"])

    obs["stay_date"] = pd.to_datetime(obs["stay_date"])
    obs["observed_date"] = pd.to_datetime(obs["observed_date"])
    obs["comp_price_obs"] = pd.to_numeric(obs["comp_price_obs"], errors="coerce")

    # LOCF by (competitor, stay_date): latest observed quote at or before decision_date for DBA0..7.
    rows: List[Dict[str, object]] = []
    for (competitor_id, stay_date), g in obs.groupby(["competitor_id", "stay_date"]):
        g = g.sort_values("observed_date").drop_duplicates("observed_date", keep="last")
        for dba in DBA_DAYS:
            decision_date = stay_date - pd.Timedelta(days=dba)
            valid = g[g["observed_date"] <= decision_date]
            if valid.empty:
                continue
            rows.append(
                {
                    "competitor_id": competitor_id,
                    "stay_date": stay_date,
                    "decision_date": decision_date,
                    "comp_price_raw": float(valid.iloc[-1]["comp_price_obs"]),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["competitor_id", "stay_date", "decision_date", "comp_price_raw"])
    df["stay_date"] = pd.to_datetime(df["stay_date"])
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["comp_price_raw"] = pd.to_numeric(df["comp_price_raw"], errors="coerce")
    df = df.sort_values(["competitor_id", "stay_date", "decision_date"]).reset_index(drop=True)
    return df

# Pivot competitor long data to wide raw feature columns (comp_price_raw__*).
def build_comp_wide_features(comp_df: pd.DataFrame) -> pd.DataFrame:
    if comp_df.empty:
        return pd.DataFrame(columns=["stay_date", "decision_date", "comp_count_available"])

    d = comp_df.copy()
    d["comp_col"] = d["competitor_id"].map(lambda x: f"comp_price_raw__{sanitize_token(x)}")

    wide = (
        d.pivot_table(
            index=["stay_date", "decision_date"],
            columns="comp_col",
            values="comp_price_raw",
            aggfunc="first",
        )
        .reset_index()
    )
    # Count how many competitor prices are available per row.
    comp_cols = [c for c in wide.columns if c.startswith("comp_price_raw__")]
    if comp_cols:
        wide["comp_count_available"] = wide[comp_cols].notna().sum(axis=1).astype(int)
    else:
        wide["comp_count_available"] = 0

    wide.columns = [str(c) for c in wide.columns]
    return wide

# Choose own-price signal (min vs avg) by missingness; tie-break to avg for smoother signal.
def choose_own_price_variant(panel: pd.DataFrame) -> tuple[str, str]:
    if "own_price_bar" in panel.columns and panel["own_price_bar"].notna().any():
        return "own_price_bar", "bar_min_roomtype_intraday_avg"

    miss_min = float(panel["own_price_min"].isna().mean()) if "own_price_min" in panel.columns else 1.0
    miss_avg = float(panel["own_price_avg"].isna().mean()) if "own_price_avg" in panel.columns else 1.0

    if miss_avg < miss_min:
        return "own_price_avg", "lower_missing"
    if miss_min < miss_avg:
        return "own_price_min", "lower_missing"

    # tie-breaker: use average price as smoother signal when coverage is equal
    return "own_price_avg", "tie_break_smoother_avg"


def clip_price_outliers(panel: pd.DataFrame, price_cols: Sequence[str]) -> pd.DataFrame:
    out = panel.copy()
    for c in price_cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        valid = s[s.notna()]
        if valid.empty:
            continue
        lo = float(np.percentile(valid, OUTLIER_LOWER_PCT))
        hi = float(np.percentile(valid, OUTLIER_UPPER_PCT))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if hi < lo:
            lo, hi = hi, lo
        out[c] = s.clip(lower=lo, upper=hi)
    return out

# Merge scaffold + demand + own price + compset into one hotel-level LS panel
def build_hotel_ls_panel(
    hotel_id: str,
    inv_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    own_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    decision_date_start: pd.Timestamp,
) -> pd.DataFrame:
    if inv_df.empty:
        return pd.DataFrame()

    scaffold = build_inventory_scaffold(inv_df)

    panel = scaffold.merge(
        inv_df[["stay_date", "total_rooms"]],
        on="stay_date",
        how="left",
    )

    if not demand_df.empty:
        panel = panel.merge(
            demand_df[
                [
                    "stay_date",
                    "decision_date",
                    "gross_bookings",
                    "gross_rooms_pickup",
                    "net_bookings",
                    "net_rooms_pickup",
                ]
            ],
            on=["stay_date", "decision_date"],
            how="left",
        )
    else:
        panel["gross_bookings"] = np.nan
        panel["gross_rooms_pickup"] = np.nan
        panel["net_bookings"] = np.nan
        panel["net_rooms_pickup"] = np.nan
    # Fill demand labels to zero so no-booking days are retained in training universe.
    panel["gross_bookings"] = panel["gross_bookings"].fillna(0.0)
    panel["gross_rooms_pickup"] = panel["gross_rooms_pickup"].fillna(0.0)
    panel["net_bookings"] = panel["net_bookings"].fillna(0.0)
    panel["net_rooms_pickup"] = panel["net_rooms_pickup"].fillna(0.0)

    if not own_df.empty:
        panel = panel.merge(own_df, on=["stay_date", "decision_date"], how="left")
    else:
        panel["own_price_min"] = np.nan
        panel["own_price_avg"] = np.nan
        panel["own_price_bar"] = np.nan

    comp_wide = build_comp_wide_features(comp_df)
    if not comp_wide.empty:
        panel = panel.merge(comp_wide, on=["stay_date", "decision_date"], how="left")
    else:
        panel["comp_count_available"] = 0

    comp_cols = sorted([c for c in panel.columns if c.startswith("comp_price_raw__")])
    own_cols = [c for c in ["own_price_min", "own_price_avg", "own_price_bar"] if c in panel.columns]
    price_cols = own_cols + comp_cols

    # Notebook-aligned step: forward-fill prices within each DBA bucket across stay_date.
    if price_cols:
        for dba in DBA_DAYS:
            idx = panel.loc[panel["lead_time"] == dba].sort_values("stay_date").index
            panel.loc[idx, price_cols] = panel.loc[idx, price_cols].ffill()

    # Notebook-aligned data-quality filter: drop competitor columns with >50% missing.
    sparse_comp_cols: List[str] = []
    for c in comp_cols:
        miss_pct = float(panel[c].isna().mean() * 100.0)
        if miss_pct > 50.0:
            sparse_comp_cols.append(c)
    if sparse_comp_cols:
        panel = panel.drop(columns=sparse_comp_cols)
        comp_cols = [c for c in comp_cols if c not in sparse_comp_cols]

    # Clip long-tail price spikes using per-feature 1st/99th percentiles.
    clip_cols = [c for c in (own_cols + comp_cols) if c in panel.columns]
    if clip_cols:
        panel = clip_price_outliers(panel, clip_cols)

    # Recompute per-row competitor availability after filling/filtering.
    if comp_cols:
        panel["comp_count_available"] = panel[comp_cols].notna().sum(axis=1).astype(int)
    else:
        panel["comp_count_available"] = 0

    panel["hotel_id"] = hotel_id
    # Store selected own price and reason for auditability.
    own_choice, own_reason = choose_own_price_variant(panel)
    panel["own_price_selected"] = panel[own_choice]
    panel["own_price_selected_type"] = own_choice.replace("own_price_", "")
    panel["own_price_selection_reason"] = own_reason
    # Final panel filter: enforce DBA range only (decision_date naturally derives from stay_date - DBA).
    panel = panel[panel["lead_time"].isin(DBA_DAYS)].copy()
    panel = panel.sort_values(["stay_date", "decision_date"]).reset_index(drop=True)
    return panel

# Add one stay-date-based seasonality block (no duplicate decision-date block).
def add_week_features(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["stay_dow"] = out["stay_date"].dt.dayofweek
    out["stay_month"] = out["stay_date"].dt.month
    out["stay_weekofyear"] = out["stay_date"].dt.isocalendar().week.astype(int)

    out["sin_woy"] = np.sin(2.0 * np.pi * out["stay_weekofyear"] / WOY_PERIOD)
    out["cos_woy"] = np.cos(2.0 * np.pi * out["stay_weekofyear"] / WOY_PERIOD)
    out["sin_month"] = np.sin(2.0 * np.pi * out["stay_month"] / 12.0)
    out["cos_month"] = np.cos(2.0 * np.pi * out["stay_month"] / 12.0)
    out["sin_dow"] = np.sin(2.0 * np.pi * out["stay_dow"] / 7.0)
    out["cos_dow"] = np.cos(2.0 * np.pi * out["stay_dow"] / 7.0)
    return out

# Build model-input table with leakage-safe targets and engineered predictors.
def build_model_input_no_leakage(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    # Primary demand target for pricing: gross room pickup count.
    out["target_gross_rooms_pickup"] = out["gross_rooms_pickup"]
    # Secondary diagnostic target retained for stability comparisons.
    out["target_gross_bookings"] = out["gross_bookings"]
    # Keep one own-price missingness flag (selected feature only).
    out["own_price_selected_missing"] = out["own_price_selected"].isna().astype("int8")

    # Canonical lead-time representation.
    out["dba"] = pd.to_numeric(out["lead_time"], errors="coerce").astype("Int64")
    out["dba"] = out["dba"].fillna(0).astype(int)
    out["dba_norm_7"] = out["dba"] / 7.0
    for d in range(1, 8):
        out[f"dba_{d}"] = (out["dba"] == d).astype("int8")
    out = add_week_features(out)

    # Competitor raw features and missingness flags.
    comp_cols = sorted([c for c in out.columns if c.startswith("comp_price_raw__")])
    out["log_own_price_selected"] = np.where(
        pd.to_numeric(out["own_price_selected"], errors="coerce") > 0,
        np.log(pd.to_numeric(out["own_price_selected"], errors="coerce")),
        np.nan,
    )
    comp_log_cols: List[str] = []
    for c in comp_cols:
        log_c = f"log_{c}"
        out[log_c] = np.where(
            pd.to_numeric(out[c], errors="coerce") > 0,
            np.log(pd.to_numeric(out[c], errors="coerce")),
            np.nan,
        )
        out[f"{c}_missing"] = out[c].isna().astype("int8")
        comp_log_cols.append(log_c)

    # Keep only fields used by modeling and diagnostics.
    base_cols = [
        "hotel_id",
        "stay_date",
        "decision_date",
        "lead_time",
        "dba",
        "dba_norm_7",
        "dba_1",
        "dba_2",
        "dba_3",
        "dba_4",
        "dba_5",
        "dba_6",
        "dba_7",
        "total_rooms",
        "own_price_selected",
        "log_own_price_selected",
        "own_price_selected_type",
        "own_price_selection_reason",
        "comp_count_available",
        "stay_dow",
        "stay_month",
        "sin_woy",
        "cos_woy",
        "sin_month",
        "cos_month",
        "sin_dow",
        "cos_dow",
        "own_price_selected_missing",
    ]
    comp_missing_cols = [f"{c}_missing" for c in comp_cols]
    target_cols = ["target_gross_rooms_pickup", "target_gross_bookings"]
    # Keep only expected columns that exist (robust to per-hotel feature sparsity).
    keep_cols = base_cols + comp_cols + comp_log_cols + comp_missing_cols + target_cols
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols].copy()

# Build stricter model-ready subset used by GLMs: requires own price, valid target, and compset coverage.
def build_model_ready_subset(model_df: pd.DataFrame) -> pd.DataFrame:
    req = ["own_price_selected", "target_gross_rooms_pickup"]
    out = model_df.dropna(subset=req).copy()
    out = out[out["target_gross_rooms_pickup"] >= 0].copy()
    out = out[out["comp_count_available"] > 0].copy()
    return out

# Compute per-hotel QA stats: coverage, missingness, zero-demand rate, target means
def summarize_hotel(panel: pd.DataFrame, model_df: pd.DataFrame) -> Dict[str, float]:
    n = len(panel)
    complete_price_mask = panel["own_price_selected"].notna() & (panel["comp_count_available"] > 0)

    return {
        "hotel_id": str(panel["hotel_id"].iloc[0]) if n else "unknown",
        "rows_ls_panel": int(n),
        "stay_dates": int(panel["stay_date"].nunique()) if n else 0,
        "decision_dates": int(panel["decision_date"].nunique()) if n else 0,
        "zero_booking_rows_pct": float((panel["gross_rooms_pickup"] == 0).mean() * 100.0) if n else np.nan,
        "mean_target_gross_bookings": float(panel["gross_bookings"].mean()) if n else np.nan,
        "mean_target_gross_rooms_pickup": float(panel["gross_rooms_pickup"].mean()) if n else np.nan,
        "mean_target_net_bookings": float(panel["net_bookings"].mean()) if n else np.nan,
        "mean_target_net_rooms_pickup": float(panel["net_rooms_pickup"].mean()) if n else np.nan,
        "own_price_min_missing_pct": float(panel["own_price_min"].isna().mean() * 100.0) if n else np.nan,
        "own_price_avg_missing_pct": float(panel["own_price_avg"].isna().mean() * 100.0) if n else np.nan,
        "own_price_selected_missing_pct": float(panel["own_price_selected"].isna().mean() * 100.0) if n else np.nan,
        "comp_no_quote_pct": float((panel["comp_count_available"] == 0).mean() * 100.0) if n else np.nan,
        "comp_ge1_pct": float((panel["comp_count_available"] >= 1).mean() * 100.0) if n else np.nan,
        "comp_ge2_pct": float((panel["comp_count_available"] >= 2).mean() * 100.0) if n else np.nan,
        "complete_price_rows_pct": float(complete_price_mask.mean() * 100.0) if n else np.nan,
        "complete_price_rows": int(complete_price_mask.sum()) if n else 0,
        "model_input_rows": int(len(model_df)),
        "own_price_selected_type": str(panel["own_price_selected_type"].iloc[0]) if n else "unknown",
        "own_price_selection_reason": str(panel["own_price_selection_reason"].iloc[0]) if n else "unknown",
    }

# QA breakdown by hotel and lead_time for booking sparsity and feature completeness checks.
def summarize_by_hotel_dba(model_df: pd.DataFrame) -> pd.DataFrame:
    if model_df.empty:
        return pd.DataFrame()

    grp = model_df.groupby(["hotel_id", "lead_time"], as_index=False)
    out = grp.agg(
        rows=("target_gross_rooms_pickup", "size"),
        zero_booking_pct=("target_gross_rooms_pickup", lambda s: float((s == 0).mean() * 100.0)),
        own_price_selected_missing_pct=("own_price_selected_missing", lambda s: float(s.mean() * 100.0)),
        comp_no_quote_pct=("comp_count_available", lambda s: float((s == 0).mean() * 100.0)),
        mean_target_gross_rooms_pickup=("target_gross_rooms_pickup", "mean"),
        mean_target_gross_bookings=("target_gross_bookings", "mean"),
    )
    return out.sort_values(["hotel_id", "lead_time"]).reset_index(drop=True)


def build_compset_audit_tables(
    hotel_id: str,
    expected_comp_ids: Sequence[str],
    model_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected_cols = [f"comp_price_raw__{sanitize_token(cid)}" for cid in expected_comp_ids]
    total_rows = int(len(model_df))
    expected_n = len(expected_cols)

    rowwise = model_df[["hotel_id", "stay_date", "decision_date", "comp_count_available"]].copy()
    rowwise["expected_comp_count"] = expected_n
    rowwise["missing_comp_count"] = rowwise["expected_comp_count"] - rowwise["comp_count_available"]
    rowwise["comp_coverage_ratio"] = np.where(
        rowwise["expected_comp_count"] > 0,
        rowwise["comp_count_available"] / rowwise["expected_comp_count"],
        np.nan,
    )

    per_comp_rows: List[Dict[str, object]] = []
    for c in expected_cols:
        if c in model_df.columns:
            available_rows = int(model_df[c].notna().sum())
        else:
            available_rows = 0
        missing_rows = total_rows - available_rows
        missing_pct = (missing_rows / total_rows) * 100.0 if total_rows > 0 else np.nan
        if available_rows == 0:
            reason = "no_valid_quote_in_window_or_mapping_gap"
        elif missing_rows == 0:
            reason = "full_coverage"
        else:
            reason = "intermittent_quote_gaps"

        per_comp_rows.append(
            {
                "hotel_id": hotel_id,
                "competitor_col": c,
                "expected_in_comp_map": True,
                "available_rows": available_rows,
                "missing_rows": missing_rows,
                "missing_pct": missing_pct,
                "coverage_reason": reason,
            }
        )

    per_comp = pd.DataFrame(per_comp_rows)
    reason_summary = (
        per_comp.groupby(["hotel_id", "coverage_reason"], as_index=False)
        .agg(competitors=("competitor_col", "count"))
        .sort_values(["hotel_id", "competitors"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return rowwise, per_comp, reason_summary

# Build weighted portfolio QA summary across hotels.
def build_portfolio_summary(hotel_summary: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    if hotel_summary.empty:
        return pd.DataFrame()

    total_rows = float(hotel_summary["rows_ls_panel"].sum())
    w = hotel_summary["rows_ls_panel"] / total_rows if total_rows > 0 else 0.0

    portfolio = {
        "hotels_processed": int(hotel_summary["hotel_id"].nunique()),
        "rows_ls_panel_total": int(hotel_summary["rows_ls_panel"].sum()),
        "stay_dates_total": int(hotel_summary["stay_dates"].sum()),
        "model_input_rows_total": int(hotel_summary["model_input_rows"].sum()),
        "model_ready_rows_total": int(hotel_summary["complete_price_rows"].sum()),
        "model_ready_rows_pct_total": float((hotel_summary["complete_price_rows"].sum() / max(1, hotel_summary["model_input_rows"].sum())) * 100.0),
        "weighted_zero_booking_rows_pct": float((hotel_summary["zero_booking_rows_pct"] * w).sum()),
        "weighted_own_price_selected_missing_pct": float((hotel_summary["own_price_selected_missing_pct"] * w).sum()),
        "weighted_comp_no_quote_pct": float((hotel_summary["comp_no_quote_pct"] * w).sum()),
        "weighted_complete_price_rows_pct": float((hotel_summary["complete_price_rows_pct"] * w).sum()),
        "target_mean_gross_rooms_pickup": float(model_df["target_gross_rooms_pickup"].mean()) if len(model_df) else np.nan,
        "target_mean_gross_bookings": float(model_df["target_gross_bookings"].mean()) if len(model_df) else np.nan,
        "stay_date_end_applied": str(MODELING_STAY_END.date()),
        "realized_cutoff_applied": str(REALIZED_CUTOFF.date()),
        "decision_date_start_applied": str(DECISION_DATE_START.date()),
        "price_created_at_start_applied": str(PRICE_CREATED_AT_START.date()),
    }
    return pd.DataFrame([portfolio])

# Write human-readable markdown summary of preprocessing scope, assumptions, and outputs.
def write_markdown_summary(paths: Paths, hotels: List[str], hotel_summary: pd.DataFrame, portfolio: pd.DataFrame) -> None:
    lines = [
        "# Dynamic Demand Baseline (DBA0-7, Raw Compset, No Price Imputation)",
        "",
        "## Scope Locked",
        f"- Stay-date cutoff: `stay_date <= {MODELING_STAY_END.date()}`",
        f"- Realized cutoff reference: `{REALIZED_CUTOFF.date()}`",
        f"- Decision-date start: `decision_date >= {DECISION_DATE_START.date()}`",
        f"- Price interval start: `created_at >= {PRICE_CREATED_AT_START.date()}`",
        "- LS scaffold built from `inventories` (one row per stay_date x DBA0..7)",
        "- Demand labels from `reservations` (gross + cancellation-aware net reference)",
        "- Canonical lead-time uses DBA numeric + dummies (`dba_1`..`dba_7`, DBA0 reference)",
        "- Own price from `rate_amounts` interval logic with both min and avg outputs",
        "- Hotel-level own-price selection: min vs avg (coverage-first, tie -> avg)",
        "- Compset prices from `ota_rates` as raw competitor-level daily columns (no portfolio avg/min features)",
        "- Added stay-date seasonality block (`sin/cos woy`, `sin/cos month`, `sin/cos dow`)",
        "- Outlier clipping enabled for price features (1st/99th percentile)",
        "- Log transforms added for own/compset prices",
        "- Zero-booking rows retained",
        "- No feature price imputation applied in this baseline",
        "",
        "## Hotels Processed",
        f"- {', '.join(hotels)}",
        "",
        "## Key Outputs",
        "- `output/01_long_ls_panel/all_hotels_ls_panel_dba0_7_dynamic_baseline.csv`",
        "- `output/02_model_input/all_hotels_model_input_dynamic_baseline.csv`",
        "- `output/02_model_input/all_hotels_model_input_dynamic_baseline_model_ready.csv`",
        "- `output/06_reporting/qa_summary_hotelwise.csv`",
        "- `output/06_reporting/qa_summary_portfolio.csv`",
        "- `output/06_reporting/qa_summary_by_hotel_dba.csv`",
        "- `output/06_reporting/compset_coverage_rowwise.csv`",
        "- `output/06_reporting/compset_coverage_by_competitor.csv`",
        "- `output/06_reporting/compset_coverage_reason_summary.csv`",
        "- `output/06_reporting/model_input_sample_top100_per_hotel.csv`",
    ]

    if not portfolio.empty:
        p = portfolio.iloc[0].to_dict()
        lines.extend(
            [
                "",
                "## Portfolio Snapshot",
                f"- LS panel rows: {int(p['rows_ls_panel_total']):,}",
                f"- Model input rows: {int(p['model_input_rows_total']):,}",
                f"- Model-ready rows: {int(p['model_ready_rows_total']):,} ({p['model_ready_rows_pct_total']:.2f}%)",
                f"- Weighted own-price-selected missing: {p['weighted_own_price_selected_missing_pct']:.2f}%",
                f"- Weighted no-compset-quote rows: {p['weighted_comp_no_quote_pct']:.2f}%",
                f"- Weighted zero-booking rows: {p['weighted_zero_booking_rows_pct']:.2f}%",
                f"- Mean gross rooms pickup target (primary): {p['target_mean_gross_rooms_pickup']:.4f}",
                f"- Mean gross bookings target: {p['target_mean_gross_bookings']:.4f}",
            ]
        )

    (paths.out_rep / "dynamic_baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")

def write_dan_email_draft(paths: Paths) -> None:
    msg = f"""Hi Dan,

Please find the updated dynamic DBA0-7 baseline package for review.

What I prepared:
1. LS panel from inventories (DBA0..DBA7 rows per stay-date).
2. Gross + net daily demand labels from reservations (zero-booking days retained).
3. Own-price interval reconstruction from rate_amounts with BAR-based selection.
4. Canonical lead-time encoding added (`dba` + `dba_1`..`dba_7`).
5. Raw competitor-level compset features from ota_rates (no compset avg/min feature dependence).
6. Added week-of-year cyclical features (sin/cos) and removed weekend flags from modeling inputs.
7. Added compset coverage audits (rowwise + per competitor + reason summary).

Files:
- output/02_model_input/all_hotels_model_input_dynamic_baseline.csv
- output/02_model_input/all_hotels_model_input_dynamic_baseline_model_ready.csv
- output/06_reporting/qa_summary_hotelwise.csv
- output/06_reporting/qa_summary_portfolio.csv
- output/06_reporting/qa_summary_by_hotel_dba.csv
- output/06_reporting/compset_coverage_rowwise.csv
- output/06_reporting/compset_coverage_by_competitor.csv
- output/06_reporting/compset_coverage_reason_summary.csv

Best,
Arjyahi
"""
    (paths.out_rep / "draft_email_to_dan.txt").write_text(msg, encoding="utf-8")


def safe_to_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    try:
        df.to_csv(path, index=index)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_rerun{path.suffix}")
        df.to_csv(alt, index=index)
        print(f"[WARN] File locked, wrote fallback: {alt}")
        return alt

# Main orchestration: run extraction -> feature engineering -> QA -> output exports for all hotels.
def main(target_hotels_override: Sequence[str] | None = None) -> None:
    paths = build_paths()
    db_config = load_db_config(paths.base)

    conn = get_db_connection(db_config)
    try:
        target_hotels = list(target_hotels_override) if target_hotels_override is not None else TARGET_HOTELS
        hotels, comp_map = get_active_hotels_and_comp_map(conn, target_hotels)
        print(f"[INFO] Active hotels in scope: {hotels}")

        long_frames: List[pd.DataFrame] = []
        model_frames: List[pd.DataFrame] = []
        qa_rows: List[Dict[str, float]] = []
        comp_row_audits: List[pd.DataFrame] = []
        comp_comp_audits: List[pd.DataFrame] = []
        comp_reason_audits: List[pd.DataFrame] = []
        # Per-hotel pipeline loop.
        for hotel in hotels:
            print(f"[INFO] Processing {hotel}")
            # Step 1: inventory extraction (defines potential row universe).
            inv = extract_inventory_stays(conn, hotel, DECISION_DATE_START, MODELING_STAY_END)
            if inv.empty:
                print(f"[WARN] {hotel}: no inventories rows in range; skipping")
                continue
            # Step 2: demand label extraction (gross + net).
            demand = extract_demand_daily_labels(
                conn,
                hotel,
                DECISION_DATE_START,
                DECISION_DATE_START,
                MODELING_STAY_END,
            ) # Step 3: own-price interval extraction.
            own = extract_own_price_interval_daily(
                conn,
                hotel,
                PRICE_CREATED_AT_START,
                DECISION_DATE_START,
                MODELING_STAY_END,
            ) # Step 4: competitor price extraction.
            comps = extract_comp_prices_interval_daily(
                conn,
                comp_map.get(hotel, []),
                PRICE_CREATED_AT_START,
                DECISION_DATE_START,
                MODELING_STAY_END,
            )
            # Step 5: merge into LS panel and engineer model features.
            panel = build_hotel_ls_panel(
                hotel,
                inv,
                demand,
                own,
                comps,
                DECISION_DATE_START,
            )
            if panel.empty:
                print(f"[WARN] {hotel}: assembled panel empty; skipping")
                continue
            # Step 6: build model-ready subset and write per-hotel outputs.
            model_df = build_model_input_no_leakage(panel)
            model_ready_df = build_model_ready_subset(model_df)

            safe_to_csv(panel, paths.out_long / f"{hotel}_ls_panel_dba0_7_dynamic_baseline.csv", index=False)
            safe_to_csv(model_df, paths.out_model / f"{hotel}_model_input_dynamic_baseline.csv", index=False)
            safe_to_csv(
                model_ready_df,
                paths.out_model / f"{hotel}_model_input_dynamic_baseline_model_ready.csv",
                index=False,
            )

            long_frames.append(panel)
            model_frames.append(model_df)
            qa_rows.append(summarize_hotel(panel, model_df))
            row_a, comp_a, reason_a = build_compset_audit_tables(hotel, comp_map.get(hotel, []), model_df)
            comp_row_audits.append(row_a)
            comp_comp_audits.append(comp_a)
            comp_reason_audits.append(reason_a)

        if not long_frames:
            raise RuntimeError("No hotels produced output. Check DB filters and date windows.")
        # Concatenate all hotels and write portfolio-level files + QA artifacts.
        all_long = pd.concat(long_frames, ignore_index=True)
        all_model = pd.concat(model_frames, ignore_index=True)
        all_model_ready = build_model_ready_subset(all_model)
        qa_hotel = pd.DataFrame(qa_rows).sort_values("hotel_id").reset_index(drop=True)
        qa_portfolio = build_portfolio_summary(qa_hotel, all_model)
        qa_by_hotel_dba = summarize_by_hotel_dba(all_model)

        safe_to_csv(all_long, paths.out_long / "all_hotels_ls_panel_dba0_7_dynamic_baseline.csv", index=False)
        safe_to_csv(all_model, paths.out_model / "all_hotels_model_input_dynamic_baseline.csv", index=False)
        safe_to_csv(
            all_model_ready,
            paths.out_model / "all_hotels_model_input_dynamic_baseline_model_ready.csv",
            index=False,
        )

        safe_to_csv(qa_hotel, paths.out_rep / "qa_summary_hotelwise.csv", index=False)
        safe_to_csv(qa_portfolio, paths.out_rep / "qa_summary_portfolio.csv", index=False)
        safe_to_csv(qa_by_hotel_dba, paths.out_rep / "qa_summary_by_hotel_dba.csv", index=False)
        safe_to_csv(
            pd.concat(comp_row_audits, ignore_index=True),
            paths.out_rep / "compset_coverage_rowwise.csv",
            index=False,
        )
        safe_to_csv(
            pd.concat(comp_comp_audits, ignore_index=True),
            paths.out_rep / "compset_coverage_by_competitor.csv",
            index=False,
        )
        safe_to_csv(
            pd.concat(comp_reason_audits, ignore_index=True),
            paths.out_rep / "compset_coverage_reason_summary.csv",
            index=False,
        )

        sample = all_model.sort_values(["hotel_id", "stay_date", "decision_date"]).groupby("hotel_id", as_index=False).head(100)
        safe_to_csv(sample, paths.out_rep / "model_input_sample_top100_per_hotel.csv", index=False)

        write_markdown_summary(paths, hotels, qa_hotel, qa_portfolio)
        write_dan_email_draft(paths)

        print("[OK] Dynamic baseline pipeline completed.")
        print(f"[OK] LS panel output: {paths.out_long}")
        print(f"[OK] Model-input output: {paths.out_model}")
        print(f"[OK] QA/reporting output: {paths.out_rep}")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic demand preprocessing pipeline (DBA0-7)."
    )
    parser.add_argument(
        "--hotels",
        type=str,
        default="",
        help="Comma-separated hotel global_id list, or ALL. Default uses TARGET_HOTELS constant (empty means all active).",
    )
    args = parser.parse_args()
    cli_hotels = parse_hotels_arg(args.hotels)
    main(target_hotels_override=cli_hotels if args.hotels else None)
