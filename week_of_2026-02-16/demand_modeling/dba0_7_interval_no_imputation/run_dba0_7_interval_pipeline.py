from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import psycopg2


SEED = 42
np.random.seed(SEED)

DBA_DAYS = list(range(0, 8))
TARGET_HOTELS = [
    "Anvil_Hotel",
    "Arabella_Hotel_Sedona",
    "Bon_Dia_Residences",
    "Ozarker_Lodge",
    "h2hotel",
]
PRICE_CREATED_AT_START = pd.Timestamp("2025-07-01")
DECISION_DATE_START = pd.Timestamp("2025-07-01")


@dataclass
class Paths:
    base: Path
    out_root: Path
    out_long: Path
    out_prep: Path
    out_feat: Path
    out_rep: Path


def build_paths() -> Paths:
    base = Path(__file__).resolve().parent
    out_root = base / "output"
    out_long = out_root / "01_long"
    out_prep = out_root / "02_preprocessed"
    out_feat = out_root / "03_features"
    out_rep = out_root / "06_reporting"
    for p in [out_long, out_prep, out_feat, out_rep]:
        p.mkdir(parents=True, exist_ok=True)
    return Paths(base, out_root, out_long, out_prep, out_feat, out_rep)


def load_db_config(project_root: Path) -> Dict[str, str]:
    cfg_path = project_root / "nandan_notebooks" / "configs" 
    if cfg_path.exists():
        sys.path.insert(0, str(cfg_path))
        try:
            from db_config import DB_CONFIG  # type: ignore

            return DB_CONFIG
        except Exception:
            pass

    # Fallback used in earlier exploration work.
    legacy_cfg_path = project_root / "week_of_26th_Jan_2026" / "exploration_scripts"
    if legacy_cfg_path.exists():
        sys.path.insert(0, str(legacy_cfg_path))
        try:
            from db_connector import DB_CONFIG  # type: ignore

            return DB_CONFIG
        except Exception:
            pass

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
        "DB config not found. Expected nandan_notebooks/configs/db_config.py or DB_* env vars."
    )


def get_db_connection(db_config: Dict[str, str]):
    return psycopg2.connect(
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
        sslmode=db_config.get("sslmode", "require"),
    )


def sanitize_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()


def trim_mean(vals: np.ndarray, frac: float = 0.2) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    arr.sort()
    k = int(np.floor(arr.size * frac))
    if arr.size - 2 * k <= 0:
        return float(arr.mean())
    return float(arr[k : arr.size - k].mean())


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
        raw = r["hotel_settings"]
        comps: List[str] = []
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


def extract_demand_daily(conn, hotel_id: str, decision_date_start: pd.Timestamp) -> pd.DataFrame:
    q = """
    SELECT
        stay_date::date                   AS stay_date,
        DATE(booked_at)                   AS decision_date,
        COUNT(*)                          AS gross_bookings,
        SUM(number_of_rooms)              AS rooms_pickup
    FROM (
        SELECT DISTINCT ON (reservation_id, stay_date)
            reservation_id,
            stay_date,
            booked_at,
            number_of_rooms,
            canceled_at,
            status_code
        FROM reservations
        WHERE hotel_id = %s
          AND booked_at IS NOT NULL
          AND DATE(booked_at) >= %s
        ORDER BY reservation_id, stay_date, pms_updated_at DESC
    ) dedup
    WHERE canceled_at IS NULL
      AND status_code NOT IN ('Canceled', 'NoShow')
    GROUP BY stay_date::date, DATE(booked_at)
    ORDER BY stay_date::date, DATE(booked_at);
    """
    df = pd.read_sql(q, conn, params=(hotel_id, decision_date_start.date()))
    if df.empty:
        return df
    df["stay_date"] = pd.to_datetime(df["stay_date"])
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["gross_bookings"] = pd.to_numeric(df["gross_bookings"], errors="coerce").fillna(0).astype(float)
    df["rooms_pickup"] = pd.to_numeric(df["rooms_pickup"], errors="coerce").fillna(0).astype(float)
    df["lead_time"] = (df["stay_date"] - df["decision_date"]).dt.days
    return df[df["lead_time"] >= 0].copy()


def build_dba_scaffold(stay_dates: pd.Series) -> pd.DataFrame:
    rows = []
    for sd in sorted(stay_dates.dropna().unique()):
        sd = pd.Timestamp(sd).normalize()
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


def extract_own_prices_interval_daily(conn, hotel_id: str, created_at_start: pd.Timestamp) -> pd.DataFrame:
    q = """
    WITH raw AS (
        SELECT
            stay_date::date AS stay_date,
            created_at::date AS created_date,
            COALESCE(updated_at::date, stay_date::date) AS updated_date,
            price::float AS price
        FROM rate_amounts
        WHERE hotel_id = %s
          AND stay_date IS NOT NULL
          AND created_at IS NOT NULL
          AND created_at::date >= %s
          AND price IS NOT NULL
          AND price > 0
    ),
    bounded AS (
        SELECT
            stay_date,
            GREATEST(created_date, (stay_date - INTERVAL '7 day')::date) AS start_date,
            LEAST(updated_date, stay_date) AS end_date,
            price
        FROM raw
    ),
    expanded AS (
        SELECT
            stay_date,
            gs::date AS decision_date,
            price
        FROM bounded
        CROSS JOIN LATERAL generate_series(start_date, end_date, INTERVAL '1 day') gs
        WHERE start_date <= end_date
    )
    SELECT
        stay_date,
        decision_date,
        MIN(price) AS own_price
    FROM expanded
    GROUP BY stay_date, decision_date
    ORDER BY stay_date, decision_date;
    """
    df = pd.read_sql(q, conn, params=(hotel_id, created_at_start.date()))
    if df.empty:
        return df
    df["stay_date"] = pd.to_datetime(df["stay_date"])
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["own_price"] = pd.to_numeric(df["own_price"], errors="coerce")
    return df


def extract_comp_prices_interval_daily(
    conn, competitor_ids: Sequence[str], created_at_start: pd.Timestamp
) -> pd.DataFrame:
    if not competitor_ids:
        return pd.DataFrame(columns=["competitor_id", "stay_date", "decision_date", "comp_price"])

    q = """
    WITH raw AS (
        SELECT
            hotel_id::text AS competitor_id,
            stay_date::date AS stay_date,
            created_at::date AS created_date,
            COALESCE(updated_at::date, stay_date::date) AS updated_date,
            price::float AS price
        FROM ota_rates
        WHERE hotel_id = ANY(%s)
          AND stay_date IS NOT NULL
          AND created_at IS NOT NULL
          AND created_at::date >= %s
          AND price IS NOT NULL
          AND price > 0
    ),
    bounded AS (
        SELECT
            competitor_id,
            stay_date,
            GREATEST(created_date, (stay_date - INTERVAL '7 day')::date) AS start_date,
            LEAST(updated_date, stay_date) AS end_date,
            price
        FROM raw
    ),
    expanded AS (
        SELECT
            competitor_id,
            stay_date,
            gs::date AS decision_date,
            price
        FROM bounded
        CROSS JOIN LATERAL generate_series(start_date, end_date, INTERVAL '1 day') gs
        WHERE start_date <= end_date
    )
    SELECT
        competitor_id,
        stay_date,
        decision_date,
        AVG(price) AS comp_price
    FROM expanded
    GROUP BY competitor_id, stay_date, decision_date
    ORDER BY competitor_id, stay_date, decision_date;
    """
    df = pd.read_sql(q, conn, params=(list(competitor_ids), created_at_start.date()))
    if df.empty:
        return df
    df["stay_date"] = pd.to_datetime(df["stay_date"])
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["comp_price"] = pd.to_numeric(df["comp_price"], errors="coerce")
    return df


def extract_inventory(conn, hotel_id: str) -> pd.DataFrame:
    q = """
    SELECT
        stay_date::date AS stay_date,
        SUM(total_qty) AS total_rooms,
        SUM(definitive_sold) AS rooms_sold,
        SUM(definitive_availability) AS rooms_available
    FROM inventories
    WHERE hotel_id = %s
      AND is_active = TRUE
      AND total_qty > 0
    GROUP BY stay_date::date
    ORDER BY stay_date::date;
    """
    df = pd.read_sql(q, conn, params=(hotel_id,))
    if df.empty:
        return df
    df["stay_date"] = pd.to_datetime(df["stay_date"])
    for c in ["total_rooms", "rooms_sold", "rooms_available"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["occupancy"] = df["rooms_sold"] / df["total_rooms"].replace(0, np.nan)
    return df


def add_comp_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    comp_cols = [c for c in df.columns if c.startswith("comp_rate_")]
    if not comp_cols:
        df["comp_count_available"] = 0
        df["comp_median"] = np.nan
        df["comp_trimmed_mean_20"] = np.nan
        return df

    cm = df[comp_cols]
    df["comp_count_available"] = cm.notna().sum(axis=1).astype("int16")
    df["comp_median"] = cm.median(axis=1)
    df["comp_trimmed_mean_20"] = cm.apply(lambda r: trim_mean(r.values, 0.2), axis=1)
    return df


def build_hotel_long_panel(
    hotel_id: str,
    demand_df: pd.DataFrame,
    own_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    decision_date_start: pd.Timestamp,
) -> pd.DataFrame:
    if demand_df.empty:
        return pd.DataFrame()

    scaffold = build_dba_scaffold(demand_df["stay_date"])
    demand_daily = demand_df.groupby(["stay_date", "decision_date"], as_index=False).agg(
        gross_bookings=("gross_bookings", "sum"),
        rooms_pickup=("rooms_pickup", "sum"),
    )
    panel = scaffold.merge(demand_daily, on=["stay_date", "decision_date"], how="left")
    panel["gross_bookings"] = panel["gross_bookings"].fillna(0.0)
    panel["rooms_pickup"] = panel["rooms_pickup"].fillna(0.0)

    if not own_df.empty:
        panel = panel.merge(own_df, on=["stay_date", "decision_date"], how="left")
    else:
        panel["own_price"] = np.nan

    if not comp_df.empty:
        comp_wide = comp_df.pivot_table(
            index=["stay_date", "decision_date"],
            columns="competitor_id",
            values="comp_price",
            aggfunc="first",
        ).reset_index()
        rename = {}
        for c in comp_wide.columns:
            if c in {"stay_date", "decision_date"}:
                continue
            rename[c] = f"comp_rate_{sanitize_id(c)}"
        comp_wide = comp_wide.rename(columns=rename)
        panel = panel.merge(comp_wide, on=["stay_date", "decision_date"], how="left")

    if not inv_df.empty:
        panel = panel.merge(inv_df[["stay_date", "total_rooms", "rooms_sold", "rooms_available", "occupancy"]], on="stay_date", how="left")
    else:
        panel["total_rooms"] = np.nan
        panel["rooms_sold"] = np.nan
        panel["rooms_available"] = np.nan
        panel["occupancy"] = np.nan

    panel = panel.sort_values(["stay_date", "decision_date"]).reset_index(drop=True)
    panel = panel[panel["decision_date"] >= decision_date_start].copy()
    panel["cumulative_bookings"] = panel.groupby("stay_date")["gross_bookings"].cumsum()
    panel["cumulative_rooms"] = panel.groupby("stay_date")["rooms_pickup"].cumsum()
    panel["lead_time"] = (panel["stay_date"] - panel["decision_date"]).dt.days
    panel["hotel_id"] = hotel_id
    panel = add_comp_aggregates(panel)

    panel = panel[panel["lead_time"].isin(DBA_DAYS)].copy()
    return panel


def pivot_wide(panel: pd.DataFrame) -> pd.DataFrame:
    idx = ["hotel_id", "stay_date"]
    wide = panel[idx].drop_duplicates().sort_values(idx).reset_index(drop=True)
    day_cols = [
        "gross_bookings",
        "rooms_pickup",
        "own_price",
        "comp_median",
        "comp_trimmed_mean_20",
        "comp_count_available",
        "rooms_available",
        "total_rooms",
        "occupancy",
        "cumulative_bookings",
        "cumulative_rooms",
    ]
    for c in day_cols:
        if c not in panel.columns:
            continue
        piv = panel.pivot_table(index=idx, columns="lead_time", values=c, aggfunc="first").reindex(columns=DBA_DAYS)
        piv.columns = [f"{c}_dba{int(d)}" for d in piv.columns]
        wide = wide.merge(piv.reset_index(), on=idx, how="left")

    gb = [f"gross_bookings_dba{d}" for d in DBA_DAYS if f"gross_bookings_dba{d}" in wide.columns]
    rp = [f"rooms_pickup_dba{d}" for d in DBA_DAYS if f"rooms_pickup_dba{d}" in wide.columns]
    if gb:
        wide["target_bookings_dba0"] = wide["gross_bookings_dba0"]
        wide["target_bookings_window_sum_0_7"] = wide[gb].sum(axis=1, min_count=1)
    if rp:
        wide["target_rooms_pickup_window_sum_0_7"] = wide[rp].sum(axis=1, min_count=1)
    return wide


def add_missing_flags(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    for pref in ["own_price", "comp_median", "comp_trimmed_mean_20"]:
        for d in DBA_DAYS:
            c = f"{pref}_dba{d}"
            if c in out.columns:
                out[f"{c}_is_missing"] = out[c].isna().astype("int8")
    return out


def add_window_features(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    for pref in ["own_price", "comp_median", "comp_trimmed_mean_20", "gross_bookings", "rooms_pickup"]:
        cols = [f"{pref}_dba{d}" for d in DBA_DAYS if f"{pref}_dba{d}" in out.columns]
        if not cols:
            continue
        out[f"{pref}_mean_0_7"] = out[cols].mean(axis=1)
        out[f"{pref}_std_0_7"] = out[cols].std(axis=1)
        out[f"{pref}_min_0_7"] = out[cols].min(axis=1)
        out[f"{pref}_max_0_7"] = out[cols].max(axis=1)
        c0 = f"{pref}_dba0"
        c7 = f"{pref}_dba7"
        if c0 in out.columns and c7 in out.columns:
            out[f"{pref}_dba0_minus_dba7"] = out[c0] - out[c7]
            out[f"{pref}_slope_7_to_0"] = (out[c0] - out[c7]) / 7.0

    if "own_price_mean_0_7" in out.columns and "comp_median_mean_0_7" in out.columns:
        denom = out["comp_median_mean_0_7"].replace(0, np.nan)
        out["own_to_comp_mean_ratio_0_7"] = out["own_price_mean_0_7"] / denom
        out["own_minus_comp_mean_0_7"] = out["own_price_mean_0_7"] - out["comp_median_mean_0_7"]
    return out


def summarize_hotel(panel: pd.DataFrame, wide: pd.DataFrame, feat: pd.DataFrame) -> Dict[str, float]:
    rows = len(panel)
    comp_cols = [c for c in panel.columns if c.startswith("comp_rate_")]
    comp_miss = float(panel[comp_cols].isna().mean().mean() * 100.0) if comp_cols else np.nan
    gb_cols = [f"gross_bookings_dba{d}" for d in DBA_DAYS if f"gross_bookings_dba{d}" in wide.columns]
    full_pct = float(wide[gb_cols].notna().all(axis=1).mean() * 100.0) if gb_cols else np.nan
    return {
        "hotel_id": str(panel["hotel_id"].iloc[0]) if rows else "unknown",
        "rows_long": int(rows),
        "rows_wide": int(len(wide)),
        "rows_features": int(len(feat)),
        "own_price_missing_long_pct": float(panel["own_price"].isna().mean() * 100.0) if "own_price" in panel.columns else np.nan,
        "comp_raw_missing_long_pct": comp_miss,
        "comp_median_missing_long_pct": float(panel["comp_median"].isna().mean() * 100.0) if "comp_median" in panel.columns else np.nan,
        "full_dba0_7_label_coverage_pct": full_pct,
        "avg_target_bookings_dba0": float(wide["target_bookings_dba0"].mean()) if "target_bookings_dba0" in wide.columns else np.nan,
    }


def write_markdown_summary(paths: Paths, hotels: List[str], summary_df: pd.DataFrame) -> None:
    lines = [
        "# DBA0-7 Interval Pipeline (No Imputation)",
        "",
        "This pipeline reconstructs day-level prices from `created_at`/`updated_at` validity intervals,",
        "then assembles DBA0..7 demand rows and feature datasets without imputation.",
        "",
        "## Hotels",
        f"- Processed hotels: {', '.join(hotels)}",
        "",
        "## Outputs",
        "- `output/01_long/*_long_dba0_7_interval.csv`",
        "- `output/02_preprocessed/*_wide_dba0_7_interval.csv`",
        "- `output/03_features/*_features_dba0_7_interval.csv`",
        "- `output/06_reporting/preprocessing_summary_hotelwise.csv`",
        "",
        "## Notes",
        "- Own price uses minimum price across room types each day (BAR proxy).",
        "- Competitor daily values are expanded from intervals and kept per competitor before aggregation.",
        "- No statistical imputation is applied in this pipeline.",
        "",
        "## Portfolio Snapshot",
    ]
    if not summary_df.empty:
        agg = {
            "rows_long": int(summary_df["rows_long"].sum()),
            "rows_wide": int(summary_df["rows_wide"].sum()),
            "rows_features": int(summary_df["rows_features"].sum()),
            "own_price_missing_long_pct_avg": float(summary_df["own_price_missing_long_pct"].mean()),
            "comp_median_missing_long_pct_avg": float(summary_df["comp_median_missing_long_pct"].mean()),
        }
        lines.extend(
            [
                f"- Long rows: {agg['rows_long']:,}",
                f"- Wide rows: {agg['rows_wide']:,}",
                f"- Feature rows: {agg['rows_features']:,}",
                f"- Avg own-price missing in long panel: {agg['own_price_missing_long_pct_avg']:.2f}%",
                f"- Avg comp-median missing in long panel: {agg['comp_median_missing_long_pct_avg']:.2f}%",
            ]
        )
    (paths.out_rep / "pipeline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = build_paths()
    project_root = Path(__file__).resolve().parents[3]

    db_config = load_db_config(project_root)
    conn = get_db_connection(db_config)
    try:
        hotels, comp_map = get_active_hotels_and_comp_map(conn, TARGET_HOTELS)
        print(f"[INFO] Hotels: {hotels}")

        long_frames: List[pd.DataFrame] = []
        wide_frames: List[pd.DataFrame] = []
        feat_frames: List[pd.DataFrame] = []
        summary_rows: List[Dict[str, float]] = []

        for hotel in hotels:
            print(f"[INFO] Processing {hotel}")
            demand = extract_demand_daily(conn, hotel, DECISION_DATE_START)
            if demand.empty:
                print(f"[WARN] {hotel}: no demand rows; skipping.")
                continue
            own = extract_own_prices_interval_daily(conn, hotel, PRICE_CREATED_AT_START)
            comps = extract_comp_prices_interval_daily(conn, comp_map.get(hotel, []), PRICE_CREATED_AT_START)
            inv = extract_inventory(conn, hotel)

            panel = build_hotel_long_panel(
                hotel, demand, own, comps, inv, DECISION_DATE_START
            )
            if panel.empty:
                print(f"[WARN] {hotel}: empty panel after assembly; skipping.")
                continue

            wide = add_missing_flags(pivot_wide(panel))
            feat = add_window_features(wide.copy())

            panel.to_csv(paths.out_long / f"{hotel}_long_dba0_7_interval.csv", index=False)
            wide.to_csv(paths.out_prep / f"{hotel}_wide_dba0_7_interval.csv", index=False)
            feat.to_csv(paths.out_feat / f"{hotel}_features_dba0_7_interval.csv", index=False)

            long_frames.append(panel)
            wide_frames.append(wide)
            feat_frames.append(feat)
            summary_rows.append(summarize_hotel(panel, wide, feat))

        if not long_frames:
            raise RuntimeError("No hotel data was processed successfully.")

        all_long = pd.concat(long_frames, ignore_index=True)
        all_wide = pd.concat(wide_frames, ignore_index=True)
        all_feat = pd.concat(feat_frames, ignore_index=True)
        summary = pd.DataFrame(summary_rows).sort_values("hotel_id")

        all_long.to_csv(paths.out_long / "all_hotels_long_dba0_7_interval.csv", index=False)
        all_wide.to_csv(paths.out_prep / "all_hotels_wide_dba0_7_interval.csv", index=False)
        all_feat.to_csv(paths.out_feat / "all_hotels_features_dba0_7_interval.csv", index=False)
        summary.to_csv(paths.out_rep / "preprocessing_summary_hotelwise.csv", index=False)
        write_markdown_summary(paths, hotels, summary)

        print("[OK] Interval pipeline completed.")
        print(f"[OK] Long: {paths.out_long}")
        print(f"[OK] Preprocessed: {paths.out_prep}")
        print(f"[OK] Features: {paths.out_feat}")
        print(f"[OK] Reporting: {paths.out_rep}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
