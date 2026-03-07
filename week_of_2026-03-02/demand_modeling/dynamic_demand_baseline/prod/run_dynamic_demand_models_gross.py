from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

try:
    import xgboost as xgb
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False

TARGET_PRIMARY = "target_gross_rooms_pickup"
TARGET_SECONDARY = "target_gross_bookings"
HOTEL_COL = "hotel_id"
STAY_COL = "stay_date"
DBA_COL = "dba"
SPLIT_SCHEME = "grouped_random_date_cv"

MODELING_STAY_START = pd.Timestamp("2025-07-01")
MODELING_STAY_END = pd.Timestamp("2026-02-28")
REALIZED_CUTOFF = pd.Timestamp("2026-03-06")

SEEDS = [42, 52, 62]
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3

MIN_OUTER_TRAIN_ROWS = 140
MIN_OUTER_TEST_ROWS = 28
MIN_INNER_TRAIN_ROWS = 80
MIN_INNER_TEST_ROWS = 20

COMP_LOG_PREFIX = "log_comp_price_raw__"
GLM_MIN_COMP = 3
GLM_MAX_COMP = 5
GLM_COMP_MIN_COVERAGE = 0.55
GLM_ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

XGB_PARAMS = {
    "objective": "count:poisson",
    "eval_metric": "mae",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "n_jobs": 4,
}


@dataclass
class Paths:
    base: Path
    in_dir: Path
    out_dir: Path


def build_paths() -> Paths:
    base = Path(__file__).resolve().parent
    in_dir = base / "output" / "02_model_input"
    out_dir = base / "output" / "07_model_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return Paths(base=base, in_dir=in_dir, out_dir=out_dir)


def parse_hotels_arg(raw: str):
    token = str(raw or "").strip()
    if not token or token.upper() in {"ALL", "*"}:
        return []
    return sorted({h.strip() for h in token.split(",") if h.strip()})


def safe_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    exclude = {HOTEL_COL, STAY_COL, "decision_date", "own_price_selected_type", "own_price_selection_reason"}
    for c in out.columns:
        if c in exclude:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def load_data(paths: Paths) -> pd.DataFrame:
    full_path = paths.in_dir / "all_hotels_model_input_dynamic_baseline.csv"
    if not full_path.exists():
        raise RuntimeError(f"Missing model-input file: {full_path}")
    df = pd.read_csv(full_path)
    for c in [STAY_COL, "decision_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df = safe_to_numeric(df)
    if TARGET_PRIMARY not in df.columns:
        raise RuntimeError(f"Missing target column: {TARGET_PRIMARY}")
    if TARGET_SECONDARY not in df.columns:
        df[TARGET_SECONDARY] = np.nan
    df = df[
        (df[STAY_COL] >= MODELING_STAY_START)
        & (df[STAY_COL] <= MODELING_STAY_END)
        & df[TARGET_PRIMARY].notna()
        & (df[TARGET_PRIMARY] >= 0)
    ].copy()
    if df.empty:
        raise RuntimeError("No rows left after date and target filtering.")
    return df


def grouped_random_stay_folds(df: pd.DataFrame, seed: int, n_splits: int, min_train: int, min_test: int):
    dates = np.array(sorted(pd.Series(df[STAY_COL].dropna().unique()).tolist()))
    if len(dates) < n_splits:
        return []
    rng = np.random.RandomState(seed)
    rng.shuffle(dates)
    chunks = np.array_split(dates, n_splits)
    folds = []
    for fold, test_dates in enumerate(chunks, start=1):
        if len(test_dates) == 0:
            continue
        is_test = df[STAY_COL].isin(test_dates)
        tr = df.loc[~is_test].copy()
        te = df.loc[is_test].copy()
        if len(tr) < min_train or len(te) < min_test:
            continue
        folds.append((fold, tr, te))
    return folds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    denom = float(np.sum(y_true))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "poisson_deviance": float(mean_poisson_deviance(y_true, y_pred)),
        "wape": float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan,
        "bias": float(np.mean(y_pred - y_true)),
    }


def estimate_nb_dispersion(y: pd.Series) -> float:
    vals = pd.to_numeric(y, errors="coerce").dropna().astype(float)
    if len(vals) < 2:
        return 1.0
    mu = float(vals.mean())
    var = float(vals.var(ddof=1))
    if mu <= 0:
        return 1.0
    alpha = (var - mu) / max(mu * mu, 1e-9)
    return float(np.clip(alpha, 1e-6, 10.0))


def fit_predict_dba_mean(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> np.ndarray:
    train_y = pd.to_numeric(train_df[target_col], errors="coerce")
    global_mean = float(train_y.mean()) if train_y.notna().any() else 0.0
    if DBA_COL not in train_df.columns or DBA_COL not in test_df.columns:
        return np.full(len(test_df), global_mean, dtype=float)
    by_dba = train_df.groupby(DBA_COL)[target_col].mean()
    pred = test_df[DBA_COL].map(by_dba).fillna(global_mean)
    return np.asarray(pred.values, dtype=float)


def get_log_comp_cols(df: pd.DataFrame):
    return sorted([c for c in df.columns if c.startswith(COMP_LOG_PREFIX)])


def dedup_keep_order(cols):
    out, seen = [], set()
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def available_features(train_df: pd.DataFrame, test_df: pd.DataFrame, cols):
    keep = []
    for c in cols:
        if c not in train_df.columns or c not in test_df.columns:
            continue
        tr = pd.to_numeric(train_df[c], errors="coerce")
        if tr.notna().sum() == 0:
            continue
        if tr.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return keep


def select_glm_comp_cols(train_df: pd.DataFrame, comp_cols):
    if not comp_cols:
        return []
    cov = pd.Series({c: float(train_df[c].notna().mean()) for c in comp_cols}).sort_values(ascending=False)
    eligible = cov[cov >= GLM_COMP_MIN_COVERAGE].index.tolist()
    if len(eligible) >= GLM_MIN_COMP:
        return eligible[:GLM_MAX_COMP]
    return cov.index.tolist()[:GLM_MAX_COMP]


def build_glm_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    season = ["sin_woy", "cos_woy", "sin_month", "cos_month", "sin_dow", "cos_dow"]
    dba_dummies = [f"dba_{i}" for i in range(1, 8)]
    comp_all = [c for c in get_log_comp_cols(train_df) if c in test_df.columns]
    comp_selected = select_glm_comp_cols(train_df, comp_all)
    features = dedup_keep_order(["log_own_price_selected", *dba_dummies, *season, *comp_selected])
    features = [c for c in features if c in train_df.columns and c in test_df.columns]
    return features, comp_selected


def build_xgb_features(train_df: pd.DataFrame, test_df: pd.DataFrame, include_missing_flags: bool):
    season = ["sin_woy", "cos_woy", "sin_month", "cos_month", "sin_dow", "cos_dow"]
    comp_cols = [c for c in get_log_comp_cols(train_df) if c in test_df.columns]
    comp_cols = [c for c in comp_cols if pd.to_numeric(train_df[c], errors="coerce").notna().sum() > 0]
    cols = ["log_own_price_selected", "dba", "dba_norm_7", *season, "comp_count_available", *comp_cols]
    if include_missing_flags:
        cols.append("own_price_selected_missing")
        for lc in comp_cols:
            raw = lc.replace("log_", "", 1)
            cols.append(f"{raw}_missing")
    cols = dedup_keep_order(cols)
    cols = available_features(train_df, test_df, cols)
    kept_comp = [c for c in cols if c.startswith(COMP_LOG_PREFIX)]
    return cols, kept_comp


def prepare_glm_frames(train_df: pd.DataFrame, test_df: pd.DataFrame, features):
    tr = train_df.dropna(subset=list(features) + [TARGET_PRIMARY]).copy()
    te = test_df.dropna(subset=list(features) + [TARGET_PRIMARY]).copy()
    return tr, te


def glm_family(name: str, nb_dispersion: float | None):
    if name == "poisson":
        return sm.families.Poisson()
    if name == "negbin":
        return sm.families.NegativeBinomial(alpha=float(nb_dispersion if nb_dispersion is not None else 1.0))
    raise ValueError(f"Unsupported family: {name}")


def fit_glm_regularized_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, features, family_name: str, alpha_penalty: float, nb_dispersion: float | None):
    X_train = sm.add_constant(train_df[list(features)], has_constant="add")
    y_train = train_df[TARGET_PRIMARY].astype(float)
    X_test = sm.add_constant(test_df[list(features)], has_constant="add")
    model = sm.GLM(y_train, X_train, family=glm_family(family_name, nb_dispersion))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg_res = model.fit_regularized(alpha=float(alpha_penalty), L1_wt=0.0, maxiter=400)
    pred = np.asarray(reg_res.predict(X_test), dtype=float)
    return pred


def fit_glm_inference(train_df: pd.DataFrame, features, family_name: str, nb_dispersion: float | None):
    X_train = sm.add_constant(train_df[list(features)], has_constant="add")
    y_train = train_df[TARGET_PRIMARY].astype(float)
    model = sm.GLM(y_train, X_train, family=glm_family(family_name, nb_dispersion))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(maxiter=250, disp=0)


def select_glm_alpha(train_df: pd.DataFrame, features, family_name: str, seed: int):
    folds = grouped_random_stay_folds(
        train_df, seed=seed, n_splits=INNER_N_SPLITS, min_train=MIN_INNER_TRAIN_ROWS, min_test=MIN_INNER_TEST_ROWS
    )
    if not folds:
        return float(GLM_ALPHA_GRID[0]), np.nan
    best_alpha, best_mae = float(GLM_ALPHA_GRID[0]), np.inf
    for alpha in GLM_ALPHA_GRID:
        maes = []
        for fold, tr, te in folds:
            tr_cc, te_cc = prepare_glm_frames(tr, te, features)
            if len(tr_cc) < MIN_INNER_TRAIN_ROWS or len(te_cc) < MIN_INNER_TEST_ROWS:
                continue
            nb_disp = estimate_nb_dispersion(tr_cc[TARGET_PRIMARY]) if family_name == "negbin" else None
            try:
                pred = fit_glm_regularized_predict(
                    tr_cc, te_cc, features, family_name=family_name, alpha_penalty=float(alpha), nb_dispersion=nb_disp
                )
                maes.append(float(mean_absolute_error(te_cc[TARGET_PRIMARY].values, np.clip(pred, 1e-9, None))))
            except Exception:
                continue
        if maes:
            avg = float(np.mean(maes))
            if avg < best_mae:
                best_mae = avg
                best_alpha = float(alpha)
    return best_alpha, (best_mae if best_mae < np.inf else np.nan)


def fit_xgb_predict_with_artifacts(train_df: pd.DataFrame, test_df: pd.DataFrame, features, seed: int):
    if not HAS_XGB:
        raise RuntimeError("xgboost is not installed")
    params = dict(XGB_PARAMS)
    params["random_state"] = int(seed)
    model = XGBRegressor(**params)
    X_train = train_df[list(features)]
    y_train = train_df[TARGET_PRIMARY].astype(float)
    X_test = test_df[list(features)]
    model.fit(X_train, y_train)
    pred = np.asarray(model.predict(X_test), dtype=float)

    booster = model.get_booster()
    gain_raw = booster.get_score(importance_type="gain")
    gain_map = {}
    for i, f in enumerate(features):
        gain_map[f] = float(gain_raw.get(f, gain_raw.get(f"f{i}", 0.0)))

    shap_map = {f: np.nan for f in features}
    try:
        dtest = xgb.DMatrix(X_test, feature_names=list(features))
        contrib = booster.predict(dtest, pred_contribs=True)
        if contrib.ndim == 2 and contrib.shape[1] == len(features) + 1:
            mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
            shap_map = {f: float(v) for f, v in zip(features, mean_abs)}
    except Exception:
        pass
    return pred, gain_map, shap_map


def eval_xgb_features_cv(df: pd.DataFrame, features, seed: int):
    if not HAS_XGB:
        return np.nan
    folds = grouped_random_stay_folds(
        df, seed=seed, n_splits=3, min_train=MIN_INNER_TRAIN_ROWS, min_test=MIN_INNER_TEST_ROWS
    )
    if not folds:
        return np.nan
    maes = []
    for fold, tr, te in folds:
        tr_x = tr.dropna(subset=[TARGET_PRIMARY]).copy()
        te_x = te.dropna(subset=[TARGET_PRIMARY]).copy()
        if len(tr_x) < MIN_INNER_TRAIN_ROWS or len(te_x) < MIN_INNER_TEST_ROWS:
            continue
        try:
            pred, _, _ = fit_xgb_predict_with_artifacts(tr_x, te_x, features, seed=seed * 100 + fold)
            maes.append(float(mean_absolute_error(te_x[TARGET_PRIMARY].values, np.clip(pred, 1e-9, None))))
        except Exception:
            continue
    return float(np.mean(maes)) if maes else np.nan


def choose_xgb_feature_set(hotel_df: pd.DataFrame, seed: int):
    base_feats, base_comp = build_xgb_features(hotel_df, hotel_df, include_missing_flags=False)
    if not HAS_XGB:
        return base_feats, base_comp, False, np.nan, np.nan
    flag_feats, flag_comp = build_xgb_features(hotel_df, hotel_df, include_missing_flags=True)
    if set(base_feats) == set(flag_feats):
        return base_feats, base_comp, False, np.nan, np.nan
    mae_base = eval_xgb_features_cv(hotel_df, base_feats, seed=seed)
    mae_flag = eval_xgb_features_cv(hotel_df, flag_feats, seed=seed)
    use_flags = bool(mae_flag == mae_flag and (mae_base != mae_base or mae_flag + 1e-6 < mae_base))
    if use_flags:
        return flag_feats, flag_comp, True, mae_base, mae_flag
    return base_feats, base_comp, False, mae_base, mae_flag


def deviance_r2(fit_obj):
    dev = getattr(fit_obj, "deviance", np.nan)
    null_dev = getattr(fit_obj, "null_deviance", np.nan)
    if dev == dev and null_dev == null_dev and float(null_dev) > 0.0:
        return float(1.0 - float(dev) / float(null_dev))
    return np.nan


def mcfadden_r2(fit_obj):
    llf = getattr(fit_obj, "llf", np.nan)
    llnull = getattr(fit_obj, "llnull", np.nan)
    if llf == llf and llnull == llnull and float(llnull) != 0.0:
        return float(1.0 - float(llf) / float(llnull))
    return np.nan


def sign_consistency(beta: pd.Series):
    s = np.sign(pd.to_numeric(beta, errors="coerce").dropna())
    s = s[s != 0]
    if len(s) == 0:
        return np.nan
    return float(max((s > 0).mean(), (s < 0).mean()))


def main(hotel_filter: list[str] | None = None) -> None:
    paths = build_paths()
    df = load_data(paths)
    if hotel_filter:
        df = df[df[HOTEL_COL].isin(hotel_filter)].copy()
        if df.empty:
            raise RuntimeError(f"No rows left after hotel filter: {hotel_filter}")
    hotels = sorted(df[HOTEL_COL].dropna().unique().tolist())

    holdout_rows = []
    secondary_rows = []
    glm_coef_fold_rows = []
    glm_diag_rows = []
    gain_fold_rows = []
    shap_fold_rows = []
    config_rows = []

    for hotel in hotels:
        hdf = df[df[HOTEL_COL] == hotel].copy().sort_values([STAY_COL, "decision_date"])
        if len(hdf) < (MIN_OUTER_TRAIN_ROWS + MIN_OUTER_TEST_ROWS):
            continue

        own_type = (
            str(hdf["own_price_selected_type"].dropna().iloc[0])
            if "own_price_selected_type" in hdf.columns and hdf["own_price_selected_type"].notna().any()
            else "unknown"
        )
        if "own_price_selected" in hdf.columns:
            hdf_ready = hdf[hdf["own_price_selected"].notna()].copy()
        else:
            hdf_ready = hdf.copy()

        xgb_features_full, xgb_comp_cols_full, xgb_use_missing_flags_full, xgb_mae_base_full, xgb_mae_flags_full = choose_xgb_feature_set(
            hdf, seed=SEEDS[0]
        )
        if len(hdf_ready) >= (MIN_OUTER_TRAIN_ROWS + MIN_OUTER_TEST_ROWS):
            xgb_features_ready, xgb_comp_cols_ready, xgb_use_missing_flags_ready, xgb_mae_base_ready, xgb_mae_flags_ready = choose_xgb_feature_set(
                hdf_ready, seed=SEEDS[0]
            )
        else:
            xgb_features_ready, xgb_comp_cols_ready, xgb_use_missing_flags_ready = [], [], False
            xgb_mae_base_ready, xgb_mae_flags_ready = np.nan, np.nan

        for seed in SEEDS:
            folds = grouped_random_stay_folds(
                hdf, seed=seed, n_splits=OUTER_N_SPLITS, min_train=MIN_OUTER_TRAIN_ROWS, min_test=MIN_OUTER_TEST_ROWS
            )
            if not folds:
                continue
            for fold, tr, te in folds:
                split_iter = int(seed * 100 + fold)

                secondary_rows.append(
                    {
                        "hotel_id": hotel,
                        "split_scheme": SPLIT_SCHEME,
                        "seed": int(seed),
                        "fold": int(fold),
                        "split_iter": split_iter,
                        "train_rows": int(len(tr)),
                        "test_rows": int(len(te)),
                        "train_primary_mean": float(tr[TARGET_PRIMARY].mean()),
                        "test_primary_mean": float(te[TARGET_PRIMARY].mean()),
                        "train_secondary_mean": float(pd.to_numeric(tr[TARGET_SECONDARY], errors="coerce").mean()),
                        "test_secondary_mean": float(pd.to_numeric(te[TARGET_SECONDARY], errors="coerce").mean()),
                    }
                )

                tr_b = tr.dropna(subset=[TARGET_PRIMARY]).copy()
                te_b = te.dropna(subset=[TARGET_PRIMARY]).copy()
                if len(tr_b) >= MIN_OUTER_TRAIN_ROWS and len(te_b) >= MIN_OUTER_TEST_ROWS:
                    pred = fit_predict_dba_mean(tr_b, te_b, TARGET_PRIMARY)
                    m = compute_metrics(te_b[TARGET_PRIMARY].values, pred)
                    holdout_rows.append(
                        {
                            "hotel_id": hotel,
                            "split_scheme": SPLIT_SCHEME,
                            "seed": int(seed),
                            "fold": int(fold),
                            "split_iter": split_iter,
                            "model": "Baseline_DBA_Mean",
                            "dataset_used": "full_input",
                            "target": TARGET_PRIMARY,
                            "train_rows": int(len(tr_b)),
                            "test_rows": int(len(te_b)),
                            "own_price_selected_type": own_type,
                            **m,
                        }
                    )

                tr_s = tr.dropna(subset=[TARGET_SECONDARY]).copy()
                te_s = te.dropna(subset=[TARGET_SECONDARY]).copy()
                if len(tr_s) >= MIN_OUTER_TRAIN_ROWS and len(te_s) >= MIN_OUTER_TEST_ROWS:
                    pred_s = fit_predict_dba_mean(tr_s, te_s, TARGET_SECONDARY)
                    m_s = compute_metrics(te_s[TARGET_SECONDARY].values, pred_s)
                    secondary_rows.append(
                        {
                            "hotel_id": hotel,
                            "split_scheme": SPLIT_SCHEME,
                            "seed": int(seed),
                            "fold": int(fold),
                            "split_iter": split_iter,
                            "secondary_model": "Baseline_DBA_Mean",
                            "secondary_target": TARGET_SECONDARY,
                            "secondary_mae": float(m_s["mae"]),
                            "secondary_wape": float(m_s["wape"]),
                            "secondary_bias": float(m_s["bias"]),
                            "secondary_poisson_deviance": float(m_s["poisson_deviance"]),
                        }
                    )

                glm_features, glm_comp_cols = build_glm_features(tr, te)
                tr_glm, te_glm = prepare_glm_frames(tr, te, glm_features)

                poisson_alpha = np.nan
                negbin_alpha = np.nan
                negbin_dispersion = np.nan

                if len(glm_features) >= 5 and len(tr_glm) >= MIN_OUTER_TRAIN_ROWS and len(te_glm) >= MIN_OUTER_TEST_ROWS:
                    try:
                        poisson_alpha, _ = select_glm_alpha(tr_glm, glm_features, family_name="poisson", seed=seed + fold)
                        pred_po = fit_glm_regularized_predict(
                            tr_glm,
                            te_glm,
                            glm_features,
                            family_name="poisson",
                            alpha_penalty=float(poisson_alpha),
                            nb_dispersion=None,
                        )
                        m_po = compute_metrics(te_glm[TARGET_PRIMARY].values, pred_po)
                        holdout_rows.append(
                            {
                                "hotel_id": hotel,
                                "split_scheme": SPLIT_SCHEME,
                                "seed": int(seed),
                                "fold": int(fold),
                                "split_iter": split_iter,
                                "model": "Poisson_GLM_RegLogPrice",
                                "dataset_used": "complete_case",
                                "target": TARGET_PRIMARY,
                                "train_rows": int(len(tr_glm)),
                                "test_rows": int(len(te_glm)),
                                "own_price_selected_type": own_type,
                                **m_po,
                            }
                        )
                        fit_po = fit_glm_inference(tr_glm, glm_features, family_name="poisson", nb_dispersion=None)
                        params = pd.Series(getattr(fit_po, "params", pd.Series(dtype=float)))
                        pvals = pd.Series(getattr(fit_po, "pvalues", pd.Series(dtype=float)))
                        ci = fit_po.conf_int()
                        for term, beta in params.items():
                            glm_coef_fold_rows.append(
                                {
                                    "hotel_id": hotel,
                                    "split_scheme": SPLIT_SCHEME,
                                    "seed": int(seed),
                                    "fold": int(fold),
                                    "split_iter": split_iter,
                                    "model": "Poisson_GLM_RegLogPrice",
                                    "term": str(term),
                                    "beta": float(beta),
                                    "pvalue": float(pvals.get(term, np.nan)),
                                    "ci_low": float(ci.loc[term].iloc[0]) if term in ci.index else np.nan,
                                    "ci_high": float(ci.loc[term].iloc[1]) if term in ci.index else np.nan,
                                }
                            )
                        glm_diag_rows.append(
                            {
                                "hotel_id": hotel,
                                "split_scheme": SPLIT_SCHEME,
                                "seed": int(seed),
                                "fold": int(fold),
                                "split_iter": split_iter,
                                "model": "Poisson_GLM_RegLogPrice",
                                "train_rows": int(len(tr_glm)),
                                "test_rows": int(len(te_glm)),
                                "alpha_selected": float(poisson_alpha),
                                "nb_dispersion_used": np.nan,
                                "deviance": float(getattr(fit_po, "deviance", np.nan)),
                                "null_deviance": float(getattr(fit_po, "null_deviance", np.nan)),
                                "deviance_r2": deviance_r2(fit_po),
                                "mcfadden_r2": mcfadden_r2(fit_po),
                                "aic": float(getattr(fit_po, "aic", np.nan)),
                                "bic": float(getattr(fit_po, "bic", np.nan)) if hasattr(fit_po, "bic") else np.nan,
                                "nobs": int(getattr(fit_po, "nobs", 0)),
                            }
                        )
                    except Exception:
                        pass

                    try:
                        negbin_dispersion = estimate_nb_dispersion(tr_glm[TARGET_PRIMARY])
                        negbin_alpha, _ = select_glm_alpha(
                            tr_glm, glm_features, family_name="negbin", seed=seed + 100 + fold
                        )
                        pred_nb = fit_glm_regularized_predict(
                            tr_glm,
                            te_glm,
                            glm_features,
                            family_name="negbin",
                            alpha_penalty=float(negbin_alpha),
                            nb_dispersion=float(negbin_dispersion),
                        )
                        m_nb = compute_metrics(te_glm[TARGET_PRIMARY].values, pred_nb)
                        holdout_rows.append(
                            {
                                "hotel_id": hotel,
                                "split_scheme": SPLIT_SCHEME,
                                "seed": int(seed),
                                "fold": int(fold),
                                "split_iter": split_iter,
                                "model": "NegBin_GLM_RegLogPrice",
                                "dataset_used": "complete_case",
                                "target": TARGET_PRIMARY,
                                "train_rows": int(len(tr_glm)),
                                "test_rows": int(len(te_glm)),
                                "own_price_selected_type": own_type,
                                **m_nb,
                            }
                        )
                        fit_nb = fit_glm_inference(
                            tr_glm, glm_features, family_name="negbin", nb_dispersion=float(negbin_dispersion)
                        )
                        params_nb = pd.Series(getattr(fit_nb, "params", pd.Series(dtype=float)))
                        pvals_nb = pd.Series(getattr(fit_nb, "pvalues", pd.Series(dtype=float)))
                        ci_nb = fit_nb.conf_int()
                        for term, beta in params_nb.items():
                            glm_coef_fold_rows.append(
                                {
                                    "hotel_id": hotel,
                                    "split_scheme": SPLIT_SCHEME,
                                    "seed": int(seed),
                                    "fold": int(fold),
                                    "split_iter": split_iter,
                                    "model": "NegBin_GLM_RegLogPrice",
                                    "term": str(term),
                                    "beta": float(beta),
                                    "pvalue": float(pvals_nb.get(term, np.nan)),
                                    "ci_low": float(ci_nb.loc[term].iloc[0]) if term in ci_nb.index else np.nan,
                                    "ci_high": float(ci_nb.loc[term].iloc[1]) if term in ci_nb.index else np.nan,
                                }
                            )
                        glm_diag_rows.append(
                            {
                                "hotel_id": hotel,
                                "split_scheme": SPLIT_SCHEME,
                                "seed": int(seed),
                                "fold": int(fold),
                                "split_iter": split_iter,
                                "model": "NegBin_GLM_RegLogPrice",
                                "train_rows": int(len(tr_glm)),
                                "test_rows": int(len(te_glm)),
                                "alpha_selected": float(negbin_alpha),
                                "nb_dispersion_used": float(negbin_dispersion),
                                "deviance": float(getattr(fit_nb, "deviance", np.nan)),
                                "null_deviance": float(getattr(fit_nb, "null_deviance", np.nan)),
                                "deviance_r2": deviance_r2(fit_nb),
                                "mcfadden_r2": mcfadden_r2(fit_nb),
                                "aic": float(getattr(fit_nb, "aic", np.nan)),
                                "bic": float(getattr(fit_nb, "bic", np.nan)) if hasattr(fit_nb, "bic") else np.nan,
                                "nobs": int(getattr(fit_nb, "nobs", 0)),
                            }
                        )
                    except Exception:
                        pass

                xgb_run_specs = []
                if HAS_XGB and len(xgb_features_full) >= 4:
                    tr_x_full = tr.dropna(subset=[TARGET_PRIMARY]).copy()
                    te_x_full = te.dropna(subset=[TARGET_PRIMARY]).copy()
                    xgb_run_specs.append(("full_input", tr_x_full, te_x_full, xgb_features_full))
                if HAS_XGB and len(xgb_features_ready) >= 4 and "own_price_selected" in tr.columns and "own_price_selected" in te.columns:
                    tr_x_ready = tr[tr["own_price_selected"].notna()].dropna(subset=[TARGET_PRIMARY]).copy()
                    te_x_ready = te[te["own_price_selected"].notna()].dropna(subset=[TARGET_PRIMARY]).copy()
                    xgb_run_specs.append(("model_ready", tr_x_ready, te_x_ready, xgb_features_ready))

                for dataset_used_xgb, tr_x, te_x, xgb_features_use in xgb_run_specs:
                    if len(tr_x) < MIN_OUTER_TRAIN_ROWS or len(te_x) < MIN_OUTER_TEST_ROWS:
                        continue
                    try:
                        pred_xgb, gain_map, shap_map = fit_xgb_predict_with_artifacts(
                            tr_x, te_x, xgb_features_use, seed=split_iter
                        )
                        m_xgb = compute_metrics(te_x[TARGET_PRIMARY].values, pred_xgb)
                        holdout_rows.append(
                            {
                                "hotel_id": hotel,
                                "split_scheme": SPLIT_SCHEME,
                                "seed": int(seed),
                                "fold": int(fold),
                                "split_iter": split_iter,
                                "model": "XGBoost_Poisson_DBA_Numeric",
                                "dataset_used": dataset_used_xgb,
                                "target": TARGET_PRIMARY,
                                "train_rows": int(len(tr_x)),
                                "test_rows": int(len(te_x)),
                                "own_price_selected_type": own_type,
                                **m_xgb,
                            }
                        )
                        total_shap = float(np.nansum(list(shap_map.values())))
                        for f in xgb_features_use:
                            gain_fold_rows.append(
                                {
                                    "hotel_id": hotel,
                                    "split_scheme": SPLIT_SCHEME,
                                    "seed": int(seed),
                                    "fold": int(fold),
                                    "split_iter": split_iter,
                                    "model": "XGBoost_Poisson_DBA_Numeric",
                                    "dataset_used": dataset_used_xgb,
                                    "feature": f,
                                    "gain_importance": float(gain_map.get(f, 0.0)),
                                }
                            )
                            shap_val = float(shap_map.get(f, np.nan))
                            shap_fold_rows.append(
                                {
                                    "hotel_id": hotel,
                                    "split_scheme": SPLIT_SCHEME,
                                    "seed": int(seed),
                                    "fold": int(fold),
                                    "split_iter": split_iter,
                                    "model": "XGBoost_Poisson_DBA_Numeric",
                                    "dataset_used": dataset_used_xgb,
                                    "feature": f,
                                    "mean_abs_shap": shap_val,
                                    "shap_share": (shap_val / total_shap)
                                    if total_shap > 0 and shap_val == shap_val
                                    else np.nan,
                                }
                            )
                    except Exception:
                        pass

                config_rows.append(
                    {
                        "hotel_id": hotel,
                        "split_scheme": SPLIT_SCHEME,
                        "seed": int(seed),
                        "fold": int(fold),
                        "split_iter": split_iter,
                        "target_primary": TARGET_PRIMARY,
                        "target_secondary": TARGET_SECONDARY,
                        "glm_feature_count": int(len(glm_features)),
                        "glm_comp_feature_count": int(len(glm_comp_cols)),
                        "glm_comp_features": "|".join(glm_comp_cols),
                        "poisson_alpha_selected": float(poisson_alpha) if poisson_alpha == poisson_alpha else np.nan,
                        "negbin_alpha_selected": float(negbin_alpha) if negbin_alpha == negbin_alpha else np.nan,
                        "negbin_dispersion": float(negbin_dispersion) if negbin_dispersion == negbin_dispersion else np.nan,
                        "xgb_full_feature_count": int(len(xgb_features_full)),
                        "xgb_full_comp_feature_count": int(len(xgb_comp_cols_full)),
                        "xgb_full_missing_flags_used": int(bool(xgb_use_missing_flags_full)),
                        "xgb_full_feature_selection_mae_base": float(xgb_mae_base_full)
                        if xgb_mae_base_full == xgb_mae_base_full
                        else np.nan,
                        "xgb_full_feature_selection_mae_with_flags": float(xgb_mae_flags_full)
                        if xgb_mae_flags_full == xgb_mae_flags_full
                        else np.nan,
                        "xgb_model_ready_feature_count": int(len(xgb_features_ready)),
                        "xgb_model_ready_comp_feature_count": int(len(xgb_comp_cols_ready)),
                        "xgb_model_ready_missing_flags_used": int(bool(xgb_use_missing_flags_ready)),
                        "xgb_model_ready_feature_selection_mae_base": float(xgb_mae_base_ready)
                        if xgb_mae_base_ready == xgb_mae_base_ready
                        else np.nan,
                        "xgb_model_ready_feature_selection_mae_with_flags": float(xgb_mae_flags_ready)
                        if xgb_mae_flags_ready == xgb_mae_flags_ready
                        else np.nan,
                    }
                )

    holdout_df = pd.DataFrame(holdout_rows)
    secondary_df = pd.DataFrame(secondary_rows)
    glm_coef_fold_df = pd.DataFrame(glm_coef_fold_rows)
    glm_diag_df = pd.DataFrame(glm_diag_rows)
    gain_fold_df = pd.DataFrame(gain_fold_rows)
    shap_fold_df = pd.DataFrame(shap_fold_rows)
    config_df = pd.DataFrame(config_rows)

    if holdout_df.empty:
        raise RuntimeError("No model runs completed.")

    iter_rows = []
    for (scheme, split_iter, model, dataset_used), g in holdout_df.groupby(
        ["split_scheme", "split_iter", "model", "dataset_used"]
    ):
        w = g["test_rows"] / g["test_rows"].sum()
        iter_rows.append(
            {
                "split_scheme": scheme,
                "split_iter": int(split_iter),
                "seed": int(g["seed"].iloc[0]),
                "fold": int(g["fold"].iloc[0]),
                "model": model,
                "dataset_used": dataset_used,
                "hotels_count": int(g["hotel_id"].nunique()),
                "total_test_rows": int(g["test_rows"].sum()),
                "weighted_mae": float((g["mae"] * w).sum()),
                "weighted_rmse": float((g["rmse"] * w).sum()),
                "weighted_poisson_deviance": float((g["poisson_deviance"] * w).sum()),
                "weighted_wape": float((g["wape"] * w).sum()),
                "weighted_bias": float((g["bias"] * w).sum()),
            }
        )
    portfolio_iter_df = pd.DataFrame(iter_rows)

    portfolio_df = (
        portfolio_iter_df.groupby(["split_scheme", "model", "dataset_used"], as_index=False)
        .agg(
            iters=("split_iter", "nunique"),
            hotels_count=("hotels_count", "mean"),
            total_test_rows=("total_test_rows", "mean"),
            weighted_mae=("weighted_mae", "mean"),
            weighted_mae_sd=("weighted_mae", "std"),
            weighted_rmse=("weighted_rmse", "mean"),
            weighted_rmse_sd=("weighted_rmse", "std"),
            weighted_poisson_deviance=("weighted_poisson_deviance", "mean"),
            weighted_poisson_deviance_sd=("weighted_poisson_deviance", "std"),
            weighted_wape=("weighted_wape", "mean"),
            weighted_wape_sd=("weighted_wape", "std"),
            weighted_bias=("weighted_bias", "mean"),
            weighted_bias_sd=("weighted_bias", "std"),
        )
        .sort_values(["weighted_mae", "model"])
        .reset_index(drop=True)
    )
    for c in [
        "weighted_mae_sd",
        "weighted_rmse_sd",
        "weighted_poisson_deviance_sd",
        "weighted_wape_sd",
        "weighted_bias_sd",
    ]:
        portfolio_df[c] = portfolio_df[c].fillna(0.0)
    portfolio_df["iters"] = portfolio_df["iters"].astype(int)
    portfolio_df["hotels_count"] = portfolio_df["hotels_count"].round().astype(int)
    portfolio_df["total_test_rows"] = portfolio_df["total_test_rows"].round().astype(int)

    best_candidates = holdout_df[holdout_df["model"] != "Baseline_DBA_Mean"].copy()
    best_summary = (
        best_candidates.groupby(["split_scheme", "hotel_id", "model", "dataset_used"], as_index=False)
        .agg(
            iters=("split_iter", "nunique"),
            mae=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse=("rmse", "mean"),
            poisson_deviance=("poisson_deviance", "mean"),
            wape=("wape", "mean"),
            bias=("bias", "mean"),
            train_rows=("train_rows", "mean"),
            test_rows=("test_rows", "mean"),
        )
        .sort_values(["split_scheme", "hotel_id", "mae"])
    )
    best_summary["mae_sd"] = best_summary["mae_sd"].fillna(0.0)
    best_summary["iters"] = best_summary["iters"].astype(int)
    best_summary["train_rows"] = best_summary["train_rows"].round().astype(int)
    best_summary["test_rows"] = best_summary["test_rows"].round().astype(int)
    best_by_hotel = best_summary.groupby(["split_scheme", "hotel_id"], as_index=False).first()

    baseline_ref = holdout_df[holdout_df["model"] == "Baseline_DBA_Mean"][
        ["split_scheme", "split_iter", "hotel_id", "dataset_used", "mae", "rmse", "poisson_deviance", "wape"]
    ].rename(
        columns={
            "mae": "baseline_mae",
            "rmse": "baseline_rmse",
            "poisson_deviance": "baseline_poisson_deviance",
            "wape": "baseline_wape",
        }
    )
    model_vs_baseline = holdout_df[holdout_df["model"] != "Baseline_DBA_Mean"].merge(
        baseline_ref, on=["split_scheme", "split_iter", "hotel_id", "dataset_used"], how="left"
    )
    model_vs_baseline["mae_improvement_vs_baseline"] = model_vs_baseline["baseline_mae"] - model_vs_baseline["mae"]
    model_vs_baseline["rmse_improvement_vs_baseline"] = model_vs_baseline["baseline_rmse"] - model_vs_baseline["rmse"]
    model_vs_baseline["poisson_dev_improvement_vs_baseline"] = (
        model_vs_baseline["baseline_poisson_deviance"] - model_vs_baseline["poisson_deviance"]
    )
    model_vs_baseline["wape_improvement_vs_baseline"] = model_vs_baseline["baseline_wape"] - model_vs_baseline["wape"]

    if not gain_fold_df.empty:
        gain_df = (
            gain_fold_df.groupby(["hotel_id", "split_scheme", "model", "dataset_used", "feature"], as_index=False)
            .agg(mean_gain=("gain_importance", "mean"), sd_gain=("gain_importance", "std"), folds=("gain_importance", "size"))
            .sort_values(["hotel_id", "model", "dataset_used", "mean_gain"], ascending=[True, True, True, False])
            .reset_index(drop=True)
        )
        gain_df["sd_gain"] = gain_df["sd_gain"].fillna(0.0)
    else:
        gain_df = pd.DataFrame(
            columns=["hotel_id", "split_scheme", "model", "dataset_used", "feature", "mean_gain", "sd_gain", "folds"]
        )

    if not shap_fold_df.empty:
        shap_df = (
            shap_fold_df.groupby(["hotel_id", "split_scheme", "model", "dataset_used", "feature"], as_index=False)
            .agg(
                mean_abs_shap=("mean_abs_shap", "mean"),
                sd_abs_shap=("mean_abs_shap", "std"),
                shap_share=("shap_share", "mean"),
                sd_shap_share=("shap_share", "std"),
                folds=("mean_abs_shap", "size"),
            )
            .sort_values(["hotel_id", "model", "dataset_used", "mean_abs_shap"], ascending=[True, True, True, False])
            .reset_index(drop=True)
        )
        shap_df["sd_abs_shap"] = shap_df["sd_abs_shap"].fillna(0.0)
        shap_df["sd_shap_share"] = shap_df["sd_shap_share"].fillna(0.0)
    else:
        shap_df = pd.DataFrame(
            columns=[
                "hotel_id",
                "split_scheme",
                "model",
                "dataset_used",
                "feature",
                "mean_abs_shap",
                "sd_abs_shap",
                "shap_share",
                "sd_shap_share",
                "folds",
            ]
        )

    if not shap_df.empty:
        shap_top_overall = (
            shap_df.sort_values(["hotel_id", "model", "dataset_used", "mean_abs_shap"], ascending=[True, True, True, False])
            .groupby(["hotel_id", "model", "dataset_used"], as_index=False, group_keys=False)
            .head(10)
            .reset_index(drop=True)
        )
        price_mask = shap_df["feature"].eq("log_own_price_selected") | shap_df["feature"].astype(str).str.startswith(
            COMP_LOG_PREFIX
        )
        shap_top_price = (
            shap_df[price_mask]
            .sort_values(["hotel_id", "model", "dataset_used", "mean_abs_shap"], ascending=[True, True, True, False])
            .groupby(["hotel_id", "model", "dataset_used"], as_index=False, group_keys=False)
            .head(10)
            .reset_index(drop=True)
        )
    else:
        shap_top_overall = shap_df.copy()
        shap_top_price = shap_df.copy()

    if not glm_coef_fold_df.empty:
        glm_coef_agg = (
            glm_coef_fold_df.groupby(["hotel_id", "split_scheme", "model", "term"], as_index=False)
            .agg(
                beta_mean=("beta", "mean"),
                beta_sd=("beta", "std"),
                pvalue_mean=("pvalue", "mean"),
                pvalue_median=("pvalue", "median"),
                ci_low_mean=("ci_low", "mean"),
                ci_high_mean=("ci_high", "mean"),
                folds=("beta", "size"),
            )
            .reset_index(drop=True)
        )
        glm_coef_agg["beta_sd"] = glm_coef_agg["beta_sd"].fillna(0.0)
        sign_df = (
            glm_coef_fold_df.groupby(["hotel_id", "split_scheme", "model", "term"])["beta"]
            .apply(sign_consistency)
            .reset_index(name="sign_consistency")
        )
        glm_coef_agg = glm_coef_agg.merge(sign_df, on=["hotel_id", "split_scheme", "model", "term"], how="left")
        glm_coef_agg["beta"] = glm_coef_agg["beta_mean"]
        glm_coef_agg["pvalue"] = glm_coef_agg["pvalue_mean"]
        glm_coef_agg["ci_low"] = glm_coef_agg["ci_low_mean"]
        glm_coef_agg["ci_high"] = glm_coef_agg["ci_high_mean"]

        glm_coef_overall_top = (
            glm_coef_agg.assign(abs_beta=lambda x: np.abs(x["beta_mean"]))
            .sort_values(["hotel_id", "model", "abs_beta"], ascending=[True, True, False])
            .groupby(["hotel_id", "model"], as_index=False, group_keys=False)
            .head(10)
            .drop(columns=["abs_beta"])
            .reset_index(drop=True)
        )
        glm_price_mask = glm_coef_agg["term"].eq("log_own_price_selected") | glm_coef_agg["term"].astype(str).str.startswith(
            COMP_LOG_PREFIX
        )
        glm_coef_price_top = (
            glm_coef_agg[glm_price_mask]
            .assign(abs_beta=lambda x: np.abs(x["beta_mean"]))
            .sort_values(["hotel_id", "model", "abs_beta"], ascending=[True, True, False])
            .groupby(["hotel_id", "model"], as_index=False, group_keys=False)
            .head(10)
            .drop(columns=["abs_beta"])
            .reset_index(drop=True)
        )
    else:
        glm_coef_agg = pd.DataFrame(
            columns=[
                "hotel_id",
                "split_scheme",
                "model",
                "term",
                "beta_mean",
                "beta_sd",
                "pvalue_mean",
                "pvalue_median",
                "ci_low_mean",
                "ci_high_mean",
                "folds",
                "sign_consistency",
                "beta",
                "pvalue",
                "ci_low",
                "ci_high",
            ]
        )
        glm_coef_overall_top = glm_coef_agg.copy()
        glm_coef_price_top = glm_coef_agg.copy()

    holdout_path = paths.out_dir / "model_holdout_hotelwise_gross.csv"
    portfolio_path = paths.out_dir / "model_portfolio_comparison_gross.csv"
    portfolio_iter_path = paths.out_dir / "model_portfolio_comparison_by_split_iter_gross.csv"
    best_path = paths.out_dir / "best_model_by_hotel_gross.csv"
    vs_baseline_path = paths.out_dir / "model_vs_dba_mean_baseline_hotelwise_gross.csv"
    secondary_diag_path = paths.out_dir / "secondary_target_diagnostics_gross.csv"
    gain_path = paths.out_dir / "xgb_feature_importance_gain_gross.csv"
    shap_path = paths.out_dir / "xgb_shap_mean_abs_gross.csv"
    shap_fold_path = paths.out_dir / "xgb_shap_foldwise_gross.csv"
    shap_top_price_path = paths.out_dir / "xgb_shap_top10_price_gross.csv"
    shap_top_overall_path = paths.out_dir / "xgb_shap_top10_overall_gross.csv"
    glm_coef_path = paths.out_dir / "glm_coefficients_hotelwise_gross.csv"
    glm_coef_fold_path = paths.out_dir / "glm_coefficients_foldwise_hotelwise_gross.csv"
    glm_diag_path = paths.out_dir / "glm_fit_diagnostics_hotelwise_gross.csv"
    glm_top_price_path = paths.out_dir / "glm_coeff_top10_price_gross.csv"
    glm_top_overall_path = paths.out_dir / "glm_coeff_top10_overall_gross.csv"
    config_path = paths.out_dir / "model_feature_set_config_hotelwise_gross.csv"
    summary_path = paths.out_dir / "model_run_summary_gross.md"

    holdout_df.to_csv(holdout_path, index=False)
    portfolio_df.to_csv(portfolio_path, index=False)
    portfolio_iter_df.to_csv(portfolio_iter_path, index=False)
    best_by_hotel.to_csv(best_path, index=False)
    model_vs_baseline.to_csv(vs_baseline_path, index=False)
    secondary_df.to_csv(secondary_diag_path, index=False)
    gain_df.to_csv(gain_path, index=False)
    shap_df.to_csv(shap_path, index=False)
    shap_fold_df.to_csv(shap_fold_path, index=False)
    shap_top_price.to_csv(shap_top_price_path, index=False)
    shap_top_overall.to_csv(shap_top_overall_path, index=False)
    glm_coef_agg.to_csv(glm_coef_path, index=False)
    glm_coef_fold_df.to_csv(glm_coef_fold_path, index=False)
    glm_diag_df.to_csv(glm_diag_path, index=False)
    glm_coef_price_top.to_csv(glm_top_price_path, index=False)
    glm_coef_overall_top.to_csv(glm_top_overall_path, index=False)
    config_df.to_csv(config_path, index=False)

    lines = [
        "# Dynamic Demand Model Run (Primary: Rooms Pickup)",
        "",
        "## Locked Policy",
        f"- Stay window: `{MODELING_STAY_START.date()}` to `{MODELING_STAY_END.date()}`",
        f"- Realized cutoff reference: `{REALIZED_CUTOFF.date()}`",
        f"- Primary target: `{TARGET_PRIMARY}`",
        f"- Secondary diagnostic target: `{TARGET_SECONDARY}`",
        "",
        "## Modeling Design",
        f"- Split: `{SPLIT_SCHEME}` with {OUTER_N_SPLITS} folds x {len(SEEDS)} seeds",
        "- GLM: regularized log-price with inner alpha selection; NegBin uses fold-specific dispersion",
        "- XGB: per-hotel feature set with own/compset log prices, numeric DBA, seasonality, comp coverage count",
        "",
        "## Portfolio Metrics",
    ]
    for _, r in portfolio_df.iterrows():
        lines.append(
            f"- {r['model']} ({r['dataset_used']}, iters={int(r['iters'])}): "
            f"MAE={r['weighted_mae']:.4f} +/- {r['weighted_mae_sd']:.4f}, "
            f"WAPE={r['weighted_wape']:.4f} +/- {r['weighted_wape_sd']:.4f}, "
            f"Bias={r['weighted_bias']:.4f} +/- {r['weighted_bias_sd']:.4f}, "
            f"PoissonDev={r['weighted_poisson_deviance']:.4f} +/- {r['weighted_poisson_deviance_sd']:.4f}, "
            f"RMSE={r['weighted_rmse']:.4f} +/- {r['weighted_rmse_sd']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Key Output Files",
            f"- `{holdout_path}`",
            f"- `{portfolio_path}`",
            f"- `{portfolio_iter_path}`",
            f"- `{best_path}`",
            f"- `{vs_baseline_path}`",
            f"- `{secondary_diag_path}`",
            f"- `{gain_path}`",
            f"- `{shap_path}`",
            f"- `{glm_coef_path}`",
            f"- `{glm_diag_path}`",
            f"- `{config_path}`",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Dynamic demand modeling completed with grouped random stay-date CV.")
    print(f"[OK] Holdout rows: {len(holdout_df)}")
    print(f"[OK] Portfolio rows: {len(portfolio_df)}")
    print(f"[OK] SHAP rows (agg): {len(shap_df)}")
    print(f"[OK] GLM coefficient rows (agg): {len(glm_coef_agg)}")
    print(f"[OK] Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic demand modeling pipeline (grouped random date CV)."
    )
    parser.add_argument(
        "--hotels",
        type=str,
        default="",
        help="Comma-separated hotel IDs to model, or ALL. Default is all hotels found in model input file.",
    )
    args = parser.parse_args()
    hotels = parse_hotels_arg(args.hotels)
    main(hotel_filter=hotels if args.hotels else None)
