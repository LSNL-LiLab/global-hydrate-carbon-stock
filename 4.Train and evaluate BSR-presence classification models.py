"""
Train and evaluate BSR-presence classification models.

This script compares XGBoost and LightGBM for predicting BSR occurrence
from the 11-feature BSR dataset. Two evaluation strategies are used:

1. Random train/test split (stratified 70/30).
2. Spatial cross-validation with 5-degree blocks (GroupKFold),
   which prevents spatial autocorrelation from inflating accuracy.

Note on reproducibility
-----------------------
Tree-based models (XGBoost, LightGBM) can show small run-to-run
variability due to floating-point non-determinism, thread scheduling,
and differences in library versions or hardware. The random seed is
fixed (``RANDOM_STATE = 42``), so results should be close across
runs, but exact metric values may differ slightly from those reported
in the paper. The overall trends and conclusions are stable, and the
methodology and evaluation protocol are fully reproducible.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------
FEATURE_COLS = [
    "crustal_age",
    "dist_to_ridge",
    "dist_to_subduction",
    "sediment_thickness",
    "BGHS_thickness",
    "bottom_salinity",
    "bottom_temp",
    "geothermal_gradient",
    "slope",
    "water_depth",
    "tectonic_environment",
]

BLOCK_SIZE = 5.0
N_SPLITS = 5
TEST_SIZE = 0.3
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------
def build_xgb_classifier():
    """Return an XGBoost classifier with the project defaults."""
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
    )


def build_lgb_classifier():
    """Return a LightGBM classifier with the project defaults."""
    return lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    """Return a dict of classification metrics."""
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
    }


def print_confusion(y_true, y_pred, title):
    """Print a 2x2 confusion matrix with a title."""
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[{title}]")
    print(f"  TN={cm[0, 0]:5d}  FP={cm[0, 1]:5d}")
    print(f"  FN={cm[1, 0]:5d}  TP={cm[1, 1]:5d}")


def evaluate_random_split(model, X_tr, y_tr, X_te, y_te, name):
    """Evaluate a fitted model on train and test splits."""
    print("\n" + "=" * 60)
    print(f"  {name} (random split)")
    print("=" * 60)

    results = {}
    for split_name, X_, y_ in (("train", X_tr, y_tr), ("test", X_te, y_te)):
        y_pred = model.predict(X_)
        y_proba = model.predict_proba(X_)[:, 1]
        m = compute_metrics(y_, y_pred, y_proba)

        print(f"\n[{split_name}]")
        print(f"  Accuracy : {m['acc']:.4f}")
        print(f"  Precision: {m['prec']:.4f}")
        print(f"  Recall   : {m['rec']:.4f}")
        print(f"  F1       : {m['f1']:.4f}")
        print(f"  AUC      : {m['auc']:.4f}")
        results[split_name] = m

    print_confusion(y_te, model.predict(X_te), "test confusion matrix")
    return results


def run_spatial_cv(model_type, X, y, groups, n_splits=N_SPLITS):
    """Run GroupKFold spatial cross-validation.

    Parameters
    ----------
    model_type : {"xgb", "lgb"}
        Which classifier to build.
    X, y : pd.DataFrame / pd.Series
        Features and labels.
    groups : np.ndarray
        Spatial block identifiers.
    n_splits : int
        Number of folds.

    Returns
    -------
    tuple
        ``(fold_results, importances, y_true, y_pred, y_proba)``.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    all_importances = []
    all_y_true, all_y_pred, all_y_proba = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = (
            build_xgb_classifier()
            if model_type == "xgb"
            else build_lgb_classifier()
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, y_pred, y_proba)

        fold_results.append(
            {
                "fold": fold + 1,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_pos_test": int((y_te == 1).sum()),
                "n_neg_test": int((y_te == 0).sum()),
                **m,
            }
        )
        all_importances.append(model.feature_importances_)
        all_y_true.extend(y_te.values)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

        print(
            f"  Fold {fold + 1}: train={len(train_idx)}, test={len(test_idx)} "
            f"(pos {int((y_te == 1).sum())}/neg {int((y_te == 0).sum())}) | "
            f"ACC={m['acc']:.4f} AUC={m['auc']:.4f} F1={m['f1']:.4f}"
        )

    return (
        fold_results,
        np.array(all_importances),
        np.array(all_y_true),
        np.array(all_y_pred),
        np.array(all_y_proba),
    )


def summarize_cv(fold_results, name):
    """Print a fold-level summary table and return it as a DataFrame."""
    df_f = pd.DataFrame(fold_results)
    print("\n" + "=" * 60)
    print(f"  {name} spatial CV summary")
    print("=" * 60)
    print(df_f.to_string(index=False))

    for key, label in (
        ("acc", "Accuracy"),
        ("prec", "Precision"),
        ("rec", "Recall"),
        ("f1", "F1"),
        ("auc", "AUC"),
    ):
        print(
            f"  Mean {label:9s}: {df_f[key].mean():.4f} "
            f"+/- {df_f[key].std():.4f}"
        )
    return df_f


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate BSR-presence classifiers."
    )
    parser.add_argument(
        "--csv",
        default="BSR_ML_dataset.csv",
        help="Input CSV produced by build_bsr_dataset.py.",
    )
    parser.add_argument(
        "--block-size",
        type=float,
        default=BLOCK_SIZE,
        help="Spatial block size in degrees for GroupKFold.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help="Number of spatial CV folds.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of data used for the random test split.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the full classification workflow."""
    args = parse_args(argv)

    # -----------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(
        f"Positives: {(df['label'] == 1).sum()}, "
        f"negatives: {(df['label'] == 0).sum()}"
    )

    X = df[FEATURE_COLS]
    y = df["label"]

    print(f"\nUsing {len(FEATURE_COLS)} features:")
    for f in FEATURE_COLS:
        print(f"  - {f}")

    # -----------------------------------------------------------------
    # 2. Random train/test split
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(
        f"\nRandom split -> train: {len(X_train)}, test: {len(X_test)}"
    )

    # -----------------------------------------------------------------
    # 3. Random-split evaluation
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Training XGBoost (random split)")
    print("=" * 60)
    xgb_model = build_xgb_classifier()
    xgb_model.fit(X_train, y_train)
    evaluate_random_split(
        xgb_model, X_train, y_train, X_test, y_test, "XGBoost"
    )

    print("\n" + "=" * 60)
    print("  Training LightGBM (random split)")
    print("=" * 60)
    lgb_model = build_lgb_classifier()
    lgb_model.fit(X_train, y_train)
    evaluate_random_split(
        lgb_model, X_train, y_train, X_test, y_test, "LightGBM"
    )

    # -----------------------------------------------------------------
    # 4. Spatial cross-validation
    # -----------------------------------------------------------------
    df["block_lon"] = np.floor(df["X"] / args.block_size).astype(int)
    df["block_lat"] = np.floor(df["Y"] / args.block_size).astype(int)
    df["spatial_block"] = (
        df["block_lon"].astype(str) + "_" + df["block_lat"].astype(str)
    )
    groups = df["spatial_block"].values

    print(
        f"\nSpatial blocks: {args.block_size} x {args.block_size} degrees, "
        f"{df['spatial_block'].nunique()} unique blocks"
    )

    print("\n" + "=" * 60)
    print("  XGBoost spatial cross-validation")
    print("=" * 60)
    xgb_folds, xgb_imp, xgb_yt, xgb_yp, xgb_ypr = run_spatial_cv(
        "xgb", X, y, groups, args.n_splits
    )

    print("\n" + "=" * 60)
    print("  LightGBM spatial cross-validation")
    print("=" * 60)
    lgb_folds, lgb_imp, lgb_yt, lgb_yp, lgb_ypr = run_spatial_cv(
        "lgb", X, y, groups, args.n_splits
    )

    # -----------------------------------------------------------------
    # 5. Summaries
    # -----------------------------------------------------------------
    xgb_summary = summarize_cv(xgb_folds, "XGBoost")
    lgb_summary = summarize_cv(lgb_folds, "LightGBM")

    # -----------------------------------------------------------------
    # 6. Overall confusion matrices
    # -----------------------------------------------------------------
    print_confusion(xgb_yt, xgb_yp, "XGBoost spatial CV (overall)")
    print(f"  Overall accuracy: {accuracy_score(xgb_yt, xgb_yp):.4f}")
    print(f"  Overall AUC     : {roc_auc_score(xgb_yt, xgb_ypr):.4f}")

    print_confusion(lgb_yt, lgb_yp, "LightGBM spatial CV (overall)")
    print(f"  Overall accuracy: {accuracy_score(lgb_yt, lgb_yp):.4f}")
    print(f"  Overall AUC     : {roc_auc_score(lgb_yt, lgb_ypr):.4f}")

    # -----------------------------------------------------------------
    # 7. Feature importance (across folds)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Feature importance (spatial CV, averaged across folds)")
    print("=" * 60)

    imp_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLS,
            "XGBoost_mean": xgb_imp.mean(axis=0),
            "XGBoost_std": xgb_imp.std(axis=0),
            "LightGBM_mean": lgb_imp.mean(axis=0),
            "LightGBM_std": lgb_imp.std(axis=0),
        }
    )
    imp_df["XGB_rank"] = (
        imp_df["XGBoost_mean"].rank(ascending=False).astype(int)
    )
    imp_df["LGB_rank"] = (
        imp_df["LightGBM_mean"].rank(ascending=False).astype(int)
    )
    imp_df = imp_df.sort_values("XGBoost_mean", ascending=False)
    print(imp_df.to_string(index=False))

    # -----------------------------------------------------------------
    # 8. Comparison with the previous 12-feature run
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Comparison: before vs after removing ocean_stratigraphic_age")
    print("=" * 60)
    print("Before (12 features):")
    print(
        "  XGBoost  spatial CV: ACC=0.9160, AUC=0.9795, "
        "F1=0.8774, Recall=0.8230"
    )
    print(
        "  LightGBM spatial CV: ACC=0.9142, AUC=0.9779, "
        "F1=0.8733, Recall=0.8135"
    )
    print("\nAfter (11 features):")
    print(
        f"  XGBoost  spatial CV: ACC={xgb_summary['acc'].mean():.4f}, "
        f"AUC={xgb_summary['auc'].mean():.4f}, "
        f"F1={xgb_summary['f1'].mean():.4f}, "
        f"Recall={xgb_summary['rec'].mean():.4f}"
    )
    print(
        f"  LightGBM spatial CV: ACC={lgb_summary['acc'].mean():.4f}, "
        f"AUC={lgb_summary['auc'].mean():.4f}, "
        f"F1={lgb_summary['f1'].mean():.4f}, "
        f"Recall={lgb_summary['rec'].mean():.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)