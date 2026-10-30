"""
Compare five classifiers for BSR-presence prediction and export artifacts.

This script evaluates XGBoost, LightGBM, Random Forest, SVM, and MLP on
the 11-feature BSR dataset using 5-degree spatial cross-validation
(GroupKFold). It then:

1. Saves a model-comparison table (Table 1).
2. Trains a final XGBoost model on all data for global prediction and SHAP.
3. Exports SHAP values and the corresponding feature matrix.
4. Extracts false negatives and true positives from the spatial CV.
5. Saves cross-fold feature importance for XGBoost.

Reproducibility note
--------------------
Tree-based models (XGBoost, LightGBM, Random Forest) can show small
run-to-run variability due to floating-point non-determinism, thread
scheduling, and differences in library versions or hardware. The random
seed is fixed (``RANDOM_STATE = 42``), so results should be close across
runs, but exact metric values may differ slightly from those reported in
the paper. The overall trends and conclusions are stable, and the
methodology and evaluation protocol are fully reproducible.
"""

import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
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

MODEL_NAMES = ["XGBoost", "LightGBM", "Random Forest", "SVM", "MLP"]

BLOCK_SIZE = 5.0
N_SPLITS = 5
RANDOM_STATE = 42

SHAP_MAX_SAMPLES = 3000
XGB_SCALE_POS_WEIGHT = 5.0


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------
def build_model(model_name):
    """Return a fresh classifier (or pipeline) for the given model name.

    Parameters
    ----------
    model_name : str
        One of ``MODEL_NAMES``.

    Returns
    -------
    sklearn-like estimator
    """
    if model_name == "XGBoost":
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
    if model_name == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    if model_name == "Random Forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])
    if model_name == "SVM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ])
    if model_name == "MLP":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=RANDOM_STATE,
                early_stopping=True,
                n_iter_no_change=20,
            )),
        ])
    raise ValueError(f"Unknown model name: {model_name}")


def extract_importances(model):
    """Return feature_importances_ from a raw model or pipeline.

    Returns
    -------
    np.ndarray or None
    """
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_importances_"):
                return step.feature_importances_
    return None


# ---------------------------------------------------------------------
# Spatial cross-validation
# ---------------------------------------------------------------------
def run_spatial_cv(model_name, X, y, groups, n_splits=N_SPLITS):
    """Run GroupKFold spatial CV for one model.

    Returns
    -------
    tuple
        ``(fold_df, importances, y_true, y_pred, y_proba)``.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    all_importances = []
    all_y_true, all_y_pred, all_y_proba = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = build_model(model_name)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        fold_results.append({
            "fold": fold + 1,
            "acc": accuracy_score(y_te, y_pred),
            "prec": precision_score(y_te, y_pred, zero_division=0),
            "rec": recall_score(y_te, y_pred, zero_division=0),
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "auc": roc_auc_score(y_te, y_proba),
        })

        all_y_true.extend(y_te.values)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

        imp = extract_importances(model)
        if imp is not None:
            all_importances.append(imp)

        print(
            f"  Fold {fold + 1}: "
            f"ACC={fold_results[-1]['acc']:.4f} "
            f"AUC={fold_results[-1]['auc']:.4f} "
            f"F1={fold_results[-1]['f1']:.4f}"
        )

    return (
        pd.DataFrame(fold_results),
        np.array(all_importances) if all_importances else None,
        np.array(all_y_true),
        np.array(all_y_pred),
        np.array(all_y_proba),
    )


def build_summary_table(all_summaries):
    """Build the paper's Table 1 from per-model fold DataFrames."""
    rows = []
    for name in MODEL_NAMES:
        f = all_summaries[name]
        rows.append({
            "Model": name,
            "Accuracy": f"{f['acc'].mean() * 100:.2f} ± "
                        f"{f['acc'].std() * 100:.2f}%",
            "Precision": f"{f['prec'].mean() * 100:.2f} ± "
                         f"{f['prec'].std() * 100:.2f}%",
            "Recall": f"{f['rec'].mean() * 100:.2f} ± "
                      f"{f['rec'].std() * 100:.2f}%",
            "F1": f"{f['f1'].mean() * 100:.2f} ± "
                  f"{f['f1'].std() * 100:.2f}%",
            "AUC": f"{f['auc'].mean() * 100:.2f} ± "
                   f"{f['auc'].std() * 100:.2f}%",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Final model and artifact export
# ---------------------------------------------------------------------
def train_final_xgb(X, y, save_dir):
    """Fit the final XGBoost model on all data and save it.

    Returns
    -------
    xgb.XGBClassifier
    """
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=XGB_SCALE_POS_WEIGHT,
    )
    model.fit(X, y)
    model.save_model(os.path.join(save_dir, "xgboost_final.json"))
    joblib.dump(model, os.path.join(save_dir, "xgboost_final.pkl"))
    print("Saved: xgboost_final.json / .pkl")
    return model


def export_shap_values(model, X, save_dir, max_samples=SHAP_MAX_SAMPLES):
    """Compute SHAP values on a random subsample and save them."""
    print("\nComputing global SHAP values...")
    try:
        dmatrix = xgb.DMatrix(X)
        shap_all = model.get_booster().predict(dmatrix, pred_contribs=True)
        shap_all = shap_all[:, :-1]  # drop base_score column

        rng = np.random.default_rng(RANDOM_STATE)
        n_samples = min(max_samples, len(X))
        idx = rng.choice(len(X), n_samples, replace=False)

        np.save(
            os.path.join(save_dir, "shap_values.npy"),
            shap_all[idx],
        )
        X.iloc[idx].to_csv(
            os.path.join(save_dir, "shap_features.csv"),
            index=False,
        )
        print(f"Saved: shap_values.npy (shape {shap_all[idx].shape})")
        print(f"Saved: shap_features.csv (shape {X.iloc[idx].shape})")
    except Exception as exc:  # noqa: BLE001
        print(f"SHAP computation failed: {exc}")


def export_cv_false_negatives(df, X, y, groups, save_dir, n_splits=N_SPLITS):
    """Re-run XGBoost spatial CV and export false negatives / true positives."""
    print("\nExtracting XGBoost CV false negatives and true positives...")

    gkf = GroupKFold(n_splits=n_splits)
    fn_records, tp_records = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = build_model("XGBoost")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        fn_mask = (y_te.values == 1) & (y_pred == 0)
        tp_mask = (y_te.values == 1) & (y_pred == 1)

        if fn_mask.sum() > 0:
            fn_df = df.iloc[test_idx][fn_mask].copy()
            fn_df["fold"] = fold + 1
            fn_records.append(fn_df)

        if tp_mask.sum() > 0:
            tp_df = df.iloc[test_idx][tp_mask].copy()
            tp_df["fold"] = fold + 1
            tp_records.append(tp_df)

    if fn_records:
        fn_all = pd.concat(fn_records, ignore_index=True)
        fn_all.to_csv(
            os.path.join(save_dir, "false_negatives_cv.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Saved false negatives: {len(fn_all)} rows")
    else:
        print("No false negatives found.")

    if tp_records:
        tp_all = pd.concat(tp_records, ignore_index=True)
        tp_all.to_csv(
            os.path.join(save_dir, "true_positives_cv.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Saved true positives: {len(tp_all)} rows")
    else:
        print("No true positives found.")


def save_feature_importance(xgb_imp, save_dir):
    """Save cross-fold XGBoost feature importance to CSV."""
    if xgb_imp is None:
        print("No XGBoost feature importance available.")
        return

    imp_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "XGBoost_mean": xgb_imp.mean(axis=0),
        "XGBoost_std": xgb_imp.std(axis=0),
    })
    imp_df["rank"] = (
        imp_df["XGBoost_mean"].rank(ascending=False).astype(int)
    )
    imp_df = imp_df.sort_values("XGBoost_mean", ascending=False)
    print(imp_df.to_string(index=False))
    imp_df.to_csv(
        os.path.join(save_dir, "feature_importance.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print("Saved: feature_importance.csv")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare classifiers for BSR-presence prediction."
    )
    parser.add_argument(
        "--csv",
        default="BSR_ML_dataset.csv",
        help="Input CSV produced by build_bsr_dataset.py.",
    )
    parser.add_argument(
        "--save-dir",
        default="Classifier_models",
        help="Directory for model files and exported tables.",
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
        "--shap-max-samples",
        type=int,
        default=SHAP_MAX_SAMPLES,
        help="Maximum number of samples used for SHAP export.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the full model-comparison and export workflow."""
    args = parse_args(argv)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Models and SHAP data will be saved to: {save_dir}")

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

    print(f"\nUsing {len(FEATURE_COLS)} features:")
    for f in FEATURE_COLS:
        print(f"  - {f}")

    X = df[FEATURE_COLS]
    y = df["label"]

    pd.Series(FEATURE_COLS).to_csv(
        os.path.join(save_dir, "feature_names.csv"),
        index=False,
        header=["feature"],
    )

    # -----------------------------------------------------------------
    # 2. Spatial blocks
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

    # -----------------------------------------------------------------
    # 3. Run all five models
    # -----------------------------------------------------------------
    all_summaries = {}
    cv_predictions = {}

    for model_name in MODEL_NAMES:
        print("\n" + "=" * 60)
        print(f"  {model_name} spatial cross-validation")
        print("=" * 60)

        fold_df, imp, yt, yp, ypr = run_spatial_cv(
            model_name, X, y, groups, args.n_splits
        )
        all_summaries[model_name] = fold_df
        cv_predictions[model_name] = (yt, yp, ypr, imp)

        for key, label in (
            ("acc", "Accuracy"),
            ("prec", "Precision"),
            ("rec", "Recall"),
            ("f1", "F1"),
            ("auc", "AUC"),
        ):
            print(
                f"  Mean {label:9s}: {fold_df[key].mean():.4f} "
                f"+/- {fold_df[key].std():.4f}"
            )

    # -----------------------------------------------------------------
    # 4. Summary table
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Table 1: model comparison (spatial CV, mean +/- std)")
    print("=" * 70)

    summary_df = build_summary_table(all_summaries)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(save_dir, "Table1_model_comparison.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {summary_path}")

    # -----------------------------------------------------------------
    # 5. Final XGBoost model
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Training final XGBoost on all data (global prediction + SHAP)")
    print("=" * 60)

    xgb_final = train_final_xgb(X, y, save_dir)

    # -----------------------------------------------------------------
    # 6. SHAP export
    # -----------------------------------------------------------------
    export_shap_values(xgb_final, X, save_dir, args.shap_max_samples)

    # -----------------------------------------------------------------
    # 7. False negatives / true positives
    # -----------------------------------------------------------------
    export_cv_false_negatives(df, X, y, groups, save_dir, args.n_splits)

    # -----------------------------------------------------------------
    # 8. Feature importance
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Feature importance (XGBoost spatial CV, across folds)")
    print("=" * 60)

    xgb_imp = cv_predictions["XGBoost"][3]
    save_feature_importance(xgb_imp, save_dir)

    print("\n" + "=" * 60)
    print(f"  All done. Outputs saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)