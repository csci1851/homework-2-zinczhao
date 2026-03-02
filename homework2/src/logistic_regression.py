#!/usr/bin/env python3
"""
Train Logistic Regression on cancer genomics using only top-K features ranked by
a (single-run) Gradient Boosting model.

Fix included:
- If GB importance names are like 'f5167', we convert them to integer column
  indices and slice X by POSITION (works for NumPy arrays or pandas DataFrames).

No hyperparameter tuning: we pick strong default hyperparams for both models.

Expected repo structure (same as your notebook):
- hw2_loader.py provides HW2DataLoader
- model.py provides GradientBoostingModel
- data directory contains:
    cancer_genomics.csv
    labels_cancer_genomics.csv

Usage:
  python train_lr_topk_from_gb.py --k 100 --data_dir data
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from hw2_loader import HW2DataLoader
from model import GradientBoostingModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data", help="Path to data directory.")
    p.add_argument("--k", type=int, default=100, help="Top-K features to keep (by GB importance).")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split fraction.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--save_cm", type=str, default="", help="Optional path to save confusion matrix PNG.")
    return p.parse_args()


def f_to_index(f: str) -> int:
    """Convert 'f123' -> 123 (positional feature index)."""
    m = re.fullmatch(r"f(\d+)", str(f))
    if m is None:
        raise ValueError(f"Unexpected feature name: {f} (expected like 'f123')")
    return int(m.group(1))


def slice_topk(X, top_idx: list[int]):
    """
    Slice feature matrix by column POSITION.
    Works for:
      - pandas DataFrame (uses .iloc)
      - numpy ndarray (uses [:, idx])
    """
    if hasattr(X, "iloc"):  # pandas DataFrame
        return X.iloc[:, top_idx].copy()
    return X[:, top_idx].copy()


def maybe_get_feature_names(X, top_idx: list[int]) -> list[str] | None:
    """
    If X is a pandas DataFrame, return the corresponding column names for top_idx.
    Otherwise return None.
    """
    if hasattr(X, "columns"):
        cols = list(X.columns)
        return [str(cols[i]) for i in top_idx]
    return None


def main() -> None:
    args = parse_args()
    sns.set_style("whitegrid")

    data_dir = Path(args.data_dir)
    cancer_path = data_dir / "cancer_genomics.csv"
    labels_path = data_dir / "labels_cancer_genomics.csv"

    # -------------------------
    # 1) Load dataset
    # -------------------------
    loader = HW2DataLoader()
    X_cancer, y_cancer = loader.get_cancer_genomics_data(
        csv_path=cancer_path,
        labels_path=labels_path,
    )
    print(f"Loaded X_cancer={getattr(X_cancer, 'shape', None)}, y_cancer={getattr(y_cancer, 'shape', None)}")
    try:
        print("Label distribution:", y_cancer.value_counts().to_dict())
    except Exception:
        pass

    # -------------------------
    # 2) Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_cancer,
        y_cancer,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_cancer,
    )

    # -------------------------
    # 3) Fit Gradient Boosting (no tuning; chosen "good default" hyperparams)
    # -------------------------
    gb = GradientBoostingModel(
        task="classification",
        max_depth=3,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=args.seed,
        use_scaler=False,  # trees don't need scaling
    )
    gb.fit(X_train, y_train)

    # Optional: quick evaluation with your helper (if implemented)
    try:
        print("\n[Gradient Boosting] evaluate():", gb.evaluate(X_test, y_test))
    except Exception:
        pass

    # -------------------------
    # 4) Select top-K features by GB importance
    #    IMPORTANT: importance features are 'f####' (positional), so slice by position.
    # -------------------------
    fi = gb.get_feature_importance(plot=False)  # expects columns: feature, importance
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)

    K = int(args.k)
    if K <= 0:
        raise ValueError("--k must be positive.")
    if K > len(fi):
        print(f"Warning: K={K} > num_features={len(fi)}; using all features.")
        K = len(fi)

    # ---- NEW: print the top-K importance table (exact style like your example)
    print("\nTop-K features by Gradient Boosting importance:")
    print(fi.head(K).to_string(index=True))

    top_f = fi.head(K)["feature"].tolist()          # like ['f5167', 'f4437', ...]
    top_idx = [f_to_index(f) for f in top_f]        # like [5167, 4437, ...]
    print(f"\nUsing top-K={K} features from GB importance.")
    print("Top 10 (positional IDs):", top_f[:10])

    # If X is a DataFrame, also show the true column names (optional but helpful)
    top_names = maybe_get_feature_names(X_train, top_idx)
    if top_names is not None:
        print("Top 10 (actual column names):", top_names[:10])

        # Optional: show a small mapping table for interpretability
        print("\nTop-K mapping (feature_id -> column_name -> importance):")
        tmp = fi.head(K).copy()
        tmp["column_name"] = top_names
        print(tmp[["feature", "column_name", "importance"]].head(20).to_string(index=True))
        if K > 20:
            print(f"... (showing first 20 of {K})")

    X_train_k = slice_topk(X_train, top_idx)
    X_test_k = slice_topk(X_test, top_idx)

    # -------------------------
    # 5) Fit Logistic Regression on top-K features (no tuning; chosen defaults)
    # -------------------------
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            multi_class="auto",
            class_weight="balanced",  # helps if labels are imbalanced
            random_state=args.seed,
        )),
    ])

    lr.fit(X_train_k, y_train)
    y_pred = lr.predict(X_test_k)

    print("\n[LogReg] Test classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # -------------------------
    # 6) Confusion matrix
    # -------------------------
    labels = np.unique(np.array(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Cancer Genomics Confusion Matrix (Test) — LogReg (Top-{K} GB Features)")
    plt.tight_layout()

    if args.save_cm:
        out = Path(args.save_cm)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"\nSaved confusion matrix to: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()