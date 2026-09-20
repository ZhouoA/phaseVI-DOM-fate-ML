"""Phase VI DOM-fate classification and interpretable machine learning.

This script implements the analysis described in SI Text S8 for the paired
Phase VI PD/A influent and effluent FT-ICR MS formula tables. It:

1. assigns precursors, products, and resistant formulas using Text S5 rules;
2. constructs the 17 prespecified molecular descriptors;
3. performs a stratified 70:30 train/test split;
4. removes highly correlated descriptors within training folds only;
5. tunes random-forest and regularized multinomial logistic-regression models;
6. evaluates both models once on the held-out test set; and
7. calculates SHAP values for the fitted random-forest model.

Raw input workbooks are read only. Input and output locations are supplied on
the command line. A previously generated processed dataset can also be used to
reproduce the modeling and SHAP workflow without distributing raw workbooks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


DEFAULT_PROCESSED_DIR = Path("results/run/processed")
DEFAULT_OUTPUT_DIR = Path("results/run")

FEATURES = [
    "O/C",
    "MW",
    "H",
    "DBE-O",
    "NOSC",
    "S/C",
    "O",
    "S",
    "C",
    "H/C",
    "DBE",
    "AImod",
    "N",
    "DBE/C",
    "N/C",
    "P/C",
    "P",
]

CLASS_ORDER = ["precursors", "products", "resistant"]
DISPLAY_NAMES = {
    "precursors": "Precursors",
    "products": "Products",
    "resistant": "Resistant formulas",
}
CLASS_COLORS = {
    "precursors": "#377EB8",
    "products": "#E41A1C",
    "resistant": "#4DAF4A",
}

RATIO_COMPONENTS = {
    "O/C": {"O", "C"},
    "H/C": {"H", "C"},
    "N/C": {"N", "C"},
    "S/C": {"S", "C"},
    "P/C": {"P", "C"},
}
SECONDARY_COMPONENTS = {
    "DBE-O": {"DBE", "O"},
    "DBE/C": {"DBE", "C"},
}


@dataclass(frozen=True)
class RunConfig:
    seed: int = 20260920
    test_size: float = 0.30
    cv_folds: int = 5
    r2_threshold: float = 0.80
    lower_fc: float = 0.50
    upper_fc: float = 2.00


@dataclass
class TunedModel:
    """Minimal container for a cross-validated model selected by macro-F1."""

    best_estimator_: Pipeline
    best_params_: dict[str, Any]
    best_score_: float


class ChemicalCorrelationSelector(BaseEstimator, TransformerMixin):
    """Remove highly correlated descriptors using prespecified decision rules.

    The transformer is fitted inside each training fold by the scikit-learn
    pipeline. Correlation is assessed with Pearson r and compared on the r^2
    scale. Chemical hierarchy is applied first; unresolved pairs are reduced by
    mean absolute correlation and then fixed feature order. Input and output
    are pandas DataFrames.
    """

    def __init__(
        self,
        r2_threshold: float = 0.80,
    ) -> None:
        self.r2_threshold = r2_threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ChemicalCorrelationSelector":
        del y
        frame = self._as_frame(X)
        retained = list(frame.columns)
        removal_log: list[dict[str, Any]] = []

        while True:
            corr = frame[retained].corr(method="pearson")
            pairs: list[tuple[float, float, str, str]] = []
            for i, left in enumerate(retained):
                for right in retained[i + 1 :]:
                    r_value = corr.loc[left, right]
                    if pd.notna(r_value) and float(r_value) ** 2 > self.r2_threshold:
                        pairs.append(
                            (float(r_value) ** 2, float(r_value), left, right)
                        )
            if not pairs:
                break

            r2_value, r_value, left, right = max(pairs, key=lambda x: x[0])
            chemical_choice = self._chemical_preference(left, right)
            if chemical_choice is not None:
                retained_feature, decision_reason = chemical_choice
                removed = right if retained_feature == left else left
                left_mean_abs = np.nan
                right_mean_abs = np.nan
            else:
                left_others = [feature for feature in retained if feature != left]
                right_others = [feature for feature in retained if feature != right]
                left_mean_abs = float(
                    corr.loc[left, left_others].abs().dropna().mean()
                )
                right_mean_abs = float(
                    corr.loc[right, right_others].abs().dropna().mean()
                )
                if not np.isclose(left_mean_abs, right_mean_abs):
                    removed = left if left_mean_abs > right_mean_abs else right
                    decision_reason = (
                        "higher mean absolute correlation with the remaining predictors"
                    )
                else:
                    feature_order = {feature: i for i, feature in enumerate(FEATURES)}
                    removed = (
                        left
                        if feature_order.get(left, len(FEATURES))
                        > feature_order.get(right, len(FEATURES))
                        else right
                    )
                    decision_reason = (
                        "equal chemical priority and mean absolute correlation; "
                        "later position in FEATURES removed"
                    )
            retained_feature = right if removed == left else left
            removal_log.append(
                {
                    "removed_feature": removed,
                    "retained_feature": retained_feature,
                    "pearson_r": r_value,
                    "r_squared": r2_value,
                    "threshold": self.r2_threshold,
                    "mean_absolute_correlation_left": left_mean_abs,
                    "mean_absolute_correlation_right": right_mean_abs,
                    "reason": decision_reason,
                }
            )
            retained.remove(removed)

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.selected_features_ = retained
        self.removal_log_ = pd.DataFrame(removal_log)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._as_frame(X)
        return frame.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return np.asarray(self.selected_features_, dtype=object)

    def _as_frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)
        raise TypeError("ChemicalCorrelationSelector requires a pandas DataFrame.")

    @staticmethod
    def _chemical_preference(left: str, right: str) -> tuple[str, str] | None:
        for ratio, components in RATIO_COMPONENTS.items():
            if ratio == left and right in components:
                return left, "elemental ratio retained over corresponding atom count"
            if ratio == right and left in components:
                return right, "elemental ratio retained over corresponding atom count"
        for secondary, components in SECONDARY_COMPONENTS.items():
            if secondary == left and right in components:
                return right, "primary descriptor retained over directly derived descriptor"
            if secondary == right and left in components:
                return left, "primary descriptor retained over directly derived descriptor"
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase VI three-class DOM-fate machine learning."
    )
    parser.add_argument(
        "--influent",
        type=Path,
        help="Phase VI influent Excel workbook containing a 'result' sheet.",
    )
    parser.add_argument(
        "--effluent",
        type=Path,
        help="Phase VI effluent Excel workbook containing a 'result' sheet.",
    )
    parser.add_argument(
        "--processed-input",
        type=Path,
        help=(
            "Processed phaseVI_dom_fate_dataset.csv. Use this instead of the "
            "raw influent and effluent workbooks to reproduce modeling."
        ),
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument(
        "--shap-only",
        action="store_true",
        help="Calculate SHAP values from the saved random-forest model without refitting.",
    )
    args = parser.parse_args()
    if not args.shap_only:
        raw_pair_supplied = args.influent is not None or args.effluent is not None
        if args.processed_input is not None and raw_pair_supplied:
            parser.error(
                "Use either --processed-input or the --influent/--effluent pair, not both."
            )
        if args.processed_input is None and (
            args.influent is None or args.effluent is None
        ):
            parser.error(
                "Provide --processed-input or both --influent and --effluent."
            )
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def safe_manifest_path(path: Path) -> str:
    """Return a repository-relative path or, for external inputs, a file name."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def load_formula_table(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="result")
    required = {
        "formula",
        "intens",
        "neu.m/z",
        "O/C",
        "H/C",
        "DBE",
        "NOSC",
        "AI_mod",
        "C",
        "H",
        "N",
        "O",
        "P",
        "S",
        "Cl",
        "Br",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if frame["formula"].duplicated().any():
        duplicates = frame.loc[frame["formula"].duplicated(), "formula"].tolist()
        raise ValueError(f"Duplicate formulas in {path.name}: {duplicates[:10]}")
    intensity = pd.to_numeric(frame["intens"], errors="coerce")
    if intensity.isna().any():
        raise ValueError(f"Missing or nonnumeric intens values in {path.name}.")
    if not np.isfinite(intensity).all():
        raise ValueError(f"Non-finite intens values in {path.name}.")
    if (intensity < 0).any():
        raise ValueError(f"Negative intens values in {path.name}.")
    intensity_sum = float(intensity.sum())
    if intensity_sum <= 0:
        raise ValueError(f"The intens sum in {path.name} must be greater than zero.")
    frame["intens"] = intensity
    frame["RI"] = intensity / intensity_sum
    return frame.set_index("formula", drop=False)


def build_dom_fate_dataset(
    influent: pd.DataFrame,
    effluent: pd.DataFrame,
    config: RunConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    in_formulas = influent.index
    out_formulas = effluent.index
    union = in_formulas.union(out_formulas)
    shared = in_formulas.intersection(out_formulas)
    in_only = in_formulas.difference(out_formulas)
    out_only = out_formulas.difference(in_formulas)

    # Formula-derived values should be identical for formulas detected in both
    # samples. Small run-specific differences in measured neutral mass are
    # accepted by np.isclose; material disagreements trigger an error.
    invariant_columns = [
        "neu.m/z",
        "O/C",
        "H/C",
        "DBE",
        "NOSC",
        "AI_mod",
        "C",
        "H",
        "N",
        "O",
        "P",
        "S",
        "Cl",
        "Br",
    ]
    mismatch_counts: dict[str, int] = {}
    for column in invariant_columns:
        left = influent.loc[shared, column].astype(float)
        right = effluent.loc[shared, column].astype(float)
        both_nan = left.isna() & right.isna()
        mismatch = ~(np.isclose(left, right, equal_nan=True) | both_nan)
        mismatch_counts[column] = int(mismatch.sum())
    if any(mismatch_counts.values()):
        raise ValueError(
            "Shared formulas have inconsistent formula-derived descriptors: "
            f"{mismatch_counts}"
        )

    source = influent.combine_first(effluent).reindex(union)
    dataset = pd.DataFrame(index=union)
    dataset.index.name = "formula"
    dataset["detected_in_influent"] = dataset.index.isin(in_formulas)
    dataset["detected_in_effluent"] = dataset.index.isin(out_formulas)
    dataset["detection_status"] = "shared"
    dataset.loc[in_only, "detection_status"] = "influent_only"
    dataset.loc[out_only, "detection_status"] = "effluent_only"
    dataset["RI_influent"] = influent["RI"].reindex(union)
    dataset["RI_effluent"] = effluent["RI"].reindex(union)
    dataset["FC"] = np.nan
    dataset.loc[shared, "FC"] = (
        dataset.loc[shared, "RI_effluent"] / dataset.loc[shared, "RI_influent"]
    )

    dataset["DOM_fate"] = "resistant"
    dataset.loc[in_only, "DOM_fate"] = "precursors"
    dataset.loc[out_only, "DOM_fate"] = "products"
    dataset.loc[
        shared[dataset.loc[shared, "FC"] < config.lower_fc], "DOM_fate"
    ] = "precursors"
    dataset.loc[
        shared[dataset.loc[shared, "FC"] > config.upper_fc], "DOM_fate"
    ] = "products"

    dataset["O/C"] = source["O/C"].astype(float)
    dataset["MW"] = source["neu.m/z"].astype(float)
    dataset["H"] = source["H"].astype(float)
    dataset["DBE-O"] = source["DBE"].astype(float) - source["O"].astype(float)
    dataset["NOSC"] = source["NOSC"].astype(float)
    dataset["S/C"] = source["S"].astype(float) / source["C"].astype(float)
    dataset["O"] = source["O"].astype(float)
    dataset["S"] = source["S"].astype(float)
    dataset["C"] = source["C"].astype(float)
    dataset["H/C"] = source["H/C"].astype(float)
    dataset["DBE"] = source["DBE"].astype(float)
    dataset["AImod"] = source["AI_mod"].astype(float)
    dataset["N"] = source["N"].astype(float)
    dataset["DBE/C"] = source["DBE"].astype(float) / source["C"].astype(float)
    dataset["N/C"] = source["N"].astype(float) / source["C"].astype(float)
    dataset["P/C"] = source["P"].astype(float) / source["C"].astype(float)
    dataset["P"] = source["P"].astype(float)

    audit = {
        "influent_formula_count": int(len(in_formulas)),
        "effluent_formula_count": int(len(out_formulas)),
        "union_formula_count": int(len(union)),
        "shared_formula_count": int(len(shared)),
        "influent_only_count": int(len(in_only)),
        "effluent_only_count": int(len(out_only)),
        "shared_fc_below_lower": int(
            (dataset.loc[shared, "FC"] < config.lower_fc).sum()
        ),
        "shared_fc_within_thresholds": int(
            (
                (dataset.loc[shared, "FC"] >= config.lower_fc)
                & (dataset.loc[shared, "FC"] <= config.upper_fc)
            ).sum()
        ),
        "shared_fc_above_upper": int(
            (dataset.loc[shared, "FC"] > config.upper_fc).sum()
        ),
        "formula_descriptor_mismatches": mismatch_counts,
        "influent_intensity_sum": float(influent["intens"].sum()),
        "effluent_intensity_sum": float(effluent["intens"].sum()),
        "influent_recomputed_RI_sum": float(influent["RI"].sum()),
        "effluent_recomputed_RI_sum": float(effluent["RI"].sum()),
        "missing_predictor_values": {
            key: int(value)
            for key, value in dataset[FEATURES].isna().sum().items()
            if value > 0
        },
    }
    audit["source_mode"] = "raw_workbooks"
    return dataset.reset_index(), audit


def load_processed_dataset(
    path: Path, config: RunConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the shareable formula-level processed dataset."""
    dataset = pd.read_csv(path)
    required = {
        "formula",
        "detected_in_influent",
        "detected_in_effluent",
        "detection_status",
        "RI_influent",
        "RI_effluent",
        "FC",
        "DOM_fate",
        *FEATURES,
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if dataset["formula"].isna().any() or dataset["formula"].duplicated().any():
        raise ValueError("Processed formula identifiers must be present and unique.")
    unexpected_classes = sorted(set(dataset["DOM_fate"]) - set(CLASS_ORDER))
    if unexpected_classes:
        raise ValueError(f"Unexpected DOM_fate labels: {unexpected_classes}")
    absent_classes = sorted(set(CLASS_ORDER) - set(dataset["DOM_fate"]))
    if absent_classes:
        raise ValueError(f"Processed dataset is missing classes: {absent_classes}")
    for column in FEATURES + ["RI_influent", "RI_effluent", "FC"]:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    if (dataset[["RI_influent", "RI_effluent"]].dropna() < 0).any().any():
        raise ValueError("Relative intensities must be non-negative.")

    in_mask = dataset["detected_in_influent"].astype(bool)
    out_mask = dataset["detected_in_effluent"].astype(bool)
    shared_mask = in_mask & out_mask
    in_only_mask = in_mask & ~out_mask
    out_only_mask = out_mask & ~in_mask
    if not np.isclose(dataset.loc[in_mask, "RI_influent"].sum(), 1.0):
        raise ValueError("Processed influent relative intensities do not sum to one.")
    if not np.isclose(dataset.loc[out_mask, "RI_effluent"].sum(), 1.0):
        raise ValueError("Processed effluent relative intensities do not sum to one.")

    expected_fate = pd.Series("resistant", index=dataset.index, dtype=object)
    expected_fate.loc[in_only_mask] = "precursors"
    expected_fate.loc[out_only_mask] = "products"
    expected_fate.loc[shared_mask & (dataset["FC"] < config.lower_fc)] = "precursors"
    expected_fate.loc[shared_mask & (dataset["FC"] > config.upper_fc)] = "products"
    if not expected_fate.equals(dataset["DOM_fate"].astype(object)):
        mismatch_count = int((expected_fate != dataset["DOM_fate"]).sum())
        raise ValueError(
            f"Processed DOM_fate labels disagree with the prespecified rules for "
            f"{mismatch_count} formulas."
        )

    audit = {
        "source_mode": "processed_dataset",
        "influent_formula_count": int(in_mask.sum()),
        "effluent_formula_count": int(out_mask.sum()),
        "union_formula_count": int(len(dataset)),
        "shared_formula_count": int(shared_mask.sum()),
        "influent_only_count": int(in_only_mask.sum()),
        "effluent_only_count": int(out_only_mask.sum()),
        "shared_fc_below_lower": int(
            (shared_mask & (dataset["FC"] < config.lower_fc)).sum()
        ),
        "shared_fc_within_thresholds": int(
            (
                shared_mask
                & (dataset["FC"] >= config.lower_fc)
                & (dataset["FC"] <= config.upper_fc)
            ).sum()
        ),
        "shared_fc_above_upper": int(
            (shared_mask & (dataset["FC"] > config.upper_fc)).sum()
        ),
        "formula_descriptor_mismatches": "checked during raw-data processing",
        "influent_intensity_sum": None,
        "effluent_intensity_sum": None,
        "influent_recomputed_RI_sum": float(
            dataset.loc[in_mask, "RI_influent"].sum()
        ),
        "effluent_recomputed_RI_sum": float(
            dataset.loc[out_mask, "RI_effluent"].sum()
        ),
        "missing_predictor_values": {
            key: int(value)
            for key, value in dataset[FEATURES].isna().sum().items()
            if value > 0
        },
    }
    return dataset, audit


def make_imputer() -> SimpleImputer:
    imputer = SimpleImputer(strategy="median")
    imputer.set_output(transform="pandas")
    return imputer


def make_pipelines(config: RunConfig) -> dict[str, Pipeline]:
    selector = ChemicalCorrelationSelector(
        r2_threshold=config.r2_threshold,
    )
    random_forest = Pipeline(
        steps=[
            ("imputer", make_imputer()),
            ("selector", selector),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=config.seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    logistic = Pipeline(
        steps=[
            ("imputer", make_imputer()),
            (
                "selector",
                ChemicalCorrelationSelector(
                    r2_threshold=config.r2_threshold,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=config.seed,
                ),
            ),
        ]
    )
    return {"random_forest": random_forest, "logistic_regression": logistic}


def parameter_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "random_forest": {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", 0.5],
        },
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    }


def audit_cv_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: RunConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cv = StratifiedKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.seed
    )
    for fold, (fit_idx, _) in enumerate(cv.split(X_train, y_train), start=1):
        fold_X = X_train.iloc[fit_idx]
        imputer = make_imputer()
        imputed = imputer.fit_transform(fold_X)
        selector = ChemicalCorrelationSelector(
            r2_threshold=config.r2_threshold,
        ).fit(imputed)
        for feature in FEATURES:
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "status": (
                        "retained"
                        if feature in selector.selected_features_
                        else "removed"
                    ),
                }
            )
    return pd.DataFrame(rows)


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: RunConfig,
) -> tuple[dict[str, TunedModel], dict[str, pd.DataFrame]]:
    """Tune each pipeline by fivefold CV without exposing validation data.

    The explicit loop avoids a NumPy 2.2/scikit-learn 1.5 masked-array warning
    raised while GridSearchCV formats integer-valued parameter columns. It uses
    the same ParameterGrid order, folds, metrics and macro-F1 selection rule.
    """
    cv = StratifiedKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.seed
    )
    splits = list(cv.split(X_train, y_train))
    fitted: dict[str, TunedModel] = {}
    cv_tables: dict[str, pd.DataFrame] = {}
    for name, pipeline in make_pipelines(config).items():
        rows: list[dict[str, Any]] = []
        for params in ParameterGrid(parameter_grids()[name]):
            fold_scores = {
                "test_macro_f1": [],
                "test_balanced_accuracy": [],
                "train_macro_f1": [],
                "train_balanced_accuracy": [],
            }
            for train_index, validation_index in splits:
                estimator = clone(pipeline).set_params(**params)
                fold_X_train = X_train.iloc[train_index]
                fold_y_train = y_train.iloc[train_index]
                fold_X_validation = X_train.iloc[validation_index]
                fold_y_validation = y_train.iloc[validation_index]
                estimator.fit(fold_X_train, fold_y_train)
                validation_prediction = estimator.predict(fold_X_validation)
                train_prediction = estimator.predict(fold_X_train)
                fold_scores["test_macro_f1"].append(
                    f1_score(
                        fold_y_validation,
                        validation_prediction,
                        average="macro",
                    )
                )
                fold_scores["test_balanced_accuracy"].append(
                    balanced_accuracy_score(
                        fold_y_validation, validation_prediction
                    )
                )
                fold_scores["train_macro_f1"].append(
                    f1_score(fold_y_train, train_prediction, average="macro")
                )
                fold_scores["train_balanced_accuracy"].append(
                    balanced_accuracy_score(fold_y_train, train_prediction)
                )

            row: dict[str, Any] = {"params": params}
            row.update({f"param_{key}": value for key, value in params.items()})
            for metric, scores in fold_scores.items():
                values = np.asarray(scores, dtype=float)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"std_{metric}"] = float(values.std(ddof=0))
                for fold_index, value in enumerate(values):
                    row[f"split{fold_index}_{metric}"] = float(value)
            rows.append(row)

        table = pd.DataFrame(rows)
        table["rank_test_macro_f1"] = (
            table["mean_test_macro_f1"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        best_index = int(table["mean_test_macro_f1"].idxmax())
        best_params = dict(table.loc[best_index, "params"])
        best_estimator = clone(pipeline).set_params(**best_params)
        best_estimator.fit(X_train, y_train)
        fitted[name] = TunedModel(
            best_estimator_=best_estimator,
            best_params_=best_params,
            best_score_=float(table.loc[best_index, "mean_test_macro_f1"]),
        )
        cv_tables[name] = table.sort_values("rank_test_macro_f1")
    return fitted, cv_tables


def evaluate_model(
    name: str,
    search: TunedModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    estimator = search.best_estimator_
    predicted = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)
    classes = list(estimator.named_steps["model"].classes_)
    if classes != CLASS_ORDER:
        order = [classes.index(label) for label in CLASS_ORDER]
        probabilities = probabilities[:, order]
        classes = CLASS_ORDER

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        predicted,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    report = pd.DataFrame(
        {
            "class": CLASS_ORDER,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    cm_count = confusion_matrix(y_test, predicted, labels=CLASS_ORDER)
    cm_norm = confusion_matrix(
        y_test, predicted, labels=CLASS_ORDER, normalize="true"
    )
    y_binary = label_binarize(y_test, classes=CLASS_ORDER)
    curve_rows: list[dict[str, Any]] = []
    curve_data: dict[str, dict[str, np.ndarray | float]] = {}
    auc_values: list[float] = []
    ap_values: list[float] = []
    for index, label in enumerate(CLASS_ORDER):
        fpr, tpr, _ = roc_curve(y_binary[:, index], probabilities[:, index])
        roc_auc = auc(fpr, tpr)
        pr_precision, pr_recall, _ = precision_recall_curve(
            y_binary[:, index], probabilities[:, index]
        )
        average_precision = average_precision_score(
            y_binary[:, index], probabilities[:, index]
        )
        auc_values.append(float(roc_auc))
        ap_values.append(float(average_precision))
        curve_data[label] = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": float(roc_auc),
            "pr_precision": pr_precision,
            "pr_recall": pr_recall,
            "average_precision": float(average_precision),
        }
        curve_rows.append(
            {
                "model": name,
                "class": label,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
            }
        )

    predictions = pd.DataFrame(
        {
            "formula": X_test.index,
            "observed_DOM_fate": y_test.loc[X_test.index].values,
            "predicted_DOM_fate": predicted,
        }
    )
    for index, label in enumerate(CLASS_ORDER):
        predictions[f"probability_{label}"] = probabilities[:, index]

    return {
        "name": name,
        "estimator": estimator,
        "best_params": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(y_test, predicted)
        ),
        "test_macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "test_macro_roc_auc": float(np.mean(auc_values)),
        "test_macro_average_precision": float(np.mean(ap_values)),
        "classification_report": report,
        "cm_count": cm_count,
        "cm_norm": cm_norm,
        "curve_metrics": pd.DataFrame(curve_rows),
        "curve_data": curve_data,
        "predictions": predictions,
    }


def save_confusion_plot(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    labels = [DISPLAY_NAMES[x] for x in CLASS_ORDER]
    for axis, (name, result) in zip(axes, results.items()):
        sns.heatmap(
            result["cm_norm"],
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=axis,
        )
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Predicted DOM fate")
        axis.set_ylabel("Observed DOM fate")
    for suffix in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"confusion_matrices.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def save_curve_plot(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)
    for row, (name, result) in enumerate(results.items()):
        roc_axis, pr_axis = axes[row]
        for label in CLASS_ORDER:
            curve = result["curve_data"][label]
            roc_axis.plot(
                curve["fpr"],
                curve["tpr"],
                color=CLASS_COLORS[label],
                label=f"{DISPLAY_NAMES[label]} (AUC={curve['roc_auc']:.3f})",
            )
            pr_axis.plot(
                curve["pr_recall"],
                curve["pr_precision"],
                color=CLASS_COLORS[label],
                label=(
                    f"{DISPLAY_NAMES[label]} "
                    f"(AP={curve['average_precision']:.3f})"
                ),
            )
        roc_axis.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1)
        roc_axis.set(
            xlim=(0, 1),
            ylim=(0, 1.02),
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            title=f"{name.replace('_', ' ').title()}: ROC",
        )
        pr_axis.set(
            xlim=(0, 1),
            ylim=(0, 1.02),
            xlabel="Recall",
            ylabel="Precision",
            title=f"{name.replace('_', ' ').title()}: precision–recall",
        )
        roc_axis.legend(frameon=False, fontsize=8)
        pr_axis.legend(frameon=False, fontsize=8)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"roc_pr_curves.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def normalized_shap_values(
    explanation: Any,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    values = explanation.values if hasattr(explanation, "values") else explanation
    if isinstance(values, list):
        values = np.stack(values, axis=2)
    values = np.asarray(values)
    if values.shape == (n_samples, n_features, n_classes):
        return values
    if values.shape == (n_classes, n_samples, n_features):
        return np.transpose(values, (1, 2, 0))
    raise ValueError(
        "Unexpected SHAP array shape: "
        f"{values.shape}; expected ({n_samples}, {n_features}, {n_classes})."
    )


def run_shap(
    rf_estimator: Pipeline,
    X_test: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    try:
        import shap
    except Exception as exc:
        raise RuntimeError(
            "SHAP could not be imported in the active Python environment. "
            "Install a SHAP/Numba/NumPy combination compatible with this Python "
            "version before running the complete analysis. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    preprocessing = rf_estimator[:-1]
    X_transformed = preprocessing.transform(X_test)
    if not isinstance(X_transformed, pd.DataFrame):
        selected = rf_estimator.named_steps["selector"].selected_features_
        X_transformed = pd.DataFrame(
            X_transformed, index=X_test.index, columns=selected
        )
    model = rf_estimator.named_steps["model"]
    classes = list(model.classes_)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_transformed)
    values = normalized_shap_values(
        explanation,
        n_samples=len(X_transformed),
        n_features=X_transformed.shape[1],
        n_classes=len(classes),
    )

    class_order = [classes.index(label) for label in CLASS_ORDER]
    values = values[:, :, class_order]
    mean_abs_by_class = np.mean(np.abs(values), axis=0)
    global_mean_abs = np.mean(np.abs(values), axis=(0, 2))

    global_table = pd.DataFrame(
        {
            "feature": X_transformed.columns,
            "mean_absolute_SHAP": global_mean_abs,
        }
    ).sort_values("mean_absolute_SHAP", ascending=False)
    class_table = pd.DataFrame(
        mean_abs_by_class,
        index=X_transformed.columns,
        columns=CLASS_ORDER,
    ).reset_index(names="feature")
    class_table["overall"] = class_table[CLASS_ORDER].mean(axis=1)
    class_table = class_table.sort_values("overall", ascending=False)

    long_tables: list[pd.DataFrame] = []
    for class_index, label in enumerate(CLASS_ORDER):
        table = pd.DataFrame(
            values[:, :, class_index],
            index=X_transformed.index,
            columns=X_transformed.columns,
        )
        table.index.name = "formula"
        table = table.reset_index().melt(
            id_vars="formula", var_name="feature", value_name="SHAP_value"
        )
        table.insert(1, "class", label)
        long_tables.append(table)
    shap_long = pd.concat(long_tables, ignore_index=True)
    shap_long.to_csv(output_dir / "shap_values_long.csv", index=False)
    np.savez_compressed(
        output_dir / "shap_values.npz",
        values=values,
        formulas=np.asarray(X_transformed.index, dtype=str),
        features=np.asarray(X_transformed.columns, dtype=str),
        classes=np.asarray(CLASS_ORDER, dtype=str),
    )

    ordered = global_table.sort_values("mean_absolute_SHAP", ascending=True)
    fig, axis = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    y_position = np.arange(len(ordered))
    axis.barh(
        y_position,
        ordered["mean_absolute_SHAP"],
        color="#4C78A8",
    )
    axis.set_yticks(y_position, ordered["feature"])
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_ylabel("Molecular descriptor")
    for suffix in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"shap_feature_importance.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    for class_index, label in enumerate(CLASS_ORDER):
        shap.summary_plot(
            values[:, :, class_index],
            X_transformed,
            max_display=min(13, X_transformed.shape[1]),
            show=False,
        )
        plt.title(f"{DISPLAY_NAMES[label]}")
        plt.tight_layout()
        for suffix in ["png", "pdf"]:
            plt.savefig(
                output_dir / f"shap_summary_{label}.{suffix}",
                dpi=600 if suffix == "png" else None,
                bbox_inches="tight",
            )
        plt.close()

    return global_table, class_table, shap.__version__


def run_saved_model_shap(args: argparse.Namespace) -> None:
    """Add SHAP outputs to an already completed model-only run."""
    dataset_path = args.processed_dir / "phaseVI_dom_fate_dataset.csv"
    split_path = args.output_dir / "train_test_split.csv"
    model_path = args.output_dir / "model_random_forest.joblib"
    required_paths = [dataset_path, split_path, model_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "SHAP-only mode requires the completed model-only run. Missing: "
            + ", ".join(missing)
        )

    dataset = pd.read_csv(dataset_path)
    split_table = pd.read_csv(split_path)
    test_formulas = split_table.loc[split_table["split"] == "test", "formula"]
    X = dataset.set_index("formula")[FEATURES]
    X_test = X.loc[test_formulas]
    rf_estimator = joblib.load(model_path)
    selected = rf_estimator.named_steps["selector"].selected_features_
    descriptor_summary = make_descriptor_summary(dataset, selected)
    descriptor_summary.to_csv(
        args.output_dir / "descriptor_summary_by_fate.csv", index=False
    )

    shap_global, shap_class, shap_version = run_shap(
        rf_estimator, X_test, args.output_dir
    )
    shap_global.to_csv(
        args.output_dir / "shap_global_importance.csv", index=False
    )
    shap_class.to_csv(
        args.output_dir / "shap_class_importance.csv", index=False
    )

    workbook = args.output_dir / "phaseVI_dom_fate_ml_results.xlsx"
    if workbook.exists():
        with pd.ExcelWriter(
            workbook,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            shap_global.to_excel(writer, sheet_name="SHAP_global", index=False)
            shap_class.to_excel(writer, sheet_name="SHAP_by_class", index=False)
            descriptor_summary.to_excel(
                writer, sheet_name="descriptor_summary", index=False
            )

    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["script_sha256"] = sha256(Path(__file__))
        manifest.setdefault("versions", {})["shap"] = shap_version
        manifest["shap_skipped"] = False
        manifest["shap_completed"] = datetime.now().astimezone().isoformat()
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                default=json_default,
            ),
            encoding="utf-8",
        )

    report_path = args.output_dir / "DATA_PROCESSING_AND_RESULTS.md"
    if report_path.exists():
        report_lines = report_path.read_text(encoding="utf-8").splitlines()
        top_shap = ", ".join(
            f"{row.feature} ({row.mean_absolute_SHAP:.4f})"
            for row in shap_global.head(10).itertuples()
        )
        prefix = "Top descriptors by mean absolute SHAP value:"
        report_lines = [
            f"{prefix} {top_shap}" if line.startswith(prefix) else line
            for line in report_lines
        ]
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("SHAP outputs added to the completed model run.")
    print(shap_global.head(10).to_string(index=False))


def make_class_count_table(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fate in CLASS_ORDER:
        subset = dataset.loc[dataset["DOM_fate"] == fate]
        rows.append(
            {
                "DOM_fate": fate,
                "total": len(subset),
                "influent_only": int((subset["detection_status"] == "influent_only").sum()),
                "shared": int((subset["detection_status"] == "shared").sum()),
                "effluent_only": int((subset["detection_status"] == "effluent_only").sum()),
                "percentage_of_union": 100 * len(subset) / len(dataset),
            }
        )
    return pd.DataFrame(rows)


def make_descriptor_summary(
    dataset: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Summarize retained descriptors without treating formulas as replicates."""
    rows: list[dict[str, Any]] = []
    for fate in CLASS_ORDER:
        subset = dataset.loc[dataset["DOM_fate"] == fate]
        for feature in features:
            values = pd.to_numeric(subset[feature], errors="coerce").dropna()
            rows.append(
                {
                    "DOM_fate": fate,
                    "feature": feature,
                    "n": len(values),
                    "median": values.median(),
                    "Q1": values.quantile(0.25),
                    "Q3": values.quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def write_excel_summary(
    output_path: Path,
    class_counts: pd.DataFrame,
    split_table: pd.DataFrame,
    selected_features: pd.DataFrame,
    removed_features: pd.DataFrame,
    model_comparison: pd.DataFrame,
    evaluations: dict[str, dict[str, Any]],
    curve_metrics: pd.DataFrame,
    descriptor_summary: pd.DataFrame,
    shap_global: pd.DataFrame | None,
    shap_class: pd.DataFrame | None,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        class_counts.to_excel(writer, sheet_name="class_counts", index=False)
        split_table.to_excel(writer, sheet_name="train_test_split", index=False)
        selected_features.to_excel(writer, sheet_name="selected_features", index=False)
        removed_features.to_excel(writer, sheet_name="correlation_filter", index=False)
        model_comparison.to_excel(writer, sheet_name="model_comparison", index=False)
        curve_metrics.to_excel(writer, sheet_name="ROC_PR_metrics", index=False)
        descriptor_summary.to_excel(
            writer, sheet_name="descriptor_summary", index=False
        )
        for name, result in evaluations.items():
            short = "RF" if name == "random_forest" else "Logistic"
            result["classification_report"].to_excel(
                writer, sheet_name=f"{short}_class_metrics", index=False
            )
            pd.DataFrame(
                result["cm_count"], index=CLASS_ORDER, columns=CLASS_ORDER
            ).to_excel(writer, sheet_name=f"{short}_confusion")
            result["predictions"].to_excel(
                writer, sheet_name=f"{short}_predictions", index=False
            )
        if shap_global is not None:
            shap_global.to_excel(writer, sheet_name="SHAP_global", index=False)
        if shap_class is not None:
            shap_class.to_excel(writer, sheet_name="SHAP_by_class", index=False)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(
            lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)
        )
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    path: Path,
    config: RunConfig,
    audit: dict[str, Any],
    class_counts: pd.DataFrame,
    split_counts: pd.DataFrame,
    fold_features: pd.DataFrame,
    selected_features: pd.DataFrame,
    removed_features: pd.DataFrame,
    model_comparison: pd.DataFrame,
    evaluations: dict[str, dict[str, Any]],
    shap_global: pd.DataFrame | None,
) -> None:
    removal_lines: list[str] = []
    for _, row in removed_features.iterrows():
        removal_lines.append(
            f"- {row['removed_feature']} was removed and {row['retained_feature']} "
            f"was retained (r² = {row['r_squared']:.4f}; {row['reason']})."
        )
    if not removal_lines:
        removal_lines.append("- No descriptors were removed from the final training set.")

    fold_lines: list[str] = []
    for fold, group in fold_features.groupby("fold", sort=True):
        removed = group.loc[group["status"] == "removed", "feature"].tolist()
        retained_count = int((group["status"] == "retained").sum())
        removed_text = ", ".join(removed) if removed else "none"
        fold_lines.append(
            f"- Fold {fold}: retained {retained_count} descriptors; removed {removed_text}."
        )

    retained_final = selected_features.loc[
        selected_features["status"] == "retained", "feature"
    ].tolist()
    removed_final = selected_features.loc[
        selected_features["status"] == "removed", "feature"
    ].tolist()
    missing_values = audit["missing_predictor_values"]
    if missing_values:
        missing_text = ", ".join(
            f"{feature}: {count}" for feature, count in missing_values.items()
        )
        imputation_text = (
            "Missing predictor values were imputed using medians fitted within "
            f"the corresponding training fold ({missing_text})."
        )
    else:
        imputation_text = "No predictor values were missing; no imputation was required."
    top_shap = "SHAP was skipped in this run."
    if shap_global is not None:
        top = shap_global.head(10)
        top_shap = ", ".join(
            f"{row.feature} ({row.mean_absolute_SHAP:.4f})"
            for row in top.itertuples(index=False)
        )

    if audit["source_mode"] == "raw_workbooks":
        source_text = (
            "The Phase VI influent and effluent FT-ICR MS formula tables were "
            "read without modifying the raw workbooks."
        )
        ri_text = (
            "RI was recalculated independently for each sample as `intens / "
            "sum(intens)`. The influent and effluent intensity sums were "
            f"{audit['influent_intensity_sum']:.0f} and "
            f"{audit['effluent_intensity_sum']:.0f}, respectively, and the "
            "recomputed RI sums were "
            f"{audit['influent_recomputed_RI_sum']:.12g} and "
            f"{audit['effluent_recomputed_RI_sum']:.12g}."
        )
    else:
        source_text = (
            "The repository's formula-level processed dataset was validated "
            "and used to reproduce model fitting and evaluation."
        )
        ri_text = (
            "The processed RI columns were generated from the raw workbooks as "
            "`intens / sum(intens)` within each sample. Their validated sums were "
            f"{audit['influent_recomputed_RI_sum']:.12g} for the influent and "
            f"{audit['effluent_recomputed_RI_sum']:.12g} for the effluent."
        )

    report = f"""# Phase VI DOM fate machine-learning analysis

## Data processing

1. {source_text}
2. {ri_text}
3. The formula union contained {audit['union_formula_count']:,} formulas: {audit['shared_formula_count']:,} shared, {audit['influent_only_count']:,} influent-only, and {audit['effluent_only_count']:,} effluent-only formulas.
4. Shared formulas were classified from FC = RI_effluent / RI_influent. FC < {config.lower_fc:g} was assigned to precursors, {config.lower_fc:g} ≤ FC ≤ {config.upper_fc:g} to resistant formulas, and FC > {config.upper_fc:g} to products. Influent-only and effluent-only formulas were assigned to precursors and products, respectively; nondetection was not replaced by zero.
5. {len(FEATURES)} prespecified descriptors were constructed. Measured neutral mass (`neu.m/z`) was used as MW, and NOSC was taken directly from the source table after invariant checking across shared formulas. {imputation_text}
6. Molecular formulas were divided into training and held-out test sets at a 70:30 ratio using stratified sampling by DOM fate (seed = {config.seed}).
7. Within each training fold, descriptors with Pearson r² > {config.r2_threshold:g} were filtered before model fitting. Median imputation, filtering, scaling and hyperparameter tuning were fitted using training data only.
8. Random forest and L2-regularized multinomial logistic regression were tuned by fivefold stratified cross-validation using macro-F1. No over- or undersampling was performed.
9. The fitted models were evaluated once on the held-out test set. SHAP values were calculated from the random forest for the held-out formulas.

## DOM fate counts

{markdown_table(class_counts, ['DOM_fate', 'total', 'influent_only', 'shared', 'effluent_only', 'percentage_of_union'])}

The precursor total comprises {audit['influent_only_count']:,} influent-only formulas and {audit['shared_fc_below_lower']:,} shared formulas with FC < {config.lower_fc:g}. The product total comprises {audit['effluent_only_count']:,} effluent-only formulas and {audit['shared_fc_above_upper']:,} shared formulas with FC > {config.upper_fc:g}. The {audit['shared_fc_within_thresholds']:,} resistant formulas were detected in both samples with FC within the stated thresholds.

## Train/test composition

{markdown_table(split_counts, ['split', 'DOM_fate', 'n'])}

## Collinearity filtering

{chr(10).join(removal_lines)}

The final training-set filter removed {len(removed_final)} descriptors ({', '.join(removed_final) if removed_final else 'none'}) and retained {len(retained_final)} descriptors ({', '.join(retained_final)}).

Training-fold selection audit:

{chr(10).join(fold_lines)}

## Model performance

{markdown_table(model_comparison, ['model', 'best_CV_macro_F1', 'test_balanced_accuracy', 'test_macro_F1', 'test_macro_ROC_AUC', 'test_macro_average_precision'])}

Best random-forest parameters: `{json.dumps(evaluations['random_forest']['best_params'], default=json_default, sort_keys=True)}`

Best logistic-regression parameters: `{json.dumps(evaluations['logistic_regression']['best_params'], default=json_default, sort_keys=True)}`

## SHAP ranking

Top descriptors by mean absolute SHAP value: {top_shap}

## Interpretation boundary

The analysis quantifies how well formula-derived descriptors discriminate the three DOM fate classes within the single paired Phase VI influent-effluent dataset. Molecular formulas are analytical observations rather than independent reactor replicates. The random train/test split therefore provides internal validation across formulas; it does not establish performance across independent reactors or sampling events. The precursor, product and resistant labels describe detection and relative-intensity patterns and do not by themselves prove biodegradation, production or chemical persistence.

## Reproduction

Run from the project root:

```bash
python src/analyze_phaseVI_dom_fate_ml.py \
  --processed-input data/processed/phaseVI_dom_fate_dataset.csv \
  --processed-dir results/run/processed \
  --output-dir results/run
```
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = RunConfig(seed=args.seed)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.shap_only:
        run_saved_model_shap(args)
        return

    input_manifest: list[dict[str, str]] = []
    if args.processed_input is not None:
        dataset, audit = load_processed_dataset(args.processed_input, config)
        input_manifest.append(
            {
                "role": "processed_dataset",
                "file": safe_manifest_path(args.processed_input),
                "sha256": sha256(args.processed_input),
            }
        )
    else:
        influent = load_formula_table(args.influent)
        effluent = load_formula_table(args.effluent)
        dataset, audit = build_dom_fate_dataset(influent, effluent, config)
        input_manifest.extend(
            [
                {
                    "role": "influent_workbook",
                    "file": safe_manifest_path(args.influent),
                    "sha256": sha256(args.influent),
                },
                {
                    "role": "effluent_workbook",
                    "file": safe_manifest_path(args.effluent),
                    "sha256": sha256(args.effluent),
                },
            ]
        )
    dataset_path = args.processed_dir / "phaseVI_dom_fate_dataset.csv"
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    class_counts = make_class_count_table(dataset)
    class_counts.to_csv(args.output_dir / "class_counts.csv", index=False)

    X = dataset.set_index("formula")[FEATURES]
    y = dataset.set_index("formula")["DOM_fate"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )
    split_table = pd.DataFrame(
        {
            "formula": X.index,
            "DOM_fate": y.loc[X.index].values,
            "split": np.where(X.index.isin(X_train.index), "training", "test"),
        }
    )
    split_table.to_csv(args.output_dir / "train_test_split.csv", index=False)
    split_counts = (
        split_table.groupby(["split", "DOM_fate"], sort=False)
        .size()
        .rename("n")
        .reset_index()
    )

    fold_features = audit_cv_feature_selection(X_train, y_train, config)
    fold_features.to_csv(
        args.output_dir / "cv_fold_feature_selection.csv", index=False
    )

    fitted, cv_tables = fit_models(X_train, y_train, config)
    for name, table in cv_tables.items():
        table.to_csv(args.output_dir / f"cv_results_{name}.csv", index=False)

    evaluations = {
        name: evaluate_model(name, search, X_test, y_test)
        for name, search in fitted.items()
    }
    model_comparison = pd.DataFrame(
        [
            {
                "model": name,
                "best_CV_macro_F1": result["best_cv_macro_f1"],
                "test_balanced_accuracy": result["test_balanced_accuracy"],
                "test_macro_F1": result["test_macro_f1"],
                "test_macro_ROC_AUC": result["test_macro_roc_auc"],
                "test_macro_average_precision": result[
                    "test_macro_average_precision"
                ],
            }
            for name, result in evaluations.items()
        ]
    )
    model_comparison.to_csv(
        args.output_dir / "model_comparison.csv", index=False
    )

    prediction_tables = []
    curve_tables = []
    for name, result in evaluations.items():
        result["classification_report"].to_csv(
            args.output_dir / f"classification_report_{name}.csv", index=False
        )
        pd.DataFrame(
            result["cm_count"], index=CLASS_ORDER, columns=CLASS_ORDER
        ).to_csv(args.output_dir / f"confusion_matrix_{name}.csv")
        result["predictions"].insert(0, "model", name)
        prediction_tables.append(result["predictions"])
        curve_tables.append(result["curve_metrics"])
        joblib.dump(
            result["estimator"], args.output_dir / f"model_{name}.joblib"
        )
    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    curve_metrics = pd.concat(curve_tables, ignore_index=True)
    curve_metrics.to_csv(args.output_dir / "roc_pr_metrics.csv", index=False)
    save_confusion_plot(evaluations, args.output_dir)
    save_curve_plot(evaluations, args.output_dir)

    rf_best = fitted["random_forest"].best_estimator_
    logistic_best = fitted["logistic_regression"].best_estimator_
    rf_selected = rf_best.named_steps["selector"].selected_features_
    logistic_selected = logistic_best.named_steps["selector"].selected_features_
    if rf_selected != logistic_selected:
        raise RuntimeError(
            "The two pipelines retained different descriptors on the same training set."
        )
    selected_features = pd.DataFrame(
        {
            "feature": FEATURES,
            "status": [
                "retained" if feature in rf_selected else "removed"
                for feature in FEATURES
            ],
        }
    )
    selected_features.to_csv(
        args.output_dir / "selected_features.csv", index=False
    )
    descriptor_summary = make_descriptor_summary(dataset, rf_selected)
    descriptor_summary.to_csv(
        args.output_dir / "descriptor_summary_by_fate.csv", index=False
    )
    removed_features = rf_best.named_steps["selector"].removal_log_.copy()
    removed_features.to_csv(
        args.output_dir / "removed_correlated_features.csv", index=False
    )

    imputed_train = rf_best.named_steps["imputer"].transform(X_train)
    correlation_before = imputed_train.corr(method="pearson")
    correlation_after = imputed_train[rf_selected].corr(method="pearson")
    correlation_before.to_csv(
        args.output_dir / "correlation_matrix_training_before.csv"
    )
    correlation_after.to_csv(
        args.output_dir / "correlation_matrix_training_after.csv"
    )

    shap_global: pd.DataFrame | None = None
    shap_class: pd.DataFrame | None = None
    shap_version: str | None = None
    if not args.skip_shap:
        shap_global, shap_class, shap_version = run_shap(
            rf_best, X_test, args.output_dir
        )
        shap_global.to_csv(
            args.output_dir / "shap_global_importance.csv", index=False
        )
        shap_class.to_csv(
            args.output_dir / "shap_class_importance.csv", index=False
        )

    write_excel_summary(
        args.output_dir / "phaseVI_dom_fate_ml_results.xlsx",
        class_counts,
        split_table,
        selected_features,
        removed_features,
        model_comparison,
        evaluations,
        curve_metrics,
        descriptor_summary,
        shap_global,
        shap_class,
    )

    manifest = {
        "created": datetime.now().astimezone().isoformat(),
        "script": "src/analyze_phaseVI_dom_fate_ml.py",
        "script_sha256": sha256(Path(__file__)),
        "inputs": input_manifest,
        "processed_dataset": safe_manifest_path(dataset_path),
        "config": config.__dict__,
        "features_initial": FEATURES,
        "features_retained": rf_selected,
        "class_order": CLASS_ORDER,
        "audit": audit,
        "best_parameters": {
            name: result["best_params"] for name, result in evaluations.items()
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
            "joblib": joblib.__version__,
            "shap": shap_version,
        },
        "shap_skipped": bool(args.skip_shap),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "DATA_PROCESSING_AND_RESULTS.md",
        config,
        audit,
        class_counts,
        split_counts,
        fold_features,
        selected_features,
        removed_features,
        model_comparison,
        evaluations,
        shap_global,
    )

    print(class_counts.to_string(index=False))
    print("\nSelected features:", ", ".join(rf_selected))
    print("\nModel comparison:")
    print(model_comparison.to_string(index=False))
    if shap_global is not None:
        print("\nTop SHAP descriptors:")
        print(shap_global.head(10).to_string(index=False))
    print(f"\nProcessed dataset: {dataset_path}")
    print(f"Analysis outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
