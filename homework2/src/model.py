"""
Model stencil for Homework 2: Ensemble Methods with Gradient Boosting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, Union

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

# Set plotting style
sns.set_style("whitegrid")


class GradientBoostingModel:
    """Gradient Boosting model implementation with comprehensive evaluation and analysis tools"""

    def __init__(
        self,
        task: str = "classification",
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_estimators: int = 50,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: int = 42,
        use_scaler: bool = False,
    ):
        """
        Initialize Gradient Boosting model with customizable parameters

        Args:
            task: 'classification' or 'regression'
            max_depth: Maximum depth of a tree (controls pruning)
            learning_rate: Step size shrinkage to prevent overfitting
            n_estimators: Number of boosting rounds/trees
            subsample: Subsample ratio of training instances
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            max_features: Number of features to consider when looking for the best split
            random_state: Random seed for reproducibility
            use_scaler: Whether to apply StandardScaler before training/prediction
        """
        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
        }

        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")

        self.model = None
        self.feature_names = None
        self.task = task
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None

        self.model = GradientBoostingClassifier(**self.params) if self.task == "classification" else GradientBoostingRegressor(**self.params)

    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Split data into training and testing sets

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test: Split datasets
        """
        # TODO: Implement train/test split and track feature names
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if self.task == "classification" else None,
        )
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True):
        """
        Train the Gradient Boosting model

        Args:
            X_train: Training features
            y_train: Training targets
            verbose: Whether to print training progress

        Returns:
            self: Trained model instance
        """
        # TODO: Create classifier/regressor based on task and fit it
        X_fit = X_train
        if self.use_scaler:
            if self.scaler is None:
                self.scaler = StandardScaler()
            X_fit = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns,
            )

        self.model.fit(X_fit, y_train)

    def predict(
        self, X: pd.DataFrame, return_proba: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions with the trained model

        Args:
            X: Feature data for prediction
            return_proba: If True and model is a classifier, return probability estimates

        Returns:
            Predictions or probability estimates
        """
        # TODO: Apply scaler when enabled, then predict
        X_in = X
        if self.use_scaler:
            if self.scaler is None:
                raise ValueError("Scaler is enabled but not fitted. Train the model first.")
            X_in = pd.DataFrame(
                self.scaler.transform(X),
                index=X.index,
                columns=X.columns,
            )

        if self.task == "classification" and return_proba:
            proba = self.model.predict_proba(X_in)
            # Return a DataFrame with class-labeled columns (nice for debugging)
            return pd.DataFrame(proba, columns=[f"p(class={c})" for c in self.model.classes_], index=X.index)

        return self.model.predict(X_in)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance on test data

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """

        # TODO: Compute metrics (classification vs regression)
        if self.task == "classification":
            y_pred = self.predict(X_test)
            # For ROC-AUC, prefer probabilities if available
            y_proba_df = self.predict(X_test, return_proba=True)
            y_proba = y_proba_df.to_numpy()

            # Handle binary vs multiclass ROC-AUC safely
            classes = getattr(self.model, "classes_", None)
            unique = np.unique(y_test)

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0, average="binary" if len(unique) == 2 else "macro")),
                "recall": float(recall_score(y_test, y_pred, zero_division=0, average="binary" if len(unique) == 2 else "macro")),
                "f1": float(f1_score(y_test, y_pred, zero_division=0, average="binary" if len(unique) == 2 else "macro")),
                "roc_auc": None,
            }

            try:
                if len(unique) == 2:
                    # pick proba of the "positive" class (assume second class in classes_)
                    if classes is not None and len(classes) == 2:
                        pos_idx = 1
                    else:
                        pos_idx = 1
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, pos_idx]))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
            except Exception:
                metrics["roc_auc"] = None

            return metrics

        # Regression
        y_pred = self.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> Dict:
        """
        Perform cross-validation

        Args:
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds

        Returns:
            Dictionary of cross-validation results using sklearn cross_val_score
        """
        # TODO: Use Pipeline when scaling, and choose classifier/regressor based on task
        model = GradientBoostingClassifier(**self.params) if self.task == "classification" else GradientBoostingRegressor(**self.params)

        if self.use_scaler:
            estimator = Pipeline([("scaler", StandardScaler()), ("model", model)])
        else:
            estimator = model

        # TODO: Choose scoring metrics based on classification vs regression
        if self.task == "classification":
            scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        else:
            scoring = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]

        results = {}
        # TODO: Get mean, stdev of cross_val_score for each metric
        for metric in scoring:
            scores = cross_val_score(estimator, X, y, cv=cv, scoring=metric)

            if self.task == "regression":
                # Convert negative metrics to positive + convert MSE to RMSE
                if metric == "neg_mean_squared_error":
                    vals = np.sqrt(-scores)
                    key = "rmse"
                elif metric == "neg_mean_absolute_error":
                    vals = -scores
                    key = "mae"
                else:
                    vals = scores
                    key = "r2"
            else:
                vals = scores
                key = metric

            results[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        return results

    def get_feature_importance(
        self, plot: bool = False, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importances

        Returns:
            DataFrame with feature importances
        """

        # TODO: Optionally plot a bar chart of top_n feature importances
        if self.model is None:
            raise ValueError("Model is not trained yet. Call .fit() first.")
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("This model does not expose feature_importances_.")

        names = self.feature_names
        if names is None:
            # fallback
            names = [f"f{i}" for i in range(len(self.model.feature_importances_))]

        df = pd.DataFrame(
            {"feature": names, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)

        if plot:
            top = df.head(top_n).iloc[::-1]  # reverse for nicer horizontal plot
            plt.figure(figsize=(10, max(4, 0.35 * len(top))))
            sns.barplot(data=top, x="importance", y="feature")
            plt.title(f"Top {min(top_n, len(df))} Feature Importances")
            plt.tight_layout()
            plt.show()

        return df

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict,
        cv: int = 3,
        scoring: str = "roc_auc",
    ) -> Dict:
        """
        Perform grid search for hyperparameter tuning

        Args:
            X: Feature data
            y: Target data
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            scoring: Scoring metric to evaluate

        Returns:
            Dictionary with best parameters and results
        """
        # TODO: Choose classifier or regressor based on task
        model = GradientBoostingClassifier(**self.params) if self.task == "classification" else GradientBoostingRegressor(**self.params)

        if self.use_scaler:
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
            # If user gave raw GBM params, map them to pipeline namespace
            mapped_grid = {}
            for k, v in param_grid.items():
                mapped_grid[f"model__{k}" if not k.startswith("model__") else k] = v
            estimator = pipeline
            grid = mapped_grid
        else:
            estimator = model
            grid = param_grid

        # TODO: Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        grid_search.fit(X, y)

        # TODO: Perform grid search for hyperparameter tuning
        self.model = grid_search.best_estimator_
        if self.use_scaler and isinstance(self.model, Pipeline):
            self.scaler = self.model.named_steps["scaler"]

        return {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "cv_results": grid_search.cv_results_,
            "best_estimator": grid_search.best_estimator_,
        }

    def plot_tree(
        self, tree_index: int = 0, figsize: Tuple[int, int] = (20, 15)
    ) -> None:
        """
        Plot a specific tree from the ensemble

        Args:
            tree_index: Index of the tree to plot
            figsize: Figure size for the plot
        """
        tree = self.model.estimators_[tree_index][0]

        plt.figure(figsize=figsize)

        plot_tree(
            tree,
            feature_names=self.feature_names,
            filled=True,
        )

        plt.title(f"Tree {tree_index}")
        plt.show()
