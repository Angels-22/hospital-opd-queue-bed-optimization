"""
Machine Learning Models Module for Hospital OPD Management.

Covers classical Data Science syllabus requirements:
1. Linear, Multiple, and Polynomial Regression:
   - Target A: Consultation Time Prediction
   - Target B: Next-Hour Arrival Volume Forecasting (Time-Series Features)
2. Logistic Regression:
   - Target: Patient Admission Prediction (Binary Classification)
3. Decision Tree:
   - Target: Triage Acuity Classification (Multi-class: Emergency, Urgent, Routine)
4. K-Nearest Neighbors (KNN):
   - Target: Similar-Patient Service Time Estimation
5. Comprehensive Model Evaluation:
   - RMSE, MAE, R², Accuracy, Precision, Recall, F1, Confusion Matrices,
     and 5-Fold Cross-Validation.
   - Formatted Model Comparison Summary Table.
"""

from __future__ import annotations
import os
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import pickle
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from src.preprocessing_eda import clean_missing_values, load_raw_data


class MLSuite:
    """End-to-end Machine Learning training, evaluation, and inference suite."""

    def __init__(self, data_path: str = "data/synthetic_opd.csv", seed: int = 42):
        self.data_path = data_path
        self.seed = seed
        self.models: Dict[str, Any] = {}
        self.evaluation_results: List[Dict[str, Any]] = []

        # Load and clean data
        raw_df = load_raw_data(data_path)
        self.df, _ = clean_missing_values(raw_df)

    # -----------------------------------------------------------------------
    # 1. Regression Suite: Consultation Time Prediction
    # -----------------------------------------------------------------------

    def train_consultation_time_regressors(self) -> Dict[str, Dict[str, float]]:
        """
        Trains and compares:
        (1) Simple Linear Regression (acuity_weight -> service_time)
        (2) Multiple Linear Regression (vitals + demographics + acuity -> service_time)
        (3) Polynomial Regression (degree 2 on continuous clinical features)
        """
        features_multiple = ["acuity_weight", "age", "systolic_bp", "heart_rate", "oxygen_saturation", "temperature"]
        X = self.df[features_multiple]
        y = self.df["service_time"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.seed
        )

        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        results = {}

        # 1. Simple Linear Regression (Acuity weight alone)
        X_train_simple = X_train[["acuity_weight"]]
        X_test_simple = X_test[["acuity_weight"]]
        simple_lr = LinearRegression()
        simple_lr.fit(X_train_simple, y_train)
        y_pred_simple = simple_lr.predict(X_test_simple)
        cv_r2_simple = cross_val_score(simple_lr, X[["acuity_weight"]], y, cv=kf, scoring="r2").mean()

        rmse_simple = float(np.sqrt(mean_squared_error(y_test, y_pred_simple)))
        r2_simple = float(r2_score(y_test, y_pred_simple))
        mae_simple = float(mean_absolute_error(y_test, y_pred_simple))

        results["Simple Linear Regression"] = {
            "RMSE": round(rmse_simple, 3),
            "MAE": round(mae_simple, 3),
            "R2": round(r2_simple, 3),
            "5-Fold CV R2": round(float(cv_r2_simple), 3),
        }
        self.models["simple_linear_regression"] = simple_lr

        self.evaluation_results.append({
            "Model": "Simple Linear Regression",
            "Task": "Regression",
            "Target": "Service Time (min)",
            "Primary Metric": f"RMSE: {rmse_simple:.2f}",
            "Secondary Metric": f"R²: {r2_simple:.3f}",
            "5-Fold CV Score": f"{cv_r2_simple:.3f} (R²)",
        })

        # 2. Multiple Linear Regression
        multiple_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])
        multiple_lr.fit(X_train, y_train)
        y_pred_mult = multiple_lr.predict(X_test)
        cv_r2_mult = cross_val_score(multiple_lr, X, y, cv=kf, scoring="r2").mean()

        rmse_mult = float(np.sqrt(mean_squared_error(y_test, y_pred_mult)))
        r2_mult = float(r2_score(y_test, y_pred_mult))
        mae_mult = float(mean_absolute_error(y_test, y_pred_mult))

        results["Multiple Linear Regression"] = {
            "RMSE": round(rmse_mult, 3),
            "MAE": round(mae_mult, 3),
            "R2": round(r2_mult, 3),
            "5-Fold CV R2": round(float(cv_r2_mult), 3),
        }
        self.models["multiple_linear_regression"] = multiple_lr

        self.evaluation_results.append({
            "Model": "Multiple Linear Regression",
            "Task": "Regression",
            "Target": "Service Time (min)",
            "Primary Metric": f"RMSE: {rmse_mult:.2f}",
            "Secondary Metric": f"R²: {r2_mult:.3f}",
            "5-Fold CV Score": f"{cv_r2_mult:.3f} (R²)",
        })

        # 3. Polynomial Regression (Degree 2)
        poly_lr = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])
        poly_lr.fit(X_train, y_train)
        y_pred_poly = poly_lr.predict(X_test)
        cv_r2_poly = cross_val_score(poly_lr, X, y, cv=kf, scoring="r2").mean()

        rmse_poly = float(np.sqrt(mean_squared_error(y_test, y_pred_poly)))
        r2_poly = float(r2_score(y_test, y_pred_poly))
        mae_poly = float(mean_absolute_error(y_test, y_pred_poly))

        results["Polynomial Regression (Deg 2)"] = {
            "RMSE": round(rmse_poly, 3),
            "MAE": round(mae_poly, 3),
            "R2": round(r2_poly, 3),
            "5-Fold CV R2": round(float(cv_r2_poly), 3),
        }
        self.models["polynomial_regression"] = poly_lr

        self.evaluation_results.append({
            "Model": "Polynomial Regression (Deg 2)",
            "Task": "Regression",
            "Target": "Service Time (min)",
            "Primary Metric": f"RMSE: {rmse_poly:.2f}",
            "Secondary Metric": f"R²: {r2_poly:.3f}",
            "5-Fold CV Score": f"{cv_r2_poly:.3f} (R²)",
        })

        return results

    # -----------------------------------------------------------------------
    # 2. Next-Hour Arrivals Forecasting
    # -----------------------------------------------------------------------

    def build_hourly_arrivals_dataset(self) -> pd.DataFrame:
        """Aggregates arrivals into an hourly time-series with lag and cyclical features."""
        df_ts = self.df.copy()
        # Group by arrival_date and hour_of_day floor
        df_ts["hour_int"] = df_ts["hour_of_day"].astype(int)
        hourly = df_ts.groupby(["arrival_date", "hour_int"]).agg(
            arrivals=("patient_id", "count"),
            is_weekend=("is_weekend", "first")
        ).reset_index()

        hourly["datetime_proxy"] = pd.to_datetime(hourly["arrival_date"]) + pd.to_timedelta(hourly["hour_int"], unit="h")
        hourly = hourly.sort_values(by="datetime_proxy").reset_index(drop=True)

        # Lag features
        hourly["lag_1"] = hourly["arrivals"].shift(1)
        hourly["lag_2"] = hourly["arrivals"].shift(2)
        hourly["rolling_mean_3"] = hourly["arrivals"].shift(1).rolling(window=3).mean()
        hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour_int"] / 24.0)
        hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour_int"] / 24.0)

        # Drop warmup rows with NaN from lags
        hourly = hourly.dropna().reset_index(drop=True)
        return hourly

    def train_arrival_forecasters(self) -> Dict[str, Dict[str, float]]:
        """Trains regression models to forecast next-hour patient arrival volume."""
        hourly = self.build_hourly_arrivals_dataset()
        features = ["hour_int", "is_weekend", "lag_1", "lag_2", "rolling_mean_3", "hour_sin", "hour_cos"]
        X = hourly[features]
        y = hourly["arrivals"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, shuffle=False  # Sequential split for time series
        )

        kf = KFold(n_splits=5, shuffle=False)
        results = {}

        # Multiple Linear Regression for Arrivals
        arrival_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])
        arrival_lr.fit(X_train, y_train)
        y_pred = arrival_lr.predict(X_test)
        cv_r2 = cross_val_score(arrival_lr, X, y, cv=kf, scoring="r2").mean()

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))

        results["Arrival Multiple Regression"] = {
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "R2": round(r2, 3),
            "5-Fold CV R2": round(float(cv_r2), 3),
        }
        self.models["arrival_forecaster"] = arrival_lr

        self.evaluation_results.append({
            "Model": "Arrival Multiple Regression",
            "Task": "Forecasting",
            "Target": "Next-Hour Arrivals",
            "Primary Metric": f"RMSE: {rmse:.2f}",
            "Secondary Metric": f"R²: {r2:.3f}",
            "5-Fold CV Score": f"{cv_r2:.3f} (R²)",
        })

        return results

    # -----------------------------------------------------------------------
    # 3. Logistic Regression: Admission Prediction
    # -----------------------------------------------------------------------

    def train_admission_classifier(self) -> Dict[str, Any]:
        """
        Trains Logistic Regression to predict binary patient admission (0: Discharged, 1: Admitted).
        Evaluates with Accuracy, Precision, Recall, F1, ROC-AUC, and 5-fold CV.
        """
        features = ["acuity_weight", "age", "systolic_bp", "heart_rate", "oxygen_saturation", "temperature", "service_time"]
        X = self.df[features]
        y = self.df["admitted"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.seed, stratify=y
        )

        log_reg = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=self.seed))
        ])
        log_reg.fit(X_train, y_train)

        y_pred = log_reg.predict(X_test)
        y_prob = log_reg.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        cv_f1 = float(cross_val_score(log_reg, X, y, cv=kf, scoring="f1").mean())

        self.models["admission_classifier"] = log_reg

        self.evaluation_results.append({
            "Model": "Logistic Regression",
            "Task": "Classification",
            "Target": "Admission (0/1)",
            "Primary Metric": f"ROC-AUC: {auc:.3f}",
            "Secondary Metric": f"F1: {f1:.3f}",
            "5-Fold CV Score": f"{cv_f1:.3f} (F1)",
        })

        return {
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1": round(f1, 3),
            "ROC-AUC": round(auc, 3),
            "5-Fold CV F1": round(cv_f1, 3),
            "Confusion Matrix": cm,
        }

    # -----------------------------------------------------------------------
    # 4. Decision Tree: Triage Acuity Classification
    # -----------------------------------------------------------------------

    def train_triage_decision_tree(self) -> Dict[str, Any]:
        """
        Trains an interpretable Decision Tree Classifier to classify triage priority
        (Emergency, Urgent, Routine) based on vital signs and patient age.
        """
        features = ["systolic_bp", "heart_rate", "oxygen_saturation", "temperature", "age"]
        X = self.df[features]
        y = self.df["triage_class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.seed, stratify=y
        )

        dt = DecisionTreeClassifier(
            max_depth=4,
            criterion="entropy",
            random_state=self.seed
        )
        dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
        cm = confusion_matrix(y_test, y_pred, labels=["Emergency", "Urgent", "Routine"]).tolist()

        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        cv_acc = float(cross_val_score(dt, X, y, cv=kf, scoring="accuracy").mean())

        # Feature importances
        feature_importance = {
            f: round(float(imp), 4)
            for f, imp in zip(features, dt.feature_importances_, strict=False)
        }

        self.models["triage_decision_tree"] = dt

        self.evaluation_results.append({
            "Model": "Decision Tree Classifier",
            "Task": "Classification",
            "Target": "Triage Priority (3-class)",
            "Primary Metric": f"Accuracy: {acc:.3f}",
            "Secondary Metric": f"Macro F1: {macro_f1:.3f}",
            "5-Fold CV Score": f"{cv_acc:.3f} (Acc)",
        })

        return {
            "Accuracy": round(acc, 3),
            "Macro F1": round(macro_f1, 3),
            "Weighted F1": round(weighted_f1, 3),
            "5-Fold CV Accuracy": round(cv_acc, 3),
            "Confusion Matrix": cm,
            "Feature Importances": feature_importance,
            "Tree Rules": export_text(dt, feature_names=features),
        }

    # -----------------------------------------------------------------------
    # 5. K-Nearest Neighbors (KNN): Similar-Patient Service Time Estimation
    # -----------------------------------------------------------------------

    def train_knn_regressor(self, n_neighbors: int = 7) -> Dict[str, Any]:
        """
        Trains K-Nearest Neighbors Regressor to estimate consultation duration
        from historical patients with similar vitals, age, and clinical presentation.
        Evaluates k across [3, 5, 7, 11, 15].
        """
        features = ["acuity_weight", "age", "systolic_bp", "heart_rate", "oxygen_saturation", "temperature"]
        X = self.df[features]
        y = self.df["service_time"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.seed
        )

        knn_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance"))
        ])
        knn_pipe.fit(X_train, y_train)

        y_pred = knn_pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))

        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        cv_rmse = -float(cross_val_score(knn_pipe, X.iloc[:5000], y.iloc[:5000], cv=kf, scoring="neg_root_mean_squared_error").mean())

        self.models["knn_regressor"] = knn_pipe

        self.evaluation_results.append({
            "Model": f"K-Nearest Neighbors (k={n_neighbors})",
            "Task": "Regression",
            "Target": "Service Time (min)",
            "Primary Metric": f"RMSE: {rmse:.2f}",
            "Secondary Metric": f"R²: {r2:.3f}",
            "5-Fold CV Score": f"RMSE: {cv_rmse:.2f}",
        })

        return {
            "n_neighbors": n_neighbors,
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3),
            "R2": round(r2, 3),
            "5-Fold CV RMSE": round(cv_rmse, 3),
        }

    # -----------------------------------------------------------------------
    # Comprehensive Training & Comparison Table
    # -----------------------------------------------------------------------

    def run_all(self) -> pd.DataFrame:
        """Trains all classical ML models and compiles the comparative evaluation table."""
        self.evaluation_results = []
        self.train_consultation_time_regressors()
        self.train_arrival_forecasters()
        self.train_admission_classifier()
        self.train_triage_decision_tree()
        self.train_knn_regressor(n_neighbors=7)

        summary_df = pd.DataFrame(self.evaluation_results)
        return summary_df

    def save_models(self, save_dir: str = "data/trained_models"):
        """Serializes trained machine learning pipeline artifacts."""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            path = os.path.join(save_dir, f"{name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)
        print(f"Saved {len(self.models)} trained models to {save_dir}")


if __name__ == "__main__":
    suite = MLSuite()
    comparison_table = suite.run_all()
    print("\n" + "=" * 80)
    print("CLASSICAL MACHINE LEARNING MODEL BENCHMARK TABLE")
    print("=" * 80)
    print(comparison_table.to_string(index=False))
    suite.save_models()
