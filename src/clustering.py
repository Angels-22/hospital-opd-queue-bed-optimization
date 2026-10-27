"""
Clustering and Demand Regime Detection Module.

Covers classical Data Science syllabus requirements:
1. K-Means Clustering:
   - Patient Type Clustering (Clinical Phenotyping)
   - Elbow Method (Inertia SSE) and Silhouette Score analysis
2. Agglomerative Hierarchical Clustering:
   - Hourly/Daily Demand Regime Detection (Normal, Peak, Surge)
   - Dendrogram Generation (Ward linkage)
3. Innovation 3 Integration:
   - Classifies operational regimes in real-time to drive regime-aware
     dynamic doctor staffing and tunable alpha priority policies.
"""

from __future__ import annotations
import os
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing_eda import clean_missing_values, load_raw_data


class OPDClusteringAnalysis:
    """Performs patient phenotyping and hourly demand regime clustering."""

    def __init__(self, data_path: str = "data/synthetic_opd.csv", seed: int = 42):
        self.data_path = data_path
        self.seed = seed

        raw_df = load_raw_data(data_path)
        self.df, _ = clean_missing_values(raw_df)

        self.scaler_patient = StandardScaler()
        self.scaler_regime = StandardScaler()

        self.patient_kmeans: KMeans | None = None
        self.regime_kmeans: KMeans | None = None
        self.regime_agglomerative: AgglomerativeClustering | None = None

        self.hourly_df: pd.DataFrame | None = None
        self.regime_centers: pd.DataFrame | None = None

    # -----------------------------------------------------------------------
    # 1. Patient Phenotype Clustering (K-Means)
    # -----------------------------------------------------------------------

    def analyze_patient_clusters(
        self,
        k_range: range = range(2, 7)
    ) -> Tuple[Dict[str, List[float]], int, pd.DataFrame]:
        """
        Evaluates K-Means clustering on clinical patient features across multiple k values.
        Computes Inertia (Elbow method) and Silhouette Scores.
        Returns evaluation metrics, optimal k, and cluster summary profiles.
        """
        features = ["age", "systolic_bp", "heart_rate", "oxygen_saturation", "temperature", "acuity_weight", "service_time"]
        # Use a representative sample of 10,000 for fast silhouette calculation
        sample_df = self.df[features].sample(n=min(10000, len(self.df)), random_state=self.seed)
        X_scaled = self.scaler_patient.fit_transform(sample_df)

        inertias = []
        silhouettes = []
        k_values = list(k_range)

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(float(km.inertia_))
            sil = float(silhouette_score(X_scaled, labels))
            silhouettes.append(sil)

        best_k = k_values[int(np.argmax(silhouettes))]
        # Fit optimal K-Means on full dataset
        X_full = self.scaler_patient.fit_transform(self.df[features])
        self.patient_kmeans = KMeans(n_clusters=best_k, random_state=self.seed, n_init=10)
        self.df["patient_cluster"] = self.patient_kmeans.fit_predict(X_full)

        # Profile clusters
        cluster_profiles = self.df.groupby("patient_cluster")[features + ["wait_time", "admitted"]].mean().round(2)

        metrics = {
            "k_values": k_values,
            "inertias": inertias,
            "silhouettes": silhouettes,
        }
        return metrics, best_k, cluster_profiles

    # -----------------------------------------------------------------------
    # 2. Demand Regime Clustering (Normal, Peak, Surge)
    # -----------------------------------------------------------------------

    def build_hourly_demand_summary(self) -> pd.DataFrame:
        """Aggregates records into an hourly operational summary."""
        df_copy = self.df.copy()
        df_copy["hour_int"] = df_copy["hour_of_day"].astype(int)

        hourly = df_copy.groupby(["arrival_date", "hour_int"]).agg(
            arrival_count=("patient_id", "count"),
            emergency_count=("triage_class", lambda s: (s == "Emergency").sum()),
            urgent_count=("triage_class", lambda s: (s == "Urgent").sum()),
            routine_count=("triage_class", lambda s: (s == "Routine").sum()),
            avg_wait_min=("wait_time", "mean"),
            avg_service_min=("service_time", "mean"),
            admit_count=("admitted", "sum")
        ).reset_index()

        hourly["emergency_ratio"] = hourly["emergency_count"] / hourly["arrival_count"]
        hourly["urgent_ratio"] = hourly["urgent_count"] / hourly["arrival_count"]
        hourly["admission_rate"] = hourly["admit_count"] / hourly["arrival_count"]

        self.hourly_df = hourly
        return hourly

    def cluster_demand_regimes(
        self,
        n_clusters: int = 3
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Clusters hourly operational demand using both:
        (1) K-Means with Elbow & Silhouette evaluation
        (2) Agglomerative Hierarchical Clustering (Ward linkage)
        Labels clusters into: Normal, Peak, and Surge regimes.
        """
        if self.hourly_df is None:
            self.build_hourly_demand_summary()

        regime_features = ["arrival_count", "emergency_ratio", "avg_wait_min", "admission_rate"]
        X = self.hourly_df[regime_features]
        X_scaled = self.scaler_regime.fit_transform(X)

        # Evaluate K-means elbow and silhouette for k=2..5
        k_vals = [2, 3, 4, 5]
        inertias = []
        silhouettes = []
        for k in k_vals:
            km_temp = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
            labs = km_temp.fit_predict(X_scaled)
            inertias.append(float(km_temp.inertia_))
            silhouettes.append(float(silhouette_score(X_scaled, labs)))

        # K-Means clustering with n_clusters=3
        self.regime_kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        km_labels = self.regime_kmeans.fit_predict(X_scaled)

        # Agglomerative Hierarchical Clustering with Ward Linkage
        self.regime_agglomerative = AgglomerativeClustering(
            n_clusters=n_clusters, metric="euclidean", linkage="ward"
        )
        agg_labels = self.regime_agglomerative.fit_predict(X_scaled)

        self.hourly_df["kmeans_regime_id"] = km_labels
        self.hourly_df["agg_regime_id"] = agg_labels

        # Label regimes systematically by ascending arrival volume:
        # Cluster with lowest arrivals = Normal
        # Cluster with medium arrivals = Peak
        # Cluster with highest arrivals = Surge
        arrival_means = self.hourly_df.groupby("kmeans_regime_id")["arrival_count"].mean()
        ordered_ids = arrival_means.sort_values().index.tolist()
        label_map = {
            ordered_ids[0]: "Normal",
            ordered_ids[1]: "Peak",
            ordered_ids[2]: "Surge"
        }
        self.hourly_df["regime_name"] = self.hourly_df["kmeans_regime_id"].map(label_map)

        # Summary profile of each demand regime
        regime_summary = self.hourly_df.groupby("regime_name").agg(
            hours_count=("arrival_count", "count"),
            mean_arrivals_per_hr=("arrival_count", "mean"),
            mean_wait_min=("avg_wait_min", "mean"),
            emergency_pct=("emergency_ratio", lambda s: round(s.mean() * 100, 2)),
            admission_pct=("admission_rate", lambda s: round(s.mean() * 100, 2)),
        ).loc[["Normal", "Peak", "Surge"]].round(2)

        self.regime_centers = regime_summary

        metrics = {
            "k_values": k_vals,
            "inertias": inertias,
            "silhouettes": silhouettes,
            "linkage_matrix": linkage(X_scaled, method="ward"),
        }

        return metrics, regime_summary

    def classify_current_regime(self, arrivals_per_hour: float, avg_wait: float) -> str:
        """Classifies incoming simulation hour into Normal, Peak, or Surge regime."""
        if self.regime_centers is None:
            if arrivals_per_hour >= 42.0 or avg_wait >= 40.0:
                return "Surge"
            elif arrivals_per_hour >= 28.0 or avg_wait >= 22.0:
                return "Peak"
            else:
                return "Normal"

        norm_arr = self.regime_centers.loc["Normal", "mean_arrivals_per_hr"]
        peak_arr = self.regime_centers.loc["Peak", "mean_arrivals_per_hr"]
        surge_arr = self.regime_centers.loc["Surge", "mean_arrivals_per_hr"]

        # Nearest center
        dists = {
            "Normal": abs(arrivals_per_hour - norm_arr),
            "Peak": abs(arrivals_per_hour - peak_arr),
            "Surge": abs(arrivals_per_hour - surge_arr),
        }
        return min(dists, key=dists.get)

    # -----------------------------------------------------------------------
    # 3. Plotting: Dendrogram, Elbow, and Silhouette
    # -----------------------------------------------------------------------

    def generate_clustering_plots(self, save_dir: str = "data/clustering_plots") -> Dict[str, str]:
        """Generates and saves dendrogram, elbow SSE, and silhouette diagnostic plots."""
        os.makedirs(save_dir, exist_ok=True)
        plot_paths = {}

        metrics, _ = self.cluster_demand_regimes(n_clusters=3)

        # 1. Elbow & Silhouette Curves Side-by-Side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(metrics["k_values"], metrics["inertias"], marker="o", color="#2980b9", lw=2)
        ax1.set_title("K-Means Elbow Method (Inertia SSE)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
        ax1.set_ylabel("Inertia (Sum of Squared Errors)", fontsize=11)
        ax1.set_xticks(metrics["k_values"])
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2.plot(metrics["k_values"], metrics["silhouettes"], marker="s", color="#27ae60", lw=2)
        ax2.set_title("Silhouette Score vs Number of Clusters", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
        ax2.set_ylabel("Silhouette Coefficient", fontsize=11)
        ax2.set_xticks(metrics["k_values"])
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        p1 = os.path.join(save_dir, "elbow_silhouette_analysis.png")
        fig.savefig(p1, dpi=200)
        plt.close(fig)
        plot_paths["elbow_silhouette"] = p1

        # 2. Agglomerative Hierarchical Dendrogram
        fig, ax = plt.subplots(figsize=(10, 6))
        Z = metrics["linkage_matrix"]
        dendrogram(
            Z,
            truncate_mode="lastp",
            p=25,
            show_leaf_counts=True,
            ax=ax,
            color_threshold=Z[-2, 2],
            above_threshold_color="#7f8c8d"
        )
        ax.set_title("Agglomerative Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Hourly Demand Sample Clusters", fontsize=11)
        ax.set_ylabel("Ward Distance / Dissimilarity", fontsize=11)
        plt.tight_layout()
        p2 = os.path.join(save_dir, "regime_dendrogram.png")
        fig.savefig(p2, dpi=200)
        plt.close(fig)
        plot_paths["dendrogram"] = p2

        # 3. Demand Regimes Scatter (Arrivals vs Wait Time)
        fig, ax = plt.subplots(figsize=(8, 6))
        palette = {"Normal": "#2ecc71", "Peak": "#f39c12", "Surge": "#e74c3c"}
        for regime, group in self.hourly_df.groupby("regime_name"):
            ax.scatter(group["arrival_count"], group["avg_wait_min"],
                       label=regime, color=palette[regime], alpha=0.65, s=40, edgecolors="none")
        ax.set_title("Operational Demand Regimes: Arrivals vs Average Wait Time", fontsize=13, fontweight="bold")
        ax.set_xlabel("Patient Arrivals per Hour", fontsize=11)
        ax.set_ylabel("Average OPD Wait Time (Minutes)", fontsize=11)
        ax.legend(title="Regime")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p3 = os.path.join(save_dir, "demand_regimes_scatter.png")
        fig.savefig(p3, dpi=200)
        plt.close(fig)
        plot_paths["regimes_scatter"] = p3

        return plot_paths


if __name__ == "__main__":
    analyzer = OPDClusteringAnalysis()
    print("Running Patient Phenotype Clustering...")
    p_metrics, p_k, p_profiles = analyzer.analyze_patient_clusters()
    print(f"Optimal Patient Clusters (k={p_k}):\n", p_profiles)

    print("\nRunning Demand Regime Clustering (Normal, Peak, Surge)...")
    r_metrics, r_summary = analyzer.cluster_demand_regimes(n_clusters=3)
    print("Demand Regime Profiles:\n", r_summary)

    plots = analyzer.generate_clustering_plots()
    print("\nGenerated Clustering Plots:", plots)
