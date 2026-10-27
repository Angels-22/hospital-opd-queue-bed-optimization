"""
Preprocessing and Exploratory Data Analysis (EDA) Module.

Covers classical Data Science syllabus requirements:
- Data import and cleaning of missing values.
- Filtering, grouping, sorting, and ranking operations.
- Descriptive statistics (mean, median, variance, standard deviation, skewness, IQR).
- Pearson & Spearman correlation matrices.
- Publication-quality EDA visualizations.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Set aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def load_raw_data(file_path: str = "data/synthetic_opd.csv") -> pd.DataFrame:
    """Loads synthetic OPD patient records from CSV."""
    if not os.path.exists(file_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt_path = os.path.join(project_root, file_path)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {file_path} or {alt_path}. Run data_generator.py first.")
    df = pd.read_csv(file_path)
    return df


def clean_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Cleans missing values in clinical dataset using domain-guided imputation:
    - Temperature: imputed using median temperature conditioned on triage class.
    - Systolic BP: imputed using median systolic BP conditioned on age group.
    - Satisfaction Score: imputed using median satisfaction conditioned on triage and admission.
    - Ward Assigned: explicit 'Not Admitted' fill for patients discharged home.
    Returns the cleaned DataFrame and a report of missing values imputed.
    """
    df_clean = df.copy()
    missing_report = df_clean.isnull().sum().to_dict()

    # Impute Temperature conditioned on Triage Class
    temp_medians = df_clean.groupby("triage_class")["temperature"].transform("median")
    df_clean["temperature"] = df_clean["temperature"].fillna(temp_medians)
    # Global fallback if any remain
    df_clean["temperature"] = df_clean["temperature"].fillna(37.0)

    # Impute Systolic BP conditioned on Age Group
    bp_medians = df_clean.groupby("age_group")["systolic_bp"].transform("median")
    df_clean["systolic_bp"] = df_clean["systolic_bp"].fillna(bp_medians)
    df_clean["systolic_bp"] = df_clean["systolic_bp"].fillna(125.0)

    # Impute Satisfaction Score conditioned on Triage
    sat_medians = df_clean.groupby("triage_class")["satisfaction_score"].transform("median")
    df_clean["satisfaction_score"] = df_clean["satisfaction_score"].fillna(sat_medians).round().astype(int)

    # Fill Ward Assigned
    df_clean["ward_assigned"] = df_clean["ward_assigned"].fillna("Not Admitted")
    df_clean.loc[df_clean["admitted"] == 0, "ward_assigned"] = "Not Admitted"

    # Ensure datetime parsing
    if "arrival_datetime" in df_clean.columns:
        df_clean["arrival_datetime"] = pd.to_datetime(df_clean["arrival_datetime"])

    return df_clean, missing_report


def filter_group_sort_rank(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Demonstrates essential Data Science operations:
    - Filter: high-acuity or prolonged wait cases.
    - Group: departmental and demographic aggregations.
    - Sort: multi-column prioritization.
    - Rank: doctor volume, department turnaround, congestion by hour.
    """
    # 1. Filter: Prolonged wait (over 45 minutes)
    prolonged_waits = df[df["wait_time"] > 45.0].copy()

    # 2. Group: Department operational performance
    dept_performance = df.groupby("department").agg(
        patient_count=("patient_id", "count"),
        mean_wait_min=("wait_time", "mean"),
        median_wait_min=("wait_time", "median"),
        mean_consultation_min=("service_time", "mean"),
        admission_rate=("admitted", "mean"),
        avg_satisfaction=("satisfaction_score", "mean")
    ).reset_index()

    # 3. Group: Triage performance
    triage_performance = df.groupby("triage_class").agg(
        patient_count=("patient_id", "count"),
        mean_wait=("wait_time", "mean"),
        median_wait=("wait_time", "median"),
        std_wait=("wait_time", "std"),
        admission_rate=("admitted", "mean")
    ).reset_index()

    # 4. Sort: Prioritize by acuity (descending) and wait time (descending)
    sorted_queue_view = df.sort_values(
        by=["acuity_weight", "wait_time"],
        ascending=[False, False]
    ).head(100)

    # 5. Rank: Congestion by hour of day
    hourly_congestion = df.groupby(df["hour_of_day"].round().astype(int)).agg(
        arrival_count=("patient_id", "count"),
        avg_wait=("wait_time", "mean")
    ).reset_index()
    hourly_congestion.columns = ["hour", "arrival_count", "avg_wait"]
    hourly_congestion["congestion_rank"] = hourly_congestion["arrival_count"].rank(
        ascending=False, method="dense"
    ).astype(int)
    hourly_congestion = hourly_congestion.sort_values(by="congestion_rank")

    # 6. Rank: Doctor performance ranking
    doctor_rank = df.groupby(["doctor_id", "doctor_name", "department"]).agg(
        patients_seen=("patient_id", "count"),
        avg_service_time=("service_time", "mean"),
        avg_satisfaction=("satisfaction_score", "mean")
    ).reset_index()
    doctor_rank["volume_rank"] = doctor_rank["patients_seen"].rank(
        ascending=False, method="dense"
    ).astype(int)
    doctor_rank = doctor_rank.sort_values(by="volume_rank")

    return {
        "prolonged_waits": prolonged_waits,
        "dept_performance": dept_performance,
        "triage_performance": triage_performance,
        "sorted_queue_view": sorted_queue_view,
        "hourly_congestion": hourly_congestion,
        "doctor_rank": doctor_rank,
    }


def compute_descriptive_statistics(
    df: pd.DataFrame,
    numerical_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Computes rigorous descriptive statistics for all numeric clinical & operational variables:
    - Count, Mean, Median, Variance, Standard Deviation, Skewness, Min, Q25, Q75, Max, IQR.
    """
    if numerical_columns is None:
        numerical_columns = [
            "age",
            "systolic_bp",
            "heart_rate",
            "oxygen_saturation",
            "temperature",
            "service_time",
            "wait_time",
            "length_of_stay_hrs",
            "satisfaction_score"
        ]

    stats_list = []
    for col in numerical_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        n = len(series)
        mean_val = float(series.mean())
        median_val = float(series.median())
        var_val = float(series.var())
        std_val = float(series.std())
        skew_val = float(series.skew())
        min_val = float(series.min())
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        max_val = float(series.max())
        iqr = q75 - q25

        stats_list.append({
            "Variable": col,
            "Count": n,
            "Mean": round(mean_val, 2),
            "Median": round(median_val, 2),
            "Variance": round(var_val, 2),
            "Std_Dev": round(std_val, 2),
            "Skewness": round(skew_val, 2),
            "Min": round(min_val, 2),
            "Q25": round(q25, 2),
            "Q75": round(q75, 2),
            "Max": round(max_val, 2),
            "IQR": round(iqr, 2),
        })

    return pd.DataFrame(stats_list)


def compute_correlation_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes both Pearson (linear) and Spearman (rank-order) correlation matrices."""
    corr_cols = [
        "age",
        "systolic_bp",
        "heart_rate",
        "oxygen_saturation",
        "temperature",
        "acuity_weight",
        "service_time",
        "wait_time",
        "admitted",
        "satisfaction_score"
    ]
    sub_df = df[corr_cols].dropna()
    pearson_corr = sub_df.corr(method="pearson").round(3)
    spearman_corr = sub_df.corr(method="spearman").round(3)
    return pearson_corr, spearman_corr


def generate_eda_plots(df: pd.DataFrame, save_dir: str = "data/eda_plots") -> Dict[str, str]:
    """Generates and saves exploratory data analysis charts."""
    os.makedirs(save_dir, exist_ok=True)
    plot_paths = {}

    # 1. Hourly Arrival Distribution (Highlighting 9am-12pm Peak)
    fig, ax = plt.subplots(figsize=(9, 5))
    hourly_counts = df.groupby(df["hour_of_day"].round().astype(int))["patient_id"].count()
    hours = hourly_counts.index
    counts = hourly_counts.values
    colors = ["#e74c3c" if 9 <= h <= 12 else "#3498db" for h in hours]
    ax.bar(hours, counts, color=colors, edgecolor="black", alpha=0.85, width=0.7)
    ax.set_title("Hourly Patient Arrival Volume (Peak: 9:00 AM - 12:00 PM)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day (8:00 AM - 8:00 PM)", fontsize=11)
    ax.set_ylabel("Total Arrived Patients", fontsize=11)
    ax.set_xticks(range(8, 21))
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1 = os.path.join(save_dir, "hourly_arrivals.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    plot_paths["hourly_arrivals"] = p1

    # 2. Wait Time Distribution by Triage Class
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"Emergency": "#e74c3c", "Urgent": "#f39c12", "Routine": "#2ecc71"}
    sns.boxplot(x="triage_class", y="wait_time", data=df, order=["Emergency", "Urgent", "Routine"],
                hue="triage_class", palette=palette, ax=ax, width=0.5, fliersize=2, legend=False)
    ax.set_title("OPD Wait Time Distribution Across Triage Categories", fontsize=13, fontweight="bold")
    ax.set_xlabel("Triage Priority Level", fontsize=11)
    ax.set_ylabel("Waiting Time (Minutes)", fontsize=11)
    plt.tight_layout()
    p2 = os.path.join(save_dir, "wait_time_by_triage.png")
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    plot_paths["wait_time_by_triage"] = p2

    # 3. Pearson Correlation Heatmap
    p_corr, _ = compute_correlation_matrices(df)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(p_corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                fmt=".2f", square=True, linewidths=0.5, ax=ax)
    ax.set_title("Pearson Correlation Matrix (Clinical & Operational Features)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p3 = os.path.join(save_dir, "correlation_heatmap.png")
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    plot_paths["correlation_heatmap"] = p3

    # 4. Admission Rate by Demographic Group (Age & Area)
    fig, ax = plt.subplots(figsize=(8, 5))
    demo_admit = df.groupby(["age_group", "area"])["admitted"].mean().reset_index()
    demo_admit["admit_pct"] = demo_admit["admitted"] * 100.0
    sns.barplot(data=demo_admit, x="age_group", y="admit_pct", hue="area",
                order=["<18", "18-40", "41-65", ">65"], palette=["#2980b9", "#16a085"], ax=ax)
    ax.set_title("Admission Rate (%) by Age Group and Residential Area", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age Demographic Group", fontsize=11)
    ax.set_ylabel("Admission Rate (%)", fontsize=11)
    ax.legend(title="Area")
    plt.tight_layout()
    p4 = os.path.join(save_dir, "admission_demographics.png")
    fig.savefig(p4, dpi=200)
    plt.close(fig)
    plot_paths["admission_demographics"] = p4

    return plot_paths


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df, missing_rep = clean_missing_values(raw_df)
    print("Cleaned Missing Values Report:", missing_rep)
    desc_stats = compute_descriptive_statistics(cleaned_df)
    print("\nDescriptive Statistics:\n", desc_stats.to_string(index=False))
    plots = generate_eda_plots(cleaned_df)
    print("\nGenerated EDA Plots:", plots)
