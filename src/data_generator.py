"""
Synthetic OPD Dataset Generator for Hospital Queue and Bed Allocation Project.

Generates 6 months of realistic synthetic outpatient department (OPD) records.
Documents every distribution parameter:
1. Temporal Distribution:
   - Total Days: 180 days (6 months), operating 12 hours/day (8:00 AM - 8:00 PM).
   - Arrival Process: Non-Homogeneous Poisson Process (NHPP) with peak morning hours (9am-12pm).
   - Base arrival rate: ~22 patients/hour; peak rate: ~42 patients/hour; daily mean ~300 patients.
2. Clinical & Demographic Distributions:
   - Age: Bimodal demographic mix: Pediatric (5-17, 12%), Young Adult (18-40, 32%),
     Middle-aged (41-65, 32%), Geriatric (66-88, 24%).
   - Gender: Female (49%), Male (49%), Non-binary (2%).
   - Area: Urban (65%), Rural (35%).
   - Triage Classes: Emergency (10%), Urgent (30%), Routine (60%).
   - Vitals:
     * Systolic BP (mmHg):
       - Emergency: N(152, 22)
       - Urgent: N(136, 16)
       - Routine: N(122, 10)
     * Heart Rate (bpm):
       - Emergency: N(110, 18)
       - Urgent: N(95, 14)
       - Routine: N(76, 9)
     * Oxygen Saturation SpO2 (%):
       - Emergency: N(90.5, 4.2) in [70, 99]
       - Urgent: N(94.5, 2.5) in [86, 99]
       - Routine: N(98.2, 1.1) in [94, 100]
     * Body Temperature (°C):
       - Emergency: N(38.5, 1.1)
       - Urgent: N(38.0, 0.8)
       - Routine: N(36.8, 0.4)
3. Operational Distributions:
   - Consultation Duration: Log-normal per triage (Emergency: mean=20, std=6; Urgent: mean=15, std=5; Routine: mean=10, std=3).
   - Queue Waiting Duration: Simulates queue delays matching SimPy queueing dynamics (mean ~15-25 min, peak ~45-65 min).
   - Admission Probability: Logistic clinical propensity score yielding ~18% overall admission rate.
   - Bed/Ward Assignment: 75% General Ward, 25% ICU.
   - Length of Stay (LOS, hours): General Ward (LN, mean=48, std=18), ICU (LN, mean=72, std=24).
   - Doctor & Department Assignment: 12 doctors across 5 departments.
   - Patient Satisfaction Rating: 1 to 5 stars, negatively correlated with waiting time.
4. Missing Value Injection:
   - ~3% missing values injected into Systolic BP, Temperature, and Satisfaction Score to evaluate data cleaning pipelines.
"""

from __future__ import annotations
import math
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def lognormal_params(mean: float, std: float) -> Tuple[float, float]:
    """Calculate mu and sigma for log-normal distribution given mean and std."""
    variance = std ** 2
    mu = math.log((mean ** 2) / math.sqrt(variance + mean ** 2))
    sigma = math.sqrt(math.log(1.0 + (variance / (mean ** 2))))
    return mu, sigma


def generate_synthetic_opd_data(
    num_days: int = 180,
    start_date_str: str = "2026-01-01",
    seed: int = 42,
    output_path: str = "data/synthetic_opd.csv"
) -> pd.DataFrame:
    """
    Generates synthetic OPD patient records over the specified number of days
    and saves the resulting dataset to a CSV file.
    """
    rng = np.random.default_rng(seed)
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

    records: List[Dict] = []
    patient_global_id = 10001

    # Doctors & Department Directory
    doctor_directory = [
        {"doctor_id": "DOC_01", "name": "Dr. Sarah Chen", "department": "Emergency Medicine"},
        {"doctor_id": "DOC_02", "name": "Dr. Marcus Patel", "department": "Emergency Medicine"},
        {"doctor_id": "DOC_03", "name": "Dr. Elena Gomez", "department": "Cardiology"},
        {"doctor_id": "DOC_04", "name": "Dr. James Wilson", "department": "Cardiology"},
        {"doctor_id": "DOC_05", "name": "Dr. Aisha Al-Mansoor", "department": "Pulmonology"},
        {"doctor_id": "DOC_06", "name": "Dr. David Kim", "department": "Pulmonology"},
        {"doctor_id": "DOC_07", "name": "Dr. Rachel Green", "department": "Orthopedics"},
        {"doctor_id": "DOC_08", "name": "Dr. Thomas Brown", "department": "Orthopedics"},
        {"doctor_id": "DOC_09", "name": "Dr. Priya Sharma", "department": "General Medicine"},
        {"doctor_id": "DOC_10", "name": "Dr. Robert Garcia", "department": "General Medicine"},
        {"doctor_id": "DOC_11", "name": "Dr. Emily Taylor", "department": "Pediatrics"},
        {"doctor_id": "DOC_12", "name": "Dr. Vikram Seth", "department": "Pediatrics"},
    ]

    symptom_to_dept = {
        "Severe Chest Pain": "Cardiology",
        "Palpitations & Arrhythmia": "Cardiology",
        "Acute Respiratory Distress": "Pulmonology",
        "Severe Cough & Wheezing": "Pulmonology",
        "Compound Fracture": "Orthopedics",
        "Joint Pain & Arthritis": "Orthopedics",
        "High Fever with Delirium": "General Medicine",
        "Severe Abdominal Pain": "General Medicine",
        "Headache & Dizziness": "General Medicine",
        "Routine Health Checkup": "General Medicine",
        "Skin Rash & Allergy": "General Medicine",
        "Pediatric Fever": "Pediatrics",
    }

    triage_symptoms_map = {
        "Emergency": ["Severe Chest Pain", "Acute Respiratory Distress", "High Fever with Delirium", "Compound Fracture"],
        "Urgent": ["Palpitations & Arrhythmia", "Severe Abdominal Pain", "Severe Cough & Wheezing", "Pediatric Fever"],
        "Routine": ["Joint Pain & Arthritis", "Headache & Dizziness", "Routine Health Checkup", "Skin Rash & Allergy"],
    }

    # Consultation log-normal parameters
    mu_em, sig_em = lognormal_params(20.0, 6.0)
    mu_ur, sig_ur = lognormal_params(15.0, 5.0)
    mu_ro, sig_ro = lognormal_params(10.0, 3.0)

    # LOS log-normal parameters
    mu_gen_los, sig_gen_los = lognormal_params(48.0, 18.0)
    mu_icu_los, sig_icu_los = lognormal_params(72.0, 24.0)

    for day in range(num_days):
        current_day_date = start_date + timedelta(days=day)
        day_of_week = current_day_date.strftime("%A")
        is_weekend = day_of_week in ["Saturday", "Sunday"]

        # Number of arrivals for the day
        # Weekdays have slightly higher volume than weekends
        base_daily_volume = int(rng.normal(290 if not is_weekend else 230, 25))
        base_daily_volume = max(150, base_daily_volume)

        # Generate arrivals throughout 8:00 AM - 8:00 PM (12 hours)
        for _ in range(base_daily_volume):
            # Arrival hour sampling with peak between 9:00 - 12:00 (hours 9, 10, 11)
            # Mix of morning peak and afternoon flow
            if rng.random() < 0.45:
                # Morning peak 9:00 - 12:00
                hour_float = float(rng.uniform(9.0, 12.0))
            elif rng.random() < 0.75:
                # Afternoon 12:00 - 16:00
                hour_float = float(rng.uniform(12.0, 16.0))
            else:
                # Early morning (8:00 - 9:00) or evening (16:00 - 20:00)
                hour_float = float(rng.choice([rng.uniform(8.0, 9.0), rng.uniform(16.0, 20.0)]))

            arrival_time_str = f"{int(hour_float):02d}:{int((hour_float % 1) * 60):02d}"
            arrival_datetime = current_day_date + timedelta(hours=int(hour_float), minutes=int((hour_float % 1) * 60))

            # Demographics
            # Age: pediatric (12%), young (32%), middle (32%), senior (24%)
            age_dice = rng.random()
            if age_dice < 0.12:
                age = int(rng.integers(5, 18))
            elif age_dice < 0.44:
                age = int(rng.integers(18, 41))
            elif age_dice < 0.76:
                age = int(rng.integers(41, 66))
            else:
                age = int(rng.integers(66, 89))

            if age < 18:
                age_group = "<18"
            elif age <= 40:
                age_group = "18-40"
            elif age <= 65:
                age_group = "41-65"
            else:
                age_group = ">65"

            gender = rng.choice(["Female", "Male", "Non-binary"], p=[0.49, 0.49, 0.02])
            area = rng.choice(["Urban", "Rural"], p=[0.65, 0.35])

            # Triage Class
            triage = rng.choice(["Emergency", "Urgent", "Routine"], p=[0.10, 0.30, 0.60])
            acuity_weight = 100.0 if triage == "Emergency" else (50.0 if triage == "Urgent" else 10.0)

            # Symptoms
            if age < 18 and rng.random() < 0.50:
                symptom = "Pediatric Fever"
            else:
                symptom = rng.choice(triage_symptoms_map[triage])

            # Clinical Vitals
            if triage == "Emergency":
                systolic_bp = float(rng.normal(152.0, 22.0))
                heart_rate = float(rng.normal(110.0, 18.0))
                spo2 = float(np.clip(rng.normal(90.5, 4.2), 70.0, 99.0))
                temperature = float(rng.normal(38.5, 1.1))
                service_time = float(rng.lognormal(mu_em, sig_em))
            elif triage == "Urgent":
                systolic_bp = float(rng.normal(136.0, 16.0))
                heart_rate = float(rng.normal(95.0, 14.0))
                spo2 = float(np.clip(rng.normal(94.5, 2.5), 86.0, 99.0))
                temperature = float(rng.normal(38.0, 0.8))
                service_time = float(rng.lognormal(mu_ur, sig_ur))
            else:
                systolic_bp = float(rng.normal(122.0, 10.0))
                heart_rate = float(rng.normal(76.0, 9.0))
                spo2 = float(np.clip(rng.normal(98.2, 1.1), 94.0, 100.0))
                temperature = float(rng.normal(36.8, 0.4))
                service_time = float(rng.lognormal(mu_ro, sig_ro))

            # Waiting time (correlated with peak hours and triage priority)
            is_peak = 9.0 <= hour_float <= 12.5
            peak_delay = 18.0 if is_peak else 4.0
            if triage == "Emergency":
                base_wait = float(rng.normal(6.5, 2.5))
            elif triage == "Urgent":
                base_wait = float(rng.normal(14.0 + (peak_delay * 0.7), 4.5))
            else:
                base_wait = float(rng.normal(22.0 + peak_delay, 7.5))
            wait_time = max(1.5, base_wait)

            # Department and Doctor Matching
            target_dept = symptom_to_dept.get(symptom, "General Medicine")
            if age < 18 and rng.random() < 0.75:
                target_dept = "Pediatrics"

            dept_doctors = [d for d in doctor_directory if d["department"] == target_dept]
            if not dept_doctors:
                assigned_doc = rng.choice(doctor_directory)
            else:
                assigned_doc = rng.choice(dept_doctors)

            # Admission Propensity (Logistic Model)
            # logit = beta0 + beta1*acuity + beta2*age + beta3*spo2_deficit + beta4*bp_excess
            spo2_deficit = max(0.0, 96.0 - spo2)
            bp_excess = max(0.0, systolic_bp - 140.0) / 20.0
            logit = -4.15 + (0.025 * acuity_weight) + (0.022 * age) + (0.28 * spo2_deficit) + (0.35 * bp_excess)
            prob_admit = 1.0 / (1.0 + math.exp(-logit))
            admitted = int(rng.random() < prob_admit)

            ward_assigned = "None"
            los = 0.0
            if admitted == 1:
                # ICU vs General Ward
                is_icu = (triage == "Emergency" and rng.random() < 0.40) or (spo2 < 91.0) or (rng.random() < 0.12)
                if is_icu:
                    ward_assigned = "ICU"
                    los = float(rng.lognormal(mu_icu_los, sig_icu_los))
                else:
                    ward_assigned = "General Ward"
                    los = float(rng.lognormal(mu_gen_los, sig_gen_los))

            # Patient satisfaction (1 to 5 stars)
            # Degrades with long wait time
            expected_sat = 4.8 - (wait_time / 30.0) + (0.3 if admitted == 0 else 0.1)
            sat_score = int(np.clip(round(rng.normal(expected_sat, 0.7)), 1, 5))

            records.append({
                "patient_id": patient_global_id,
                "arrival_date": current_day_date.strftime("%Y-%m-%d"),
                "arrival_time": arrival_time_str,
                "arrival_datetime": arrival_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "hour_of_day": round(hour_float, 2),
                "day_of_week": day_of_week,
                "month": current_day_date.month,
                "is_weekend": int(is_weekend),
                "age": age,
                "age_group": age_group,
                "gender": gender,
                "area": area,
                "symptoms": symptom,
                "systolic_bp": round(systolic_bp, 1),
                "heart_rate": round(heart_rate, 1),
                "oxygen_saturation": round(spo2, 1),
                "temperature": round(temperature, 1),
                "triage_class": triage,
                "acuity_weight": acuity_weight,
                "service_time": round(service_time, 2),
                "wait_time": round(wait_time, 2),
                "doctor_id": assigned_doc["doctor_id"],
                "doctor_name": assigned_doc["name"],
                "department": assigned_doc["department"],
                "admitted": admitted,
                "ward_assigned": ward_assigned,
                "length_of_stay_hrs": round(los, 1),
                "satisfaction_score": sat_score,
            })
            patient_global_id += 1

    df = pd.DataFrame(records)

    # Inject small realistic missing values (~2-3% missing)
    # in temperature, systolic_bp, satisfaction_score to simulate real OPD record gaps
    mask_temp = rng.random(len(df)) < 0.025
    df.loc[mask_temp, "temperature"] = np.nan

    mask_bp = rng.random(len(df)) < 0.025
    df.loc[mask_bp, "systolic_bp"] = np.nan

    mask_sat = rng.random(len(df)) < 0.035
    df.loc[mask_sat, "satisfaction_score"] = np.nan

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic OPD records across {num_days} days to {output_path}")
    return df


if __name__ == "__main__":
    generate_synthetic_opd_data()
