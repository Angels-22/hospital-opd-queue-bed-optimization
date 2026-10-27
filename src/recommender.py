"""
Recommender Systems Module for Hospital Doctor/Department and Ward/Bed Allocation.

Covers classical Data Science syllabus requirements:
1. Content-Based Recommender:
   - TF-IDF vectorization and Cosine Similarity matching patient symptoms
     and demographics to doctor clinical specializations.
2. Collaborative Filtering Recommender:
   - Matrix Factorization / Item-Item similarity based on historical patient
     satisfaction ratings and clinical treatment records.
3. Hybrid Recommender:
   - Weighted blend of Content-Based affinity and Collaborative ratings.
4. Ward / Bed Allocation Recommender:
   - Clinical acuity and vital-sign triage matching for General Ward vs ICU.
5. Evaluation with Precision@k (Precision@1, Precision@3, Precision@5).
"""

from __future__ import annotations
import os
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing_eda import clean_missing_values, load_raw_data


class HospitalRecommenderSuite:
    """Implements Content-based, Collaborative, and Hybrid recommendations."""

    def __init__(self, data_path: str = "data/synthetic_opd.csv", seed: int = 42):
        self.data_path = data_path
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        raw_df = load_raw_data(data_path)
        self.df, _ = clean_missing_values(raw_df)

        self._build_doctor_catalog()
        self._build_content_model()
        self._build_collaborative_model()

    def _build_doctor_catalog(self):
        """Constructs rich doctor profiles and clinical specializations."""
        self.doctor_catalog = pd.DataFrame([
            {
                "doctor_id": "DOC_01",
                "name": "Dr. Sarah Chen",
                "department": "Emergency Medicine",
                "specialty_keywords": "Severe Chest Pain, Acute Respiratory Distress, Major Trauma, Stroke, Critical Emergency Resuscitation, Shock",
                "target_age": "All Ages",
            },
            {
                "doctor_id": "DOC_02",
                "name": "Dr. Marcus Patel",
                "department": "Emergency Medicine",
                "specialty_keywords": "High Fever with Delirium, Acute Trauma, Compound Fracture, Toxicology, Acute Sepsis Emergency",
                "target_age": "All Ages",
            },
            {
                "doctor_id": "DOC_03",
                "name": "Dr. Elena Gomez",
                "department": "Cardiology",
                "specialty_keywords": "Severe Chest Pain, Angina, Myocardial Infarction, Palpitations & Arrhythmia, Coronary Artery Disease, Hypertension",
                "target_age": "Adults & Geriatric",
            },
            {
                "doctor_id": "DOC_04",
                "name": "Dr. James Wilson",
                "department": "Cardiology",
                "specialty_keywords": "Palpitations & Arrhythmia, Heart Failure, Echocardiography, Valvular Disease, High Blood Pressure, Dizziness",
                "target_age": "Adults & Geriatric",
            },
            {
                "doctor_id": "DOC_05",
                "name": "Dr. Aisha Al-Mansoor",
                "department": "Pulmonology",
                "specialty_keywords": "Acute Respiratory Distress, Severe Cough & Wheezing, Asthma Exacerbation, COPD, Pneumonia, Bronchitis, Hypoxia",
                "target_age": "All Ages",
            },
            {
                "doctor_id": "DOC_06",
                "name": "Dr. David Kim",
                "department": "Pulmonology",
                "specialty_keywords": "Severe Cough & Wheezing, Chronic Lung Disease, Shortness of Breath, Pulmonary Embolism, Sleep Apnea",
                "target_age": "All Ages",
            },
            {
                "doctor_id": "DOC_07",
                "name": "Dr. Rachel Green",
                "department": "Orthopedics",
                "specialty_keywords": "Compound Fracture, Bone Trauma, Joint Pain & Arthritis, Sports Injury, Dislocation, Musculoskeletal Pain",
                "target_age": "All Ages",
            },
            {
                "doctor_id": "DOC_08",
                "name": "Dr. Thomas Brown",
                "department": "Orthopedics",
                "specialty_keywords": "Joint Pain & Arthritis, Osteoarthritis, Hip & Knee Replacement, Chronic Back Pain, Tendonitis",
                "target_age": "Adults & Geriatric",
            },
            {
                "doctor_id": "DOC_09",
                "name": "Dr. Priya Sharma",
                "department": "General Medicine",
                "specialty_keywords": "High Fever with Delirium, Severe Abdominal Pain, Headache & Dizziness, Routine Health Checkup, Skin Rash & Allergy, Infection",
                "target_age": "Adults",
            },
            {
                "doctor_id": "DOC_10",
                "name": "Dr. Robert Garcia",
                "department": "General Medicine",
                "specialty_keywords": "Routine Health Checkup, Headache & Dizziness, Diabetes Management, Hypertension Check, General Malaise, Skin Rash & Allergy",
                "target_age": "Adults & Geriatric",
            },
            {
                "doctor_id": "DOC_11",
                "name": "Dr. Emily Taylor",
                "department": "Pediatrics",
                "specialty_keywords": "Pediatric Fever, Child Asthma, Pediatric Infection, Growth Screening, Childhood Vaccination, Rash in Children",
                "target_age": "Pediatric (<18)",
            },
            {
                "doctor_id": "DOC_12",
                "name": "Dr. Vikram Seth",
                "department": "Pediatrics",
                "specialty_keywords": "Pediatric Fever, Infant Care, Neonatal Care, Pediatric Abdominal Pain, Child Respiratory Infection",
                "target_age": "Pediatric (<18)",
            },
        ])

    def _build_content_model(self):
        """Builds TF-IDF matrix for doctor clinical specialization keywords."""
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.doc_tfidf_matrix = self.tfidf.fit_transform(self.doctor_catalog["specialty_keywords"])

    def _build_collaborative_model(self):
        """
        Builds Collaborative Filtering model using TruncatedSVD Matrix Factorization
        over historical patient interactions and satisfaction scores.
        """
        # Create Patient Cluster x Doctor interaction matrix
        # Aggregate by age_group, triage_class, symptoms to avoid sparse singular matrix
        interaction_df = self.df.groupby(["age_group", "symptoms", "doctor_id"])["satisfaction_score"].mean().unstack(fill_value=3.5)
        self.cohort_profiles = interaction_df.index
        self.cohort_doctor_matrix = interaction_df.values  # Shape: (num_cohorts, num_doctors)
        self.collab_doctor_ids = list(interaction_df.columns)

        # Truncated SVD for latent factors
        n_components = min(4, self.cohort_doctor_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        self.reduced_cohort_matrix = self.svd.fit_transform(self.cohort_doctor_matrix)
        self.reduced_doctor_matrix = self.svd.components_.T  # (num_doctors, n_components)

    # -----------------------------------------------------------------------
    # 1. Content-Based Recommendation
    # -----------------------------------------------------------------------

    def recommend_content_based(
        self,
        symptoms: str,
        age: int,
        triage_class: str = "Routine",
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Recommends doctors using TF-IDF symptom matching with age and triage boosting.
        """
        query_text = f"{symptoms} {symptoms}"
        if age < 18:
            query_text += " pediatric child infant"
        if triage_class == "Emergency":
            query_text += " critical emergency trauma resuscitation"

        query_vec = self.tfidf.transform([query_text])
        sim_scores = cosine_similarity(query_vec, self.doc_tfidf_matrix).flatten()

        ranked_indices = np.argsort(sim_scores)[::-1]
        recs = []
        for idx in ranked_indices[:top_k]:
            doc_row = self.doctor_catalog.iloc[idx]
            score = float(sim_scores[idx])
            # Pediatric bonus/penalty
            if age < 18 and doc_row["department"] == "Pediatrics":
                score = min(1.0, score + 0.25)
            elif age >= 18 and doc_row["department"] == "Pediatrics":
                score = max(0.0, score - 0.40)

            recs.append({
                "doctor_id": doc_row["doctor_id"],
                "doctor_name": doc_row["name"],
                "department": doc_row["department"],
                "score": round(score, 3),
                "method": "Content-Based",
            })
        recs.sort(key=lambda x: x["score"], reverse=True)
        return recs[:top_k]

    # -----------------------------------------------------------------------
    # 2. Collaborative Filtering Recommendation
    # -----------------------------------------------------------------------

    def recommend_collaborative(
        self,
        age_group: str,
        symptoms: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Recommends doctors based on historical satisfaction ratings of similar patient cohorts.
        """
        cohort_key = (age_group, symptoms)
        if cohort_key in self.cohort_profiles:
            cohort_idx = list(self.cohort_profiles).index(cohort_key)
            pred_ratings = np.dot(self.reduced_cohort_matrix[cohort_idx], self.reduced_doctor_matrix.T)
        else:
            # Fallback to mean doctor satisfaction across entire clinic
            pred_ratings = np.mean(self.cohort_doctor_matrix, axis=0)

        top_indices = np.argsort(pred_ratings)[::-1]
        recs = []
        for idx in top_indices[:top_k]:
            doc_id = self.collab_doctor_ids[idx]
            doc_info = self.doctor_catalog[self.doctor_catalog["doctor_id"] == doc_id].iloc[0]
            # Normalize 1-5 rating into 0-1 score
            norm_score = float(np.clip((pred_ratings[idx] - 1.0) / 4.0, 0.0, 1.0))
            recs.append({
                "doctor_id": doc_id,
                "doctor_name": doc_info["name"],
                "department": doc_info["department"],
                "score": round(norm_score, 3),
                "predicted_satisfaction": round(float(pred_ratings[idx]), 2),
                "method": "Collaborative Filtering",
            })
        return recs

    # -----------------------------------------------------------------------
    # 3. Hybrid Recommendation
    # -----------------------------------------------------------------------

    def recommend_hybrid(
        self,
        symptoms: str,
        age: int,
        age_group: str,
        triage_class: str = "Routine",
        alpha_content: float = 0.65,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Recommender: Combines Content-Based similarity with Collaborative ratings.
        Score = alpha * Content_Score + (1 - alpha) * Collaborative_Score.
        """
        cb_recs = {r["doctor_id"]: r["score"] for r in self.recommend_content_based(symptoms, age, triage_class, top_k=12)}
        cf_recs = {r["doctor_id"]: r["score"] for r in self.recommend_collaborative(age_group, symptoms, top_k=12)}

        combined_scores = []
        for _, row in self.doctor_catalog.iterrows():
            d_id = row["doctor_id"]
            s_cb = cb_recs.get(d_id, 0.0)
            s_cf = cf_recs.get(d_id, 0.5)
            hybrid_score = (alpha_content * s_cb) + ((1.0 - alpha_content) * s_cf)

            combined_scores.append({
                "doctor_id": d_id,
                "doctor_name": row["name"],
                "department": row["department"],
                "score": round(float(hybrid_score), 3),
                "content_score": round(float(s_cb), 3),
                "collaborative_score": round(float(s_cf), 3),
                "method": "Hybrid",
            })

        combined_scores.sort(key=lambda x: x["score"], reverse=True)
        return combined_scores[:top_k]

    # -----------------------------------------------------------------------
    # Recommender Fallback Engine (Doctor & Department Availability)
    # -----------------------------------------------------------------------

    def recommend_with_fallback(
        self,
        symptoms: str,
        age: int,
        age_group: str,
        triage_class: str = "Routine",
        unavailable_doctors: Optional[List[str]] = None,
        unavailable_departments: Optional[List[str]] = None,
        alpha_content: float = 0.65,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Suggests primary recommendations and provides automatic fallback recommendations
        when the preferred doctor or department is unavailable (On Break, Leave, Absent, or at capacity).
        """
        unavail_docs = set(unavailable_doctors or [])
        unavail_depts = set(unavailable_departments or [])

        # Cross-department clinical fallback mapping
        department_fallbacks = {
            "Cardiology": ["Emergency Medicine", "General Medicine"],
            "Pulmonology": ["Emergency Medicine", "General Medicine"],
            "Orthopedics": ["Emergency Medicine", "General Medicine"],
            "Pediatrics": ["General Medicine", "Emergency Medicine"],
            "Emergency Medicine": ["General Medicine", "Cardiology"],
            "General Medicine": ["Emergency Medicine"],
        }

        all_hybrid = self.recommend_hybrid(
            symptoms=symptoms,
            age=age,
            age_group=age_group,
            triage_class=triage_class,
            alpha_content=alpha_content,
            top_k=len(self.doctor_catalog)
        )

        primary_recommendation = all_hybrid[0] if all_hybrid else None
        fallback_recommendations = []
        fallback_triggered = False
        fallback_reason = "Preferred doctor is available."

        # Check if primary choice is unavailable
        is_primary_doc_unavail = primary_recommendation and (
            primary_recommendation["doctor_id"] in unavail_docs
            or primary_recommendation["doctor_name"] in unavail_docs
        )
        is_primary_dept_unavail = primary_recommendation and (
            primary_recommendation["department"] in unavail_depts
        )

        if is_primary_doc_unavail or is_primary_dept_unavail:
            fallback_triggered = True
            primary_dept = primary_recommendation["department"]
            primary_name = primary_recommendation["doctor_name"]

            # Strategy 1: Find alternative doctor in the same department
            same_dept_available = [
                d for d in all_hybrid
                if d["department"] == primary_dept
                and d["doctor_id"] not in unavail_docs
                and d["doctor_name"] not in unavail_docs
            ]

            if same_dept_available and not is_primary_dept_unavail:
                fallback_recommendations = same_dept_available[:top_k]
                fallback_reason = f"Primary choice {primary_name} is currently unavailable (Break/Leave/Absent). Diverted to available department colleague {fallback_recommendations[0]['doctor_name']}."
            else:
                # Strategy 2: Fallback to cross-department alternative
                allowed_alt_depts = department_fallbacks.get(primary_dept, ["General Medicine"])
                cross_dept_available = [
                    d for d in all_hybrid
                    if d["department"] in allowed_alt_depts
                    and d["department"] not in unavail_depts
                    and d["doctor_id"] not in unavail_docs
                    and d["doctor_name"] not in unavail_docs
                ]

                if cross_dept_available:
                    fallback_recommendations = cross_dept_available[:top_k]
                    fallback_reason = f"All specialists in {primary_dept} are currently unavailable. Cross-department fallback routed to {fallback_recommendations[0]['department']} ({fallback_recommendations[0]['doctor_name']})."
                else:
                    # Strategy 3: Any available doctor
                    any_avail = [
                        d for d in all_hybrid
                        if d["doctor_id"] not in unavail_docs
                        and d["doctor_name"] not in unavail_docs
                    ]
                    fallback_recommendations = any_avail[:top_k]
                    fallback_reason = "Emergency clinical reroute to nearest available medical practitioner."
        else:
            fallback_recommendations = all_hybrid[:top_k]

        return {
            "primary_recommendation": primary_recommendation,
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "active_recommendations": fallback_recommendations[:top_k],
        }

    def recommend_ward_and_bed(
        self,
        triage_class: str,
        oxygen_saturation: float,
        systolic_bp: float,
        predicted_admit_prob: float,
        available_gen_beds: int = 15,
        available_icu_beds: int = 3
    ) -> Dict[str, Any]:
        """
        Recommends appropriate clinical bed / ward placement based on vitals,
        predicted admission probability, and real-time bed capacity.
        """
        # Decision rules
        is_high_risk = (
            triage_class == "Emergency"
            or oxygen_saturation < 92.0
            or systolic_bp > 175.0
            or systolic_bp < 85.0
            or predicted_admit_prob > 0.70
        )
        is_moderate_risk = (
            triage_class == "Urgent"
            or (92.0 <= oxygen_saturation <= 94.0)
            or (145.0 <= systolic_bp <= 175.0)
            or (0.35 <= predicted_admit_prob <= 0.70)
        )

        if is_high_risk:
            recommended_ward = "ICU"
            reason = "High acuity vital instability: SpO2 < 92% or hypertensive shock risk."
            bed_available = available_icu_beds > 0
        elif is_moderate_risk:
            recommended_ward = "General Ward"
            reason = "Moderate clinical risk requiring inpatient observation and IV medication."
            bed_available = available_gen_beds > 0
        else:
            recommended_ward = "Outpatient / Discharge Home"
            reason = "Stable vitals and low admission propensity; outpatient oral treatment sufficient."
            bed_available = True

        return {
            "recommended_ward": recommended_ward,
            "clinical_reason": reason,
            "bed_immediately_available": bed_available,
            "available_general_beds": available_gen_beds,
            "available_icu_beds": available_icu_beds,
            "estimated_admission_probability": round(predicted_admit_prob, 3),
        }

    # -----------------------------------------------------------------------
    # 5. Precision@k Evaluation
    # -----------------------------------------------------------------------

    def evaluate_precision_at_k(self, test_sample_size: int = 500, k_list: Tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
        """
        Evaluates Precision@k across Content-Based, Collaborative, and Hybrid recommenders.
        Tests whether the clinically assigned department/doctor is correctly retrieved in top-k.
        """
        sample_df = self.df.sample(n=min(test_sample_size, len(self.df)), random_state=self.seed)

        cb_hits = {k: 0 for k in k_list}
        cf_hits = {k: 0 for k in k_list}
        hyb_hits = {k: 0 for k in k_list}
        total = len(sample_df)

        for _, row in sample_df.iterrows():
            true_dept = row["department"]
            symptoms = row["symptoms"]
            age = row["age"]
            age_group = row["age_group"]
            triage = row["triage_class"]

            # Content-based top-5
            cb_recs = self.recommend_content_based(symptoms, age, triage, top_k=max(k_list))
            cb_depts = [r["department"] for r in cb_recs]

            # Collaborative top-5
            cf_recs = self.recommend_collaborative(age_group, symptoms, top_k=max(k_list))
            cf_depts = [r["department"] for r in cf_recs]

            # Hybrid top-5
            hyb_recs = self.recommend_hybrid(symptoms, age, age_group, triage, top_k=max(k_list))
            hyb_depts = [r["department"] for r in hyb_recs]

            for k in k_list:
                if true_dept in cb_depts[:k]:
                    cb_hits[k] += 1
                if true_dept in cf_depts[:k]:
                    cf_hits[k] += 1
                if true_dept in hyb_depts[:k]:
                    hyb_hits[k] += 1

        results = []
        for k in k_list:
            results.append({
                "Evaluation Metric": f"Precision@{k}",
                "Content-Based": round(cb_hits[k] / total, 3),
                "Collaborative Filtering": round(cf_hits[k] / total, 3),
                "Hybrid Recommender": round(hyb_hits[k] / total, 3),
            })

        return pd.DataFrame(results)


if __name__ == "__main__":
    rec_suite = HospitalRecommenderSuite()
    print("Testing Content-Based Recommender:")
    recs = rec_suite.recommend_content_based("Severe Chest Pain", age=58, triage_class="Emergency")
    for r in recs:
        print(f"  - {r['doctor_name']} ({r['department']}) [Score: {r['score']}]")

    print("\nTesting Ward / Bed Recommender:")
    bed_rec = rec_suite.recommend_ward_and_bed("Emergency", oxygen_saturation=89.5, systolic_bp=165.0, predicted_admit_prob=0.82)
    print(bed_rec)

    print("\nEvaluating Precision@k across Recommender Models:")
    eval_table = rec_suite.evaluate_precision_at_k(test_sample_size=400)
    print(eval_table.to_string(index=False))
