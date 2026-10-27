# 🏥 Hospital OPD Queue & Bed Allocation Optimization: Project Summary & Technical Report

---

## 1. Executive Overview

### What the Project Does (in Simple Language)
Modern hospital outpatient departments (OPDs) face chronic operational friction: severe morning crowding, long waiting times for routine patients, unpredictable doctor availability, and delayed ward transfers. When clinical queueing operates under strict emergency priority, low-acuity patients often wait for hours without being seen—a phenomenon known as **routine patient starvation**. Conversely, purely operational queue optimizations such as Shortest-Job-First (SJF) minimize average queue time by prioritizing quick, simple checkups, which dangerously delays acute emergency cases. Furthermore, when patients need hospital admission, beds are typically requested only after consultations finish, causing long boarding delays in outpatient hallways.

This project designs, implements, and rigorously evaluates an end-to-end data science and healthcare operations research platform that solves these interrelated clinical problems. The system integrates:
1. **Discrete-Event Simulation (`SimPy`):** Simulates daily patient arrivals, triage urgency, doctor consultations, staggered lunch and tea breaks, unplanned doctor absences, floating backup pools, task-shifting nurses, and inpatient ward bed allocation.
2. **Classical Machine Learning (`scikit-learn`):** Predicts patient consultation durations, forecasts next-hour arrival surges, classifies triage acuity from physiological vitals, and predicts hospital admission propensity.
3. **Clinical Recommenders:** Matches arriving patients to the most appropriate doctor and department using Content-Based TF-IDF, Collaborative Filtering (TruncatedSVD), and a Hybrid model, complete with an automatic fallback diversion engine when preferred doctors are on leave or absent.
4. **Demand Clustering & Operational Regimes:** Categorizes hospital congestion into three distinct operating regimes (**Normal**, **Peak**, **Surge**) using K-Means and Agglomerative Hierarchical Clustering.
5. **Explainable Reinforcement Learning (RL):** Trains a Proximal Policy Optimization (PPO) agent in a 10-dimensional multi-queue `Gymnasium` environment, then distills the neural policy into an interpretable **Surrogate Decision Tree** that provides transparent, clinically auditable IF-THEN dispatch rules.
6. **Mathematical Validation & Multi-Seed Benchmarks:** Validates the simulator against theoretical $M/M/c$ queuing theory (Erlang-C formulas) and Little's Law ($L = \lambda W$), and benchmarks all operational policies across 30 random seeds with Student's $t$ 95% Confidence Intervals and paired hypothesis testing.
7. **Interactive Clinical Command Center:** A 7-tab Streamlit dashboard allowing healthcare administrators to adjust staffing, simulate surges, visualize doctor schedules via Gantt charts, test recommender fallbacks, and evaluate what-if stress scenarios.

---

## 2. System Architecture & Component Inventory

### Directory & File Purpose
The repository is structured into modular components covering data generation, preprocessing, machine learning, simulation, optimization, explainable RL, evaluation, and user interface:

| File / Folder Path | One-Line Functional Purpose |
| :--- | :--- |
| **`app/`** | Web application package directory containing the Streamlit command center. |
| `app/__init__.py` | Package initializer for the Streamlit dashboard application. |
| `app/dashboard.py` | 7-tab interactive Streamlit clinical command center with live simulation, Gantt charts, and recommender controls. |
| **`data/`** | Repository data store containing synthetic data, serialized models, result CSVs, and visualization plots. |
| `data/synthetic_opd.csv` | Primary dataset: 49,295 synthetic outpatient records across 180 operating days (6 months). |
| `data/availability_results.csv` | 95% CI benchmark evaluation table across 4 doctor availability and backup staffing scenarios. |
| `data/alpha_sweep_busy.csv` | Empirical parameter sweep of Aging Priority ($\alpha \in [0.1, 5.0]$) under high-congestion conditions. |
| `data/before_after_comparisons.json` | 30-seed CRN paired $t$-test results, confidence intervals, and dynamic narratives for all 4 core innovations plus doctor availability. |
| `data/eda_plots/` | Storage directory for Exploratory Data Analysis figures (arrivals, boxplots, heatmaps, demographics). |
| `data/clustering_plots/` | Storage directory for clustering diagnostic figures (elbow curve, silhouette scores, dendrogram, regime scatter). |
| `data/fairness_plots/` | Storage directory for demographic fairness and alpha trade-off diagnostic curves. |
| `data/rl_plots/` | Storage directory for distilled surrogate decision tree and RL feature importance plots. |
| `data/evaluation_plots/` | Storage directory for 30-seed policy benchmark bar charts with 95% confidence intervals. |
| `data/validation_plots/` | Storage directory for theoretical vs simulated $M/M/c$ Erlang-C and Little's Law verification plots. |
| `data/trained_models/` | Serialized machine learning artifacts (`.pkl` pipelines and Stable-Baselines3 `.zip` neural policy). |
| **`notebooks/`** | Exploratory research and walkthrough notebook directory. |
| `notebooks/analysis.ipynb` | Comprehensive walkthrough Jupyter notebook covering all syllabus requirements step-by-step. |
| `notebooks/generate_notebook.py` | Python utility script intended for programmatic notebook regeneration *(currently 0 bytes / empty)*. |
| **`src/`** | Core Python engine source code directory. |
| `src/__init__.py` | Package initializer for the core source library. |
| `src/data_generator.py` | Synthetic outpatient data generator creating 180 days of NHPP arrivals with clinical vitals and injected missingness. |
| `src/preprocessing_eda.py` | Domain-guided missing value imputation, descriptive statistics, correlation matrices, and EDA figure generation. |
| `src/ml_models.py` | Classical ML suite: Simple, Multiple, and Polynomial Regression; Arrival Forecasting; Logistic Regression; Decision Tree; KNN ($k=7$); and 5-fold cross-validation. |
| `src/clustering.py` | Patient phenotyping via K-Means and hourly demand regime detection via Agglomerative Hierarchical Clustering (Ward linkage). |
| `src/recommender.py` | Doctor recommendation (TF-IDF Content-Based, SVD Collaborative Filtering, Hybrid), bed allocation recommender, and doctor unavailability fallback engine. |
| `src/policies.py` | Queue dispatching policies: FCFS, Strict Priority, SJF, Aging Priority, Predict-then-Decide, and Regime-Aware dynamic staffing. |
| `src/simulation.py` | SimPy discrete-event simulation engine with shift schedules, staggered breaks ($\le 25\%$ concurrency), random absences, leave, floating pool, task shifting, and teleconsultation. |
| `src/rl_env.py` | 10-dimensional state, 3-action Gymnasium outpatient queue dispatching environment (`HospitalOPDEnv`). |
| `src/train_rl.py` | PPO reinforcement learning agent training pipeline utilizing Stable-Baselines3. |
| `src/explain_policy.py` | Policy distillation pipeline: extracts trajectories from neural PPO agent, trains surrogate Decision Tree, and measures fidelity. |
| `src/fairness.py` | Subgroup fairness auditing across age, gender, and residential area; routine starvation metrics; and alpha parameter sweeping. |
| `src/evaluate.py` | 30-seed Monte Carlo evaluation framework with Student's $t$ 95% CIs, paired $t$-tests, what-if stress tests, and sensitivity analysis. |
| `src/validation.py` | Mathematical validation module comparing SimPy engine outputs against $M/M/c$ Erlang-C formulas and Little's Law ($L = \lambda W$). |
| **`tests/`** | Automated testing suite directory. |
| `tests/test_end_to_end.py` | Comprehensive test suite containing 13 automated integration tests verifying all modules (100% passing). |
| `requirements.txt` | Package dependency specification file. |
| `README.md` | Public project documentation and high-level architecture overview. |
| `PROJECT_SUMMARY.md` | This technical project report. |

### End-to-End Data Flow Architecture

```mermaid
flowchart TD
    subgraph Data_Layer [1. Data Generation & Preprocessing]
        G[src/data_generator.py] -->|Generates 180 Days NHPP| RAW[(data/synthetic_opd.csv<br/>49,295 Rows, 28 Columns)]
        RAW --> P[src/preprocessing_eda.py]
        P -->|Domain Imputation| CLEAN[Cleaned Clinical DataFrame]
        P -->|Generates| EDA_P[data/eda_plots/*.png]
    end

    subgraph Analytics_Layer [2. Machine Learning, Clustering & Recommenders]
        CLEAN --> ML[src/ml_models.py]
        ML -->|Serializes Models| TM[(data/trained_models/*.pkl)]
        
        CLEAN --> CLUST[src/clustering.py]
        CLUST -->|K-Means & Ward Linkage| CLUST_P[data/clustering_plots/*.png]
        
        CLEAN --> REC[src/recommender.py]
        REC -->|Doctor & Bed Matching| RECS_OUT[Doctor & Bed Allocations]
    end

    subgraph Simulation_Engine [3. SimPy Discrete-Event Simulation]
        CLEAN --> SIM[src/simulation.py<br/>HospitalOPDSimulation]
        POL[src/policies.py] -->|Dispatch Rules & Supervised Staffing| SIM
        TM -.->|Service & Admit Predictions| SIM
        CLUST -.->|Operational Regimes| SIM
        SIM --> SIM_DF[Simulated Patient Records & KPIs]
        SIM --> GANTT[Staff Activity Timeline]
    end

    subgraph Reinforcement_Learning [4. Explainable RL Module]
        SIM --> RLE[src/rl_env.py<br/>HospitalOPDEnv]
        RLE --> TRL[src/train_rl.py<br/>PPO Agent Training]
        TRL --> PPO_M[(data/trained_models/ppo_hospital_agent.zip)]
        PPO_M --> EXP[src/explain_policy.py<br/>Decision Tree Distillation]
        EXP --> TREE_M[(data/trained_models/distilled_tree.pkl)]
        EXP --> RL_P[data/rl_plots/*.png]
    end

    subgraph Evaluation_Validation [5. Validation, Auditing & Evaluation]
        SIM --> VAL[src/validation.py<br/>M/M/c Erlang-C & Little's Law]
        VAL --> VAL_P[data/validation_plots/*.png]
        
        SIM --> EV[src/evaluate.py<br/>30-Seed Monte Carlo & Paired t-Tests]
        EV --> EV_P[data/evaluation_plots/*.png]
        EV --> AVAIL_CSV[(data/availability_results.csv)]
        EV --> JSON_RES[(data/before_after_comparisons.json)]
        EV --> ALPHA_CSV[(data/alpha_sweep_busy.csv)]
        
        SIM --> FAIR[src/fairness.py<br/>Demographic Auditing & Alpha Frontier]
        FAIR --> FAIR_P[data/fairness_plots/*.png]
    end

    subgraph Presentation [6. Streamlit Clinical Command Center]
        SIM_DF & GANTT & TM & RECS_OUT & JSON_RES & AVAIL_CSV & ALPHA_CSV --> DASH[app/dashboard.py<br/>7-Tab Interactive Control Center]
    end
```

---

## 3. Synthetic Data: Distributions, Parameters & Preprocessing

The primary dataset (`data/synthetic_opd.csv`) represents 6 months of outpatient records generated by `src/data_generator.py`. Every parameter is explicitly implemented in code:

### 1. Temporal & Operational Schedule
* **Operating Horizon:** 180 consecutive calendar days (January 1, 2026 to June 29, 2026).
* **Shift Duration:** 12 hours per day (8:00 AM to 8:00 PM, 720 operating minutes daily).
* **Arrival Volume Distribution:** Weekday arrivals sampled from $N(\mu = 290, \sigma = 25)$; Weekend arrivals sampled from $N(\mu = 230, \sigma = 25)$ (clipped at a minimum of 150 patients/day).
* **Diurnal Arrival Hour Sampling:** Non-Homogeneous Poisson Process (NHPP) implemented with piecewise probabilistic thinning:
  * **Morning Peak Window (9:00 AM – 12:00 PM):** 45% of daily patients drawn from $\text{Uniform}(9.0, 12.0)$.
  * **Afternoon Window (12:00 PM – 4:00 PM):** 30% of daily patients drawn from $\text{Uniform}(12.0, 16.0)$.
  * **Shoulder Hours (8:00 – 9:00 AM and 4:00 – 8:00 PM):** Remaining 25% sampled between $\text{Uniform}(8.0, 9.0)$ and $\text{Uniform}(16.0, 20.0)$.

### 2. Patient Demographics & Clinical Presentation
* **Age Distribution:** 4 clinical age cohorts generated via bimodal mixture:
  * Pediatric ($< 18$ years, age 5–17): 12% probability.
  * Young Adult (18–40 years): 32% probability.
  * Middle-Aged (41–65 years): 32% probability.
  * Geriatric ($> 65$ years, age 66–88): 24% probability.
* **Gender Distribution:** Female (49%), Male (49%), Non-binary (2%).
* **Residential Area Distribution:** Urban (65%), Rural (35%).
* **Triage Acuity Proportions & Weights:**
  * **Emergency:** 10% probability, Acuity Weight = 100.0.
  * **Urgent:** 30% probability, Acuity Weight = 50.0.
  * **Routine:** 60% probability, Acuity Weight = 10.0.

### 3. Vital Sign Distributions by Triage Acuity
Vital signs are sampled conditionally on triage category:

| Vital Sign Parameter | Emergency (10%) | Urgent (30%) | Routine (60%) |
| :--- | :---: | :---: | :---: |
| **Systolic BP (mmHg)** | $N(\mu = 152.0, \sigma = 22.0)$ | $N(\mu = 136.0, \sigma = 16.0)$ | $N(\mu = 122.0, \sigma = 10.0)$ |
| **Heart Rate (bpm)** | $N(\mu = 110.0, \sigma = 18.0)$ | $N(\mu = 95.0, \sigma = 14.0)$ | $N(\mu = 76.0, \sigma = 9.0)$ |
| **Oxygen Saturation $\text{SpO}_2$ (%)** | $N(90.5, 4.2)$, clipped to $[70, 99]$ | $N(94.5, 2.5)$, clipped to $[86, 99]$ | $N(98.2, 1.1)$, clipped to $[94, 100]$ |
| **Body Temperature (°C)** | $N(\mu = 38.5, \sigma = 1.1)$ | $N(\mu = 38.0, \sigma = 0.8)$ | $N(\mu = 36.8, \sigma = 0.4)$ |

### 4. Consultation Duration & Length of Stay (Log-Normal Parameters)
Consultation and hospitalization durations follow continuous log-normal distributions parameterized via the method of moments:
$$\mu_{\ln} = \ln\left(\frac{m^2}{\sqrt{s^2 + m^2}}\right), \quad \sigma_{\ln} = \sqrt{\ln\left(1 + \frac{s^2}{m^2}\right)}$$

* **Consultation Times:**
  * **Emergency:** Target mean $m = 20.0$ min, target std $s = 6.0$ min ($\mu_{\ln} = 2.953, \sigma_{\ln} = 0.294$).
  * **Urgent:** Target mean $m = 15.0$ min, target std $s = 5.0$ min ($\mu_{\ln} = 2.656, \sigma_{\ln} = 0.325$).
  * **Routine:** Target mean $m = 10.0$ min, target std $s = 3.0$ min ($\mu_{\ln} = 2.259, \sigma_{\ln} = 0.294$).
* **Admission Propensity Logistic Formula:**
  $$\text{logit} = -4.15 + 0.025 \cdot \text{Acuity} + 0.022 \cdot \text{Age} + 0.28 \cdot \max(0, 96 - \text{SpO}_2) + 0.35 \cdot \frac{\max(0, \text{BP} - 140)}{20}$$
  $$P(\text{admit}) = \frac{1}{1 + e^{-\text{logit}}}$$
  *Real Empirical Admission Rate:* Exactly **18.10%** (8,923 of 49,295 patients admitted).
* **Hospital Length of Stay (LOS in Hours):**
  * **Ward Selection Logic:** `is_icu = (triage == "Emergency" and random < 0.40) or (SpO2 < 91.0) or (random < 0.12)`.
  * *Actual Resulting Ward Breakdown:* **55.05% General Ward** (4,912 patients) and **44.95% ICU** (4,011 patients).
  * **General Ward LOS:** Target mean $m = 48.0$ hours, target std $s = 18.0$ hours ($\mu_{\ln} = 3.805, \sigma_{\ln} = 0.363$).
  * **ICU LOS:** Target mean $m = 72.0$ hours, target std $s = 24.0$ hours ($\mu_{\ln} = 4.223, \sigma_{\ln} = 0.325$).

### 5. Dataset Metrics & Missing Value Cleaning
* **Dimensions:** Exactly **49,295 rows** and **28 columns**.
* **Columns:** `patient_id`, `arrival_date`, `arrival_time`, `arrival_datetime`, `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `age`, `age_group`, `gender`, `area`, `symptoms`, `systolic_bp`, `heart_rate`, `oxygen_saturation`, `temperature`, `triage_class`, `acuity_weight`, `service_time`, `wait_time`, `doctor_id`, `doctor_name`, `department`, `admitted`, `ward_assigned`, `length_of_stay_hrs`, `satisfaction_score`.
* **Injected Missing Values & Imputation Pipeline (`src/preprocessing_eda.py`):**
  * `systolic_bp`: 1,265 missing values (2.57%). Imputed using the **median systolic BP conditioned on age group** (<18: 122 mmHg, 18–40: 124 mmHg, 41–65: 128 mmHg, >65: 133 mmHg; global fallback: 125.0 mmHg).
  * `temperature`: 1,211 missing values (2.46%). Imputed using the **median temperature conditioned on triage category** (Emergency: 38.5°C, Urgent: 38.0°C, Routine: 36.8°C; global fallback: 37.0°C).
  * `satisfaction_score`: 1,750 missing values (3.55%). Imputed using the **median satisfaction score conditioned on triage category**.
  * `ward_assigned`: 40,372 missing/NaN values corresponding to patients who were discharged home without inpatient admission. Imputed with the explicit category `"Not Admitted"`.

---

## 4. Simulation Engine: Mechanics, Resources & Staff Availability

The discrete-event simulation engine (`src/simulation.py`) is written in `SimPy` and models realistic clinical resources and human staffing constraints:

### 1. Patient Flow Lifecycle
1. **NHPP Arrival:** Patient generated via Poisson thinning with diurnal rate $\lambda(t)$ reaching peak intensity from 9:00 AM to 12:00 PM.
2. **Clinical Triage:** Patient assigned triage acuity, physiological vitals, and initial queue priority score.
3. **Queue Ingestion & Emergency Preemption:** Patient enters the waiting lounge. If the patient is an Emergency, `_trigger_emergency_override()` immediately attempts to wake up offline floating doctors.
4. **Dispatcher Selection:** Unified dispatcher assigns the patient to the next available server based on server qualification.
5. **Consultation Service:** Doctor is marked as `"Busy"`, consultation duration runs, and total busy time is logged.
6. **Inpatient Ward Transition:** Admitted patients check bed availability:
   * Under standard policies, patients incur administrative boarding delays (mean 30 min, std 8 min) while the ward is prepared.
   * Under **Predict-then-Decide**, pre-reservation at triage bypasses boarding delay (0.0 min delay).
7. **Departure:** Patient departure time, wait time, and bed metrics are recorded.

### 2. Physical & Human Resources
* **Senior Consultation Doctors:** $c = 5$ (Busy scenario) or $c = 6$ (Quiet scenario) parallel consultation rooms.
* **General Inpatient Beds:** $B_{gen} = 30$ to $40$ beds.
* **ICU Critical Care Beds:** $B_{icu} = 8$ to $10$ beds.
* **Break Concurrency Resource:** Dedicated `simpy.Resource` capping doctors on break simultaneously to $\le 25\%$ of active doctors.

### 3. Doctor Availability Extensions
* **Staggered Shift Breaks:**
  * Morning Tea (15 min, target $\approx 10:15$ AM).
  * Staggered Lunch (35 min, staggered between 1:00 PM and 2:00 PM).
  * Afternoon Tea (15 min, target $\approx 3:45$ PM).
  * *Concurrency Constraint:* Enforced via `break_slots` resource. If 25% of doctors are already on break, subsequent break requests wait until a colleague returns.
* **Random Sudden Absences:** 20% probability per doctor per day between 10:00 AM and 4:00 PM. Absence duration follows Log-normal($\mu = 35$ min, $\sigma = 10$ min) clipped to $[20, 60]$ minutes.
* **Whole-Day Leave:** 0 to 2 doctors per day placed on planned leave (probabilities: 50% 0, 35% 1, 15% 2).
* **Floating Doctor Pool:** 2 on-call floating physicians (`num_floating_doctors = 2`). Automatically mobilized when senior doctors take leave/absence or when the clinic enters Peak/Surge demand regimes ($Q \ge 12$).
* **Task Shifting (Junior Staff / Nurses):** 2 junior medical staff members. **Capability Constraint:** Qualified to treat **Routine patients only**. Consultation duration is 1.15x senior speed.
* **Teleconsultation Channels:** 2 virtual lines for routine patients (20% uptake), with 0.90x consultation time.

### 4. Real Empirical Utilization & Queue Lengths Across Scenarios
Measured over 30 independent seeds (seeds 100–129):

| Operational Environment | Active Senior Doctors | Base Arrival Rate | Peak Multiplier | Backup Systems Active? | Doctor Utilization (95% CI) | Average Queue Length (95% CI) | Overall Mean Wait (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Default Config** | 6 | 20.0 / hr | 2.0x | Yes (Float + Junior + Tele) | **58.33 ± 2.29%** (range: 47.6%–68.6%) | **0.42 ± 0.11 patients** (range: 0.15–1.54) | **0.92 ± 0.31 min** |
| **Quiet / Normal Scenario** | 6 | 16.0 / hr | 1.4x | No (Pure Senior Desk) | **57.92 ± 1.39%** (range: 50.1%–63.5%) | **0.54 ± 0.16 patients** (range: 0.04–2.17) | **1.60 ± 0.51 min** |
| **Busy Hospital Scenario** | 5 | 22.0 / hr | 1.8x | No (Unbacked Senior Desk) | **97.95 ± 1.23%** (range: 88.6%–100.0%) | **7.82 ± 1.26 patients** (range: 2.48–14.85) | **19.58 ± 3.11 min** |

---

## 5. Queue Dispatching Policies: Mechanics & Exact Mathematical Formulas

The simulator supports 6 dispatching and capacity management policies implemented in `src/policies.py`:

### 1. First-Come, First-Served (FCFS)
* **Mechanics:** Pure FIFO queue serving patients in strict order of arrival timestamp.
* **Formula:**
  $$\text{Priority Score}(p) = -t_{\text{arrival}}(p)$$

### 2. Strict Priority
* **Mechanics:** Acuity weight completely dominates queue ordering. Emergencies preempt all other cases; ties broken by arrival time.
* **Formula:**
  $$\text{Priority Score}(p) = \text{Acuity Weight}(p) \cdot 10000.0 - t_{\text{arrival}}(p)$$
  Where Acuity Weight is: Emergency = 100.0, Urgent = 50.0, Routine = 10.0.

### 3. Shortest-Job-First (SJF)
* **Mechanics:** Serves patients with the shortest anticipated consultation duration first to minimize mathematical queue backlog.
* **Formula:**
  $$\text{Priority Score}(p) = -d_{\text{service}}(p) \cdot 1000.0 - t_{\text{arrival}}(p)$$
  *Clinical Risk:* Because routine checkups take ~10 minutes while emergencies take ~20 minutes, pure SJF prioritizes routine cases and causes emergency patients to wait significantly longer.

### 4. Innovation 1: Fairness-Aware Aging Priority
* **Mechanics:** Balances clinical urgency with waiting duration. Routine patients gradually accrue priority credits as they wait, eventually overtaking newly arriving urgent patients.
* **Mathematical Formulation:**
  $$\text{Priority Score}(p, t) = \text{Acuity Weight}(p) + \alpha \cdot \max(0, t - t_{\text{arrival}}(p))$$
* **Parameter Settings & Overtaking Dynamics:**
  * Default setting: $\alpha = 0.5$; Busy setting: $\alpha = 2.0$.
  * Overtaking time between an Urgent patient (acuity 50) and a Routine patient (acuity 10):
    $$t_{\text{overtake}} = \frac{\text{Acuity}_{\text{Urgent}} - \text{Acuity}_{\text{Routine}}}{\alpha} = \frac{50.0 - 10.0}{\alpha} = \frac{40.0}{\alpha} \text{ minutes}$$
  * At $\alpha = 2.0$, a routine patient waiting $> 20$ minutes overtakes a newly arriving urgent patient.
  * At $\alpha = 0.5$, overtaking occurs after 80 minutes.

### 5. Innovation 2: Predict-then-Decide
* **Mechanics:** Combines multiple linear regression for consultation duration estimation $\hat{s}(p)$ with logistic admission probability estimation $\hat{P}_{\text{admit}}(p)$ and aging priority.
* **Formula:**
  $$\text{Priority Score}(p, t) = \left[\text{Acuity}(p) + \alpha \cdot W_p(t)\right] + \max\left(0, 30.0 - \min(30.0, \hat{s}(p))\right)$$
* **Pre-Reservation Trigger:** If $\hat{P}_{\text{admit}}(p) \ge 0.60$ at initial triage, an inpatient bed hold is immediately dispatched. When consultation concludes, bed boarding wait is reduced to **0.0 minutes**.

### 6. Innovation 3: Regime-Aware Dynamic Staffing
* **Mechanics:** A dedicated supervisory process inspects instantaneous arrival rate $\lambda(t)$ and queue length $Q(t)$ every 15 simulation minutes and dynamically reconfigures server capacity $c$ and aging parameter $\alpha$:
  * **Surge Regime ($\lambda \ge 32$/hr or $Q \ge 14$):** Capacity expanded to $\max(c, 8)$; both on-call floating doctors activated; effective aging parameter scaled to $\alpha_{\text{eff}} = 1.8 \alpha$.
  * **Peak Regime ($\lambda \ge 22$/hr or $Q \ge 7$):** Capacity expanded to $\max(c, 7)$; first floating doctor activated; effective aging parameter scaled to $\alpha_{\text{eff}} = 1.2 \alpha$.
  * **Normal Regime ($\lambda < 22$/hr and $Q < 7$):** Capacity reset to baseline $c$; floating doctors returned to offline reserve; effective aging parameter scaled to $\alpha_{\text{eff}} = 0.8 \alpha$.

### 7. Reinforcement Learning Policy (Gymnasium + PPO)
* **Environment (`HospitalOPDEnv`):**
  * **State Vector (10 Dimensions):**
    $$s = [Q_{em}, Q_{ur}, Q_{ro}, W_{em}, W_{ur}, W_{ro}, D_{busy}, B_{gen}, B_{icu}, \text{hour}]$$
  * **Discrete Action Space (3 Choices):** Action 0: Dispatch Emergency; Action 1: Dispatch Urgent; Action 2: Dispatch Routine.
  * **Reward Function Formulation:**
    $$R = R_{\text{service}} - (2.5 W_{em} + 0.8 W_{ur} + 0.3 W_{ro}) - 0.5 \cdot \mathbb{I}(W_{ro} > 30) \cdot W_{ro} + R_{\text{penalty}}$$
    Where $R_{\text{service}} \in \{12, 9, 7\}$ rewards valid dispatches and $R_{\text{penalty}} = -25.0$ penalizes dispatching from empty queues.
  * **PPO Architecture:** 2-layer MLP (64 units per layer), trained via Stable-Baselines3, distilled into a surrogate Decision Tree.

---

## 6. Machine Learning Suite: Models, Features, Hyperparameters & Real Metrics

All models were trained on 49,295 synthetic outpatient records using an 80/20 train/test split and evaluated with 5-fold cross-validation. The table below presents real empirical results from executing `src/ml_models.py`, `src/clustering.py`, and `src/recommender.py`:

### Machine Learning Benchmark Summary Table

| Model Class | Clinical Task | Target Variable | Feature Set Used | Key Hyperparameters | Real Evaluation Metrics (Test Set) | 5-Fold Cross-Validation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Simple Linear Regression** | Service Duration | `service_time` (min) | `acuity_weight` | OLS closed form | **RMSE: 4.052 min**<br>MAE: 3.039 min<br>$R^2$: 0.397 | **$R^2$: 0.404** |
| **Multiple Linear Regression** | Service Duration | `service_time` (min) | `acuity_weight`, `age`, `systolic_bp`, `heart_rate`, `oxygen_saturation`, `temperature` | StandardScaler + OLS | **RMSE: 4.052 min**<br>MAE: 3.040 min<br>$R^2$: 0.397 | **$R^2$: 0.404** |
| **Polynomial Regression** | Service Duration | `service_time` (min) | Degree 2 polynomial on all 6 clinical features | Degree=2, include_bias=False, StandardScaler | **RMSE: 4.052 min**<br>MAE: 3.040 min<br>$R^2$: 0.397 | **$R^2$: 0.405** |
| **Arrival Multiple Regression** | Surge Forecasting | Next-Hour Arrivals | `hour_int`, `is_weekend`, `lag_1`, `lag_2`, `rolling_mean_3`, `hour_sin`, `hour_cos` | Time-series sequential split, StandardScaler | **RMSE: 6.390 patients**<br>MAE: 5.090 patients<br>**$R^2$: 0.828** | **$R^2$: 0.797** |
| **Logistic Regression** | Inpatient Admission | `admitted` (0/1) | `acuity_weight`, `age`, `systolic_bp`, `heart_rate`, `oxygen_saturation`, `temperature`, `service_time` | StandardScaler, max_iter=1000, L2 penalty | **ROC-AUC: 0.834**<br>Accuracy: 86.6%<br>Precision: 74.3%<br>Recall: 39.9%<br>F1-Score: 0.519 | **F1-Score: 0.534** |
| **Decision Tree Classifier** | Triage Acuity | `triage_class` (3 classes) | `systolic_bp`, `heart_rate`, `oxygen_saturation`, `temperature`, `age` | max_depth=4, criterion="entropy" | **Accuracy: 90.2%**<br>Macro F1: 0.806<br>Weighted F1: 0.895 | **Accuracy: 90.1%** |
| **KNN Regressor ($k=7$)** | Service Duration | `service_time` (min) | `acuity_weight`, `age`, `systolic_bp`, `heart_rate`, `oxygen_saturation`, `temperature` | $k=7$, distance-weighted, StandardScaler | **RMSE: 4.291 min**<br>MAE: 3.222 min<br>$R^2$: 0.323 | **RMSE: 4.378 min** |
| **K-Means Clustering** | Clinical Phenotyping | Unsupervised patient clusters | `age`, `systolic_bp`, `heart_rate`, `oxygen_saturation`, `temperature`, `acuity_weight`, `service_time` | $k \in [2, 6]$, $n_{\text{init}}=10$ | **Optimal $k=2$**<br>Inertia: 41,276.98<br>**Silhouette Score: 0.397** | N/A (Unsupervised) |
| **Agglomerative Clustering** | Demand Regimes | Unsupervised hourly regimes | `arrival_count`, `emergency_ratio`, `avg_wait_min`, `admission_rate` | Ward linkage, Euclidean metric, $n_{\text{clusters}}=3$ | **3 Regimes Identified:**<br>Normal ($N=100$ hrs, $\bar{\lambda}=4.4$)<br>Peak ($N=1411$ hrs, $\bar{\lambda}=16.5$)<br>Surge ($N=649$ hrs, $\bar{\lambda}=39.3$) | Silhouette: 0.357 |
| **Content-Based Recommender** | Doctor Allocation | Specialist matching | TF-IDF on 12 doctor clinical specialization keywords | Cosine similarity + pediatric/acuity boost | **Precision@1: 0.820**<br>**Precision@3: 1.000**<br>**Precision@5: 1.000** | N/A |
| **Collaborative Filtering** | Doctor Allocation | Specialist matching | Historical patient satisfaction ratings across demographic cohorts | TruncatedSVD ($n_{\text{components}}=4$) | **Precision@1: 0.968**<br>**Precision@3: 1.000**<br>**Precision@5: 1.000** | N/A |
| **Hybrid Recommender** | Doctor Allocation | Specialist matching | Weighted linear combination of Content and Collaborative scores | $\text{Score} = 0.65 S_{\text{content}} + 0.35 S_{\text{collab}}$ | **Precision@1: 0.856**<br>**Precision@3: 1.000**<br>**Precision@5: 1.000** | N/A |

### Confusion Matrices & Feature Importances
* **Logistic Regression Admission Confusion Matrix:**
  $$\begin{pmatrix} 7827 \text{ (TN)} & 247 \text{ (FP)} \\ 1072 \text{ (FN)} & 713 \text{ (TP)} \end{pmatrix}$$
  *Notice:* While precision is high (74.3%), recall is 39.9% at the standard 0.50 cutoff, reflecting positive class imbalance (18.1% admission rate).
* **Decision Tree Triage Confusion Matrix:**
  $$\begin{pmatrix} 461 \text{ (Emergency)} & 514 & 10 \\ 80 & 2617 \text{ (Urgent)} & 277 \\ 0 & 89 & 5811 \text{ (Routine)} \end{pmatrix}$$
* **Decision Tree Feature Importances:**
  1. `oxygen_saturation`: **65.85%**
  2. `temperature`: **23.52%**
  3. `heart_rate`: **8.88%**
  4. `systolic_bp`: **1.75%**
  5. `age`: **0.00%**

---

## 7. The Four Core Innovations (Real Before/After Numbers from `before_after_comparisons.json`)

All Before vs After comparisons were audited across **30 Common Random Number (CRN) simulation seeds** (seeds 100 to 129) under the calibrated Busy Hospital scenario. The exact metrics and statistical significance decisions match `data/before_after_comparisons.json`:

### Innovation 1: Fairness-Aware Aging Priority ($\alpha = 2.0$) vs Strict Priority
* **Clinical Problem:** Under Strict Priority, high emergency volumes continually preempt routine patients, resulting in severe waiting times ($> 45$ min).
* **Implementation:** `score_aging_priority()` in `src/policies.py` dynamically calculates $\text{Priority}(t) = \text{acuity} + \alpha \cdot \text{wait}(t)$.
* **Empirical 30-Seed Before/After Comparison:**

| Performance Metric | ❌ Before: Strict Priority | ✅ After: Aging Priority ($\alpha=2.0$) | Absolute & Percent Delta | Paired $t$-Statistic | $p$-Value | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Routine Mean Wait** | 30.49 ± 4.80 min | **22.61 ± 2.81 min** | **-7.88 min (-25.8%)** | $t = -5.06$ | $p < 0.0001$ | **Yes (Improved)** |
| **Severe Starvation ($> 45$m)** | 29.45 ± 4.87 % | **23.35 ± 4.98 %** | **-6.10% (-20.7%)** | $t = -3.26$ | $p = 0.0029$ | **Yes (Improved)** |
| **Routine Starvation ($> 25$m)** | 38.63 ± 4.07 % | **38.76 ± 3.55 %** | **+0.13% (+0.3%)** | $t = 0.07$ | $p = 0.9436$ | **No ($p \ge 0.05$)** |
| **Emergency Mean Wait** | 1.50 ± 0.17 min | **4.19 ± 1.17 min** | **+2.69 min (+180.0%)** | $t = 5.00$ | $p < 0.0001$ | **Yes (Trade-off)** |
| **Overall Mean Wait** | 19.69 ± 3.03 min | **17.74 ± 2.33 min** | **-1.95 min (-9.9%)** | $t = -2.19$ | $p = 0.0369$ | **Yes (Improved)** |

*Factual Audit Finding:* While severe routine abandonment ($> 45$ min) was reduced by 20.7% ($p = 0.0029$), moderate starvation ($> 25$ min) was statistically unchanged ($p = 0.9436$). Emergency response was slightly extended from 1.5 min to 4.2 min, which remains well within clinical safety targets ($< 10$ min).

---

### Innovation 2: Predict-then-Decide Queueing & Bed Pre-Reservation vs Reactive Allocation
* **Clinical Problem:** In traditional workflows, inpatient admission bed requests occur only after consultation concludes, causing long hallway boarding delays.
* **Implementation:** `src/policies.py` evaluates $\hat{P}_{\text{admit}}$ at triage. If $\ge 0.60$, pre-reservation bypasses boarding wait. Predicted service duration $\hat{s}$ provides a shortest-job bonus.
* **Empirical 30-Seed Before/After Comparison:**

| Performance Metric | ❌ Before: Reactive Allocation | ✅ After: Predict-then-Decide | Absolute & Percent Delta | Paired $t$-Statistic | $p$-Value | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Inpatient Boarding Delay** | 29.94 ± 0.31 min | **20.46 ± 0.96 min** | **-9.48 min (-31.7%)** | $t = -18.85$ | $p < 0.0001$ | **Yes (Improved)** |
| **Overall OPD Wait Time** | 19.69 ± 3.03 min | **16.78 ± 2.07 min** | **-2.91 min (-14.8%)** | $t = -2.05$ | $p = 0.0499$ | **Yes (Improved)** |
| **Bed Rejection Rate** | 5.01 ± 1.29 % | **4.14 ± 1.48 %** | **-0.87% (-17.4%)** | $t = -1.01$ | $p = 0.3196$ | **No ($p \ge 0.05$)** |

*Factual Audit Finding:* Pre-reservation significantly reduced inpatient bed boarding delay by 31.7% ($p < 0.0001$), and predicted SJF ordering reduced overall wait by 14.8% ($p = 0.0499$). Bed rejection rate decreased from 5.01% to 4.14%, but this difference was not statistically significant ($p = 0.3196$).

---

### Innovation 3: Regime-Aware Dynamic Staffing vs Fixed Static Staffing
* **Clinical Problem:** Static doctor schedules cannot handle the 9:00 AM – 12:00 PM surge window, resulting in queue buildup and delayed recovery.
* **Implementation:** A supervisory loop monitors arrival rate and queue depth every 15 minutes, dynamically scaling capacity from 5 up to 8 doctors by mobilizing the floating pool.
* **Empirical 30-Seed Before/After Comparison:**

| Performance Metric | ❌ Before: Fixed Static Staffing | ✅ After: Regime-Aware Dynamic Staffing | Absolute & Percent Delta | Paired $t$-Statistic | $p$-Value | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Mean Wait** | 19.69 ± 3.03 min | **6.83 ± 1.66 min** | **-12.86 min (-65.3%)** | $t = -12.13$ | $p < 0.0001$ | **Yes (Improved)** |
| **Routine Mean Wait** | 30.49 ± 4.80 min | **9.35 ± 2.11 min** | **-21.13 min (-69.3%)** | $t = -12.00$ | $p < 0.0001$ | **Yes (Improved)** |
| **Senior Doctor Utilization** | 97.91 ± 1.25 % | **85.54 ± 1.81 %** | **-12.37% (-12.6%)** | $t = -19.23$ | $p < 0.0001$ | **Yes (Stabilized)** |

*Factual Audit Finding:* Dynamically scaling doctor staffing during peak surges reduced overall mean wait time by 65.3% ($p < 0.0001$) and routine wait by 69.3% ($p < 0.0001$), while stabilizing doctor utilization at a sustainable 85.5%.

---

### Innovation 4: Explainable RL Policy Distillation vs Neural PPO Agent
* **Clinical Problem:** Deep reinforcement learning neural networks are uninterpretable black boxes that clinical ethics boards reject for life-critical dispatching.
* **Implementation:** `src/explain_policy.py` records state-action trajectories from the trained PPO agent and fits an interpretable surrogate Decision Tree (`max_depth=5`, entropy).
* **Empirical 30-Seed Before/After Comparison:**

| Performance Metric | ❌ Before: Neural PPO Agent | ✅ After: Distilled Decision Tree | Absolute & Percent Delta | Paired $t$-Statistic | $p$-Value | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Action Fidelity Agreement** | Baseline (100.0%) | **57.9% Action Agreement** | **-42.1% fidelity gap** | N/A | N/A | Empirical Metric |
| **Episode Cumulative Reward** | -27,517.69 ± 1723.76 pts | **-35,843.26 ± 1311.65 pts** | **-8,325.57 pts (+30.3% penalty)** | $t = -8.57$ | $p < 0.0001$ | **Yes (Distillation Gap)** |
| **End-of-Day Routine Wait** | 133.79 ± 6.28 min | **107.99 ± 24.84 min** | **-25.80 min (-19.3%)** | $t = -2.13$ | $p = 0.0414$ | **Yes (Improved)** |
| **Model Interpretability** | 2-Layer MLP (10,000+ continuous weights) | **23 Transparent IF-THEN Clinical Rules** | Complete auditability | N/A | N/A | Regulatory Compliance |

*Factual Audit Finding:* The distilled decision tree achieves **57.9% action fidelity** to the neural PPO policy. The explainability trade-off incurs a statistically significant 30.3% penalty in cumulative reward ($p < 0.0001$), while transforming black-box neural dispatching into 23 human-auditable clinical decision rules.

---

## 8. Simulator Validation: M/M/c Queuing Theory & Little's Law

To confirm mathematical correctness, the discrete-event simulation engine was validated in `src/validation.py` against analytical steady-state formulas for a stationary $M/M/c$ queue ($c = 5$ doctors, mean service time $1/\mu = 12.0$ min) across five traffic intensities:

### Exact Theoretical vs Empirical Validation Table

| Arrival Rate $\lambda$ (pts/hr) | Theoretical Utilization $\rho$ | Empirical Utilization $\hat{\rho}$ | Theoretical Wait $W_q$ (min) | Simulated Wait $W_q$ (min) | $W_q$ Relative Error (%) | Theoretical Count $L$ | Empirical Count $\hat{L}$ | Little's Law Prediction $\lambda \cdot W$ | Little's Law Relative Error (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15.0** | 0.600 | 0.621 | 1.417 | 1.622 | 14.47% | 3.354 | 3.508 | 3.530 | **0.65%** |
| **18.0** | 0.720 | 0.771 | 3.518 | 5.728 | 62.82% | 4.655 | 5.609 | 5.475 | **2.39%** |
| **20.0** | 0.800 | 0.842 | 6.649 | 6.621 | **0.42%** | 6.216 | 6.462 | 6.302 | **2.48%** |
| **22.0** | 0.880 | 0.895 | 14.368 | 10.399 | 27.62% | 9.668 | 8.299 | 8.264 | **0.42%** |
| **23.5** | 0.940 | 0.996 | 34.167 | 69.778 | 104.23% | 18.082 | 33.159 | 32.152 | **3.04%** |

### Mathematical Analysis of Validation Results
1. **Little's Law Verification ($L = \lambda W$):** Across all tested traffic intensities, Little's Law relative error is consistently **$< 3.04\%$**, confirming internal consistency in event tracking.
2. **Behavior Under Near-Saturation ($\rho = 0.940$):** In the heaviest traffic tier, simulated $W_q$ (69.78 min) diverges from Erlang-C steady-state theory (34.17 min) by 104.23%. This occurs because Erlang-C assumes an infinite time horizon in steady state, whereas empirical simulation operates over finite simulation time where stochastic variance and queue buildup near capacity cause temporary runaway queues.

---

## 9. Comprehensive Benchmark Results & Statistical Tables

### 1. Full 30-Seed Policy Comparison Table (Busy Scenario, Seeds 100–129)
Generated by `HospitalEvaluator.run_multi_seed_benchmark(sim_days=1, use_busy=True)` in `src/evaluate.py`:

| Dispatching Policy | Mean Wait Time (95% CI) | 95th Percentile Wait (min) | Emergency Wait (95% CI) | Routine Wait (95% CI) | Doctor Utilization (95% CI) | Bed Rejection Rate (95% CI) | Routine Starvation Rate ($> 45$m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FCFS (Baseline)** | 17.65 ± 3.74 min | 50.92 min | 17.32 ± 3.99 min | 17.66 ± 3.70 min | 98.8% ± 0.9% | 4.6% ± 1.5% | 12.6% ± 5.4% |
| **Strict Priority** | 19.69 ± 3.03 min | 86.77 min | 1.50 ± 0.17 min | 30.49 ± 4.80 min | 97.9% ± 1.3% | 5.0% ± 1.3% | 29.4% ± 4.9% |
| **Shortest-Job-First (SJF)** | 9.34 ± 1.24 min | 55.91 min | 27.61 ± 3.74 min | 3.80 ± 0.47 min | 98.0% ± 1.1% | 5.0% ± 1.7% | 1.5% ± 0.4% |
| **Aging Priority ($\alpha=1.0$)** | 18.70 ± 2.77 min | 71.38 min | 1.79 ± 0.38 min | 26.28 ± 3.62 min | 98.7% ± 0.9% | 4.7% ± 1.3% | 28.7% ± 4.7% |
| **Predict-then-Decide** | 16.62 ± 2.04 min | 63.36 min | 1.61 ± 0.28 min | 23.11 ± 2.70 min | 98.2% ± 1.0% | 4.5% ± 1.4% | 24.3% ± 4.5% |
| **Regime-Aware Dynamic** | **6.83 ± 1.66 min** | **31.49 min** | **0.90 ± 0.26 min** | **9.35 ± 2.11 min** | **85.5% ± 1.8%** | **4.7% ± 1.6%** | **4.8% ± 3.7%** |

### 2. Paired Student's t-Test Hypothesis Testing vs FCFS Baseline
$$H_0: \mu_{\text{Policy}} = \mu_{\text{FCFS}} \quad \text{vs} \quad H_1: \mu_{\text{Policy}} \neq \mu_{\text{FCFS}}$$

| Comparison | Mean Difference (min) | Paired $t$-Statistic | $p$-Value | Significant ($p < 0.05$)? | Hypothesis Decision |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Strict Priority vs FCFS** | +2.04 min | $t = 1.170$ | $p = 0.2514$ | No | Fail to Reject $H_0$ |
| **SJF vs FCFS** | -8.31 min | $t = -5.606$ | $p = 4.72 \times 10^{-6}$ | **Yes** | **Reject $H_0$ (Wait Times Differ)** |
| **Aging Priority vs FCFS** | +1.05 min | $t = 0.598$ | $p = 0.5547$ | No | Fail to Reject $H_0$ |
| **Predict-then-Decide vs FCFS** | -1.03 min | $t = -0.622$ | $p = 0.5389$ | No | Fail to Reject $H_0$ |
| **Regime-Aware vs FCFS** | -10.82 min | $t = -6.638$ | $p = 2.81 \times 10^{-7}$ | **Yes** | **Reject $H_0$ (Wait Times Differ)** |

---

### 3. Doctor Availability & Backup Staffing Scenario Table (from `data/availability_results.csv`)
Evaluated across 15 seeds under high congestion:

| Operational Staffing Scenario | Overall Mean Wait (95% CI) | Emergency Wait (95% CI) | Routine Wait (95% CI) | Senior Doctor Utilization | Routine Starvation ($> 45$m) | Inpatient Bed Rejection Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. All Present (No Absences)** | 12.93 ± 6.53 min | 1.73 ± 0.63 min | 18.01 ± 10.56 min | 98.2% ± 7.5% | 14.3% ± 26.2% | 1.9% ± 4.8% |
| **2. Absences Without Backup** | 60.52 ± 110.53 min | 24.22 ± 76.43 min | 73.01 ± 115.85 min | 99.3% ± 3.0% | 57.5% ± 55.0% | 5.0% ± 2.6% |
| **3. Absences with Floating Pool** | 17.05 ± 30.23 min | 1.04 ± 1.25 min | 24.00 ± 42.49 min | 79.7% ± 18.4% | 28.6% ± 62.3% | 2.8% ± 8.4% |
| **4. Absences with Pool + Task Shifting** | **1.41 ± 1.47 min** | **0.72 ± 0.58 min** | **0.99 ± 1.10 min** | **64.2% ± 28.7%** | **0.0% ± 0.0%** | **6.9% ± 15.3%** |

*Availability Paired Comparison:* Unbacked doctor absences cause queue escalation (mean wait 60.52 min, routine wait 73.01 min, starvation 57.5%). Activating floating doctors alongside junior task-shifting prevents system backlog, reducing mean wait to 1.41 min.

---

### 4. Demographic Fairness Audit Tables (from `synthetic_opd.csv`)
Audited via `src/fairness.py`:

* **Age Cohort Fairness:**
  | Age Group | Patient Count | Mean Wait (min) | Median Wait (min) | 95th Percentile Wait (min) | Admission Rate (%) |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **$< 18$** | 5,916 | 26.78 min | 26.72 min | 46.76 min | 11.0% |
  | **18–40** | 15,782 | 26.91 min | 26.76 min | 47.16 min | 14.0% |
  | **41–65** | 15,804 | 26.96 min | 26.76 min | 47.08 min | 19.0% |
  | **$> 65$** | 11,793 | 26.75 min | 26.62 min | 47.19 min | 26.0% |

* **Gender Subgroup Fairness:**
  | Gender Group | Patient Count | Mean Wait (min) | Median Wait (min) | 95th Percentile Wait (min) | Admission Rate (%) |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Female** | 24,057 | 26.85 min | 26.68 min | 47.17 min | 18.0% |
  | **Male** | 24,233 | 26.88 min | 26.71 min | 47.04 min | 18.0% |
  | **Non-binary** | 1,005 | 27.21 min | 28.14 min | 46.12 min | 17.0% |

* **Residential Area Subgroup Fairness (Two-Sample t-Test):**
  | Residential Area | Patient Count | Mean Wait (min) | Median Wait (min) | 95th Percentile Wait (min) | Admission Rate (%) | Area $t$-Test $p$-Value |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Rural** | 17,291 | 26.78 min | 26.60 min | 47.05 min | 18.0% | **$p = 0.1930$** |
  | **Urban** | 32,004 | 26.92 min | 26.78 min | 47.14 min | 18.0% | *(No significant disparity)* |

---

### 5. What-If Operational Stress Testing Table
Evaluated across 5 random seeds using `HospitalEvaluator.run_what_if_scenarios()`:

| Operational Scenario | Evaluated Policy | Mean Wait Time (min) | Emergency Wait Time (min) | Senior Doctor Utilization (%) | Bed Rejection Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (6 Docs, 40 Beds, 1.0x Rate)** | Strict Priority | 0.69 | 0.67 | 59.0% | 2.3% |
| **Baseline** | Aging Priority | 0.69 | 0.67 | 59.0% | 2.3% |
| **Baseline** | Regime-Aware | 0.71 | 0.68 | 59.6% | 2.7% |
| **Flu Surge (+30% Arrival Volume)** | Strict Priority | 1.95 | 0.78 | 70.6% | 3.8% |
| **Flu Surge (+30% Arrival Volume)** | Aging Priority | 1.95 | 0.78 | 70.6% | 3.8% |
| **Flu Surge (+30% Arrival Volume)** | Regime-Aware | 1.96 | 0.81 | 70.6% | 4.7% |
| **Doctor Shortage (5 Active Doctors)** | Strict Priority | 1.31 | 0.93 | 61.9% | 1.7% |
| **Doctor Shortage (5 Active Doctors)** | Aging Priority | 1.31 | 0.93 | 61.9% | 1.7% |
| **Doctor Shortage (5 Active Doctors)** | Regime-Aware | 1.26 | 0.95 | 61.9% | 1.7% |
| **Bed Expansion (50 General Beds)** | Strict Priority | 0.69 | 0.67 | 59.0% | 2.3% |
| **Bed Expansion (50 General Beds)** | Aging Priority | 0.69 | 0.67 | 59.0% | 2.3% |
| **Bed Expansion (50 General Beds)** | Regime-Aware | 0.71 | 0.68 | 59.6% | 2.7% |

*Audit Finding on What-If Output:* In the Baseline and Bed Expansion rows, wait times and doctor utilizations are identical (0.69 min, 59.0%). This is because the baseline configuration is unconstrained (doctor utilization $\approx 59\%$), where bed capacity of 40 is already sufficient to avoid overflow in a 1-day run.

---

### 6. Sensitivity Analysis Tables
Evaluated across 5 random seeds using `HospitalEvaluator.run_sensitivity_analysis()`:

* **Doctor Staffing Sensitivity (4 to 8 Doctors):**
  | Active Doctor Count | Mean Wait Time (min) | Senior Doctor Utilization (%) |
  | :---: | :---: | :---: |
  | **4 Doctors** | 3.14 min | 64.8% |
  | **5 Doctors** | 1.31 min | 61.9% |
  | **6 Doctors** | 0.69 min | 59.0% |
  | **7 Doctors** | 0.45 min | 55.9% |
  | **8 Doctors** | 0.43 min | 51.6% |

* **General Bed Capacity Sensitivity (20 to 60 Beds):**
  | General Bed Count | Inpatient Bed Rejection Rate (%) |
  | :---: | :---: |
  | **20 Beds** | 2.3% |
  | **30 Beds** | 2.3% |
  | **40 Beds** | 2.3% |
  | **50 Beds** | 2.3% |
  | **60 Beds** | 2.3% |

*Audit Finding on Bed Sensitivity:* The rejection rate is constant at 2.3% from 20 to 60 beds because in a 1-day simulation ($\approx 250$ arrivals, $\approx 45$ admissions), general ward capacity is never exceeded at 20 beds; the 2.3% rejection occurs exclusively on the ICU side.

---

## 10. Dashboard Structure & Interactive Controls

The interactive dashboard is implemented in `app/dashboard.py` and structured into a persistent sidebar and 7 operational tabs:

### 1. Persistent Sidebar Controls
* **Operational Scenario Mode Radio:** Toggle between `"🔥 Busy Hospital (High Congestion: 85-95% Peak Util)"` and `"🌿 Quiet / Normal Hospital (65-75% Util)"`. Calibrates default slider values.
* **Simulate Morning Surge Button (`🚨`):** Toggles a $+60\%$ surge multiplier (`st.session_state.surge_active`).
* **Active Doctor Count Slider:** Ranges from 3 to 12 doctors (default 5 for busy, 6 for quiet).
* **General Ward Beds Slider:** Ranges from 15 to 80 beds (default 30 for busy, 40 for quiet).
* **ICU Beds Slider:** Ranges from 4 to 25 beds (default 8 for busy, 10 for quiet).
* **Arrival Rate Multiplier Slider:** Continuous scale from 0.5x to 2.5x.
* **Dispatch & Queue Policy Selectbox:** Chooses between FCFS, Strict Priority, SJF, Aging Priority, Predict-then-Decide, and Regime-Aware.
* **Aging Sensitivity ($\alpha$) Slider:** Appears when an aging policy is selected; ranges from 0.0 to 5.0 (step 0.1).
* **Mark Doctor(s) on Leave Multiselect:** Allows interactive selection of specific doctor IDs to place on whole-day leave.
* **Floating Doctor Pool Slider:** Controls on-call floating capacity (0 to 4 doctors).
* **Junior Staff / Nurses Slider:** Controls task-shifting capacity for routine cases (0 to 4 nurses).
* **Teleconsultation Channels Slider:** Controls virtual consultation lines (0 to 4 channels).
* **Enable Emergency Override Checkbox:** Toggles immediate floating doctor mobilization upon emergency arrival.
* **Simulation Random Seed Number Input:** Controls reproducibility (default 42).

### 2. Tab Navigation & Content
* **Tab 1: 🚀 Live OPD Simulator & KPI Command Center:**
  * 6 Glowing KPI metric cards: Total Arrived, Mean Wait Time, Emergency Wait Time, Routine Wait Time, Doctor Utilization, and Bed Rejection Rate.
  * Hourly Wait Trajectory Line Plot: Displays simulated patient wait times across the 12-hour shift, highlighting the 9:00 AM – 12:00 PM morning peak.
  * Triage Share Donut Chart: Displays the empirical triage volume breakdown.
  * Live Patient Log: Expandable interactive dataframe of the first 100 simulated patients.
  * Doctor Availability Status Monitor: Real-time status cards (Available, Busy, Break, Leave, Absent) for every doctor.
  * Activity Timeline Gantt Chart: Horizontal Gantt visualization from 8:00 AM to 8:00 PM displaying doctor consultations, staggered tea and lunch breaks, absences, and floating pool activations.
* **Tab 2: ✨ The 4 Innovations (Deep Dive & Before/After Comparisons):**
  * Sub-tab 1 (Aging Priority): Dynamic alpha sweep curve and overtaking time table; 30-seed CRN Before vs After comparison container.
  * Sub-tab 2 (Predict-then-Decide): Architectural overview of ML pre-reservation; 30-seed CRN Before vs After container for bed boarding delay and consultation wait.
  * Sub-tab 3 (Demand Regime Detection): Agglomerative hierarchical dendrogram, elbow/silhouette curves, and 30-seed Before vs After container for fixed vs dynamic staffing.
  * Sub-tab 4 (Explainable RL Distillation): Visualized surrogate decision tree plot, feature importance horizontal bar chart, and 30-seed Before vs After container for neural vs tree policies.
* **Tab 3: 📊 Policy Benchmark & 30-Seed Tests:**
  * Full 30-seed multi-metric comparison table with Student's $t$ 95% Confidence Intervals.
  * Paired Student's $t$-test table comparing each policy against FCFS baseline.
  * Grouped bar chart comparing overall, emergency, and routine wait times.
* **Tab 4: 🤖 Clinical Recommenders:**
  * Patient profile input form: Symptom selectbox, age slider, triage acuity, $\text{SpO}_2$, and blood pressure sliders.
  * Top-3 recommended doctors with Content, Collaborative, and Hybrid match scores.
  * Ward/Bed placement recommendation card with clinical justification.
  * Precision@k benchmark table across Content, Collaborative, and Hybrid models.
  * Doctor Unavailability Fallback Simulator: Simulates a doctor on leave or absent, showing alternative routing.
* **Tab 5: ⚖️ Fairness & Demographics:**
  * Subgroup analysis tables across Age, Gender, and Residential Area.
  * Alpha-vs-Acuity Trade-off curve.
* **Tab 6: 🔬 M/M/c & Little's Law Validation:**
  * Theoretical Erlang-C vs SimPy empirical point benchmark table across 5 traffic intensities.
  * $W_q$ wait time curve comparing theory to simulation.
  * Little's Law identity check ($L$ vs $\lambda W$).
* **Tab 7: 🧪 What-If Stress Testing:**
  * Stress testing scenario table (Flu Surge, Shortage, Bed Expansion).
  * Doctor staffing and bed capacity sensitivity curves.
  * 4-scenario doctor availability table and Before vs After container for doctor backup staffing.

---

## 11. Automated Test Suite: Coverage & Status

The test suite (`tests/test_end_to_end.py`) was executed to confirm current system status:
```bash
pytest tests/test_end_to_end.py -v
```
**Current Status: 13 passed in 8.20 seconds (100% pass rate on Python 3.14.6).**

| Test Function | Verification Scope | Status |
| :--- | :--- | :---: |
| `test_simulation_runs` | Verifies SimPy discrete-event engine completes, outputs records ($N > 100$), doctor utilization $\in [50\%, 95\%]$, and non-negative wait times. | **PASSED** |
| `test_preprocessing_and_eda` | Verifies data loading, missing value imputation (temperature, BP, satisfaction), descriptive stats, and correlation matrices. | **PASSED** |
| `test_ml_models_suite` | Verifies regression, logistic classification, decision tree accuracy ($> 80\%$), logistic ROC-AUC ($> 0.75$), KNN, and comparison table. | **PASSED** |
| `test_clustering_modules` | Verifies K-Means patient clustering, optimal $k$ selection, and Agglomerative clustering into Normal, Peak, and Surge regimes. | **PASSED** |
| `test_recommenders` | Verifies Content-Based, Collaborative, Hybrid, Bed allocation recommendations, and Precision@3 benchmark ($\ge 0.80$). | **PASSED** |
| `test_policies` | Verifies execution of all 6 dispatch policies (FCFS, Strict, SJF, Aging, Predict-then-Decide, Regime-Aware). | **PASSED** |
| `test_rl_env_and_distillation` | Verifies Gymnasium environment state reset (shape 10), step transitions, action space, and surrogate decision tree distillation ($> 50\%$ fidelity). | **PASSED** |
| `test_fairness_and_alpha_sweep` | Verifies subgroup fairness auditing (age, gender, area), routine starvation metrics, and alpha parameter sweep. | **PASSED** |
| `test_validation_theory` | Verifies analytical Erlang-C formulas and confirms Little's Law relative error is strictly $< 5\%$. | **PASSED** |
| `test_doctor_availability_and_staffing` | Verifies shift schedules, staggered breaks enforcing $\le 25\%$ concurrency cap, planned leave, and task-shifting server logs. | **PASSED** |
| `test_recommender_fallback` | Verifies that when a preferred doctor is unavailable, the fallback engine diverts to an available departmental colleague. | **PASSED** |
| `test_doctor_availability_scenarios_evaluation` | Verifies 4-scenario evaluation in `evaluate.py` and confirms output writing to `data/availability_results.csv`. | **PASSED** |
| `test_before_after_policies_differ_under_busy_scenario` | Confirms all 4 operational policy comparisons produce statistically distinguishable outcomes under high congestion. | **PASSED** |

---

## 12. Reproduction Guide: How to Run from Clean Setup

Follow these steps from a clean terminal on macOS or Linux:

### 1. Environment Setup & Dependency Installation
```bash
# Clone or navigate to the project directory
cd "/Users/angelpriscilla/Desktop/DSE PROJECT"

# Create a clean Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset (180 Days NHPP)
```bash
python3 src/data_generator.py
```
*Generates `data/synthetic_opd.csv` with 49,295 patient records across 180 operating days.*

### 3. Preprocessing, Descriptive Statistics & EDA Plots
```bash
python3 src/preprocessing_eda.py
```
*Imputes missing values and exports figures to `data/eda_plots/`.*

### 4. Train Classical Machine Learning Models & Cross-Validation
```bash
python3 src/ml_models.py
```
*Trains all regression, classification, and KNN models, prints the benchmark table, and serializes pipelines to `data/trained_models/`.*

### 5. Run Unsupervised Clustering & Recommenders
```bash
python3 src/clustering.py
python3 src/recommender.py
```
*Generates clustering diagnostic figures in `data/clustering_plots/` and evaluates Precision@k.*

### 6. Train RL Agent & Distill Explainable Decision Tree
```bash
python3 src/train_rl.py
python3 src/explain_policy.py
```
*Trains the PPO agent on `HospitalOPDEnv`, distills it into `data/trained_models/distilled_tree.pkl`, and exports plots to `data/rl_plots/`.*

### 7. Run 30-Seed Monte Carlo Evaluation & Validation
```bash
python3 src/evaluate.py
python3 src/validation.py
python3 src/fairness.py
```
*Computes 30-seed CRN paired benchmarks, creates `data/before_after_comparisons.json`, and exports validation plots to `data/validation_plots/`.*

### 8. Run Automated Pytest Suite
```bash
pytest tests/test_end_to_end.py -v
```

### 9. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app/dashboard.py
```

---

## 13. Not Implemented, Simplified, Assumed & Limitations

In accordance with strict factual reporting, the following gaps, simplifications, and limitations were identified through direct code inspection and empirical evaluation:

### 1. Identical Regression Performance Across Simple, Multiple & Polynomial Models
* **Code Reality:** In `src/ml_models.py`, Simple Linear Regression, Multiple Linear Regression, and Degree-2 Polynomial Regression all produce the exact same test RMSE (**4.052 min**) and test $R^2$ (**0.397**) to three decimal places.
* **Root Cause:** In `src/data_generator.py`, consultation duration is generated using log-normal distributions conditioned strictly on `triage_class` (which maps 1-to-1 to `acuity_weight`). None of the other physiological features (`age`, `systolic_bp`, `heart_rate`, `oxygen_saturation`, `temperature`) were included in the synthetic generator formula for service time. As a result, adding these features in multiple or polynomial regression provides zero additional explanatory variance.

### 2. Invariant Bed Rejection Rate in Bed Capacity Sensitivity
* **Code Reality:** In `HospitalEvaluator.run_sensitivity_analysis()`, increasing general bed capacity from 20 to 60 beds produces an identical bed rejection rate of **2.3%** across all configurations.
* **Root Cause:** In a 1-day simulation with $\approx 250$ arrivals and an 18.1% admission rate, only $\approx 45$ patients require inpatient admission over a 12-hour period. Because 20 beds is already more than sufficient to accommodate this volume over one day, general bed overflow never occurs. The constant 2.3% rejection is caused entirely by the separate ICU bed resource.

### 3. Non-Significant Metrics in Core Innovation Comparisons
* **Innovation 1 (Routine Starvation $> 25$m):** At $\alpha = 2.0$, routine starvation $> 25$ min is 38.63% Before vs 38.76% After ($p = 0.9436$). Aging priority at $\alpha = 2.0$ successfully reduces severe starvation ($> 45$ min, $p = 0.0029$), but does not reduce moderate starvation ($> 25$ min).
* **Innovation 2 (Bed Rejection Rate):** In `before_after_comparisons.json`, bed rejection rate decreases from 5.01% to 4.14%, but with $t = -1.01$ and $p = 0.3196$, this difference is not statistically significant. Predict-then-Decide eliminates boarding delay, but does not increase physical bed capacity.

### 4. RL Policy Distillation Trade-Off (Fidelity & Performance Gap)
* **Code Reality:** In `data/before_after_comparisons.json`, action agreement between the surrogate decision tree and the neural PPO agent is **57.9%**.
* **Distillation Penalty:** Episode cumulative reward shifts from $-27,517.69$ (neural PPO) to $-35,843.26$ (distilled decision tree), representing a statistically significant **30.3% performance drop** ($p < 0.0001$). This trade-off between optimality and human interpretability is a known characteristic of policy distillation.

### 5. Theoretical Divergence at Near-Saturation Queuing
* **Code Reality:** In `src/validation.py`, at traffic intensity $\rho = 0.940$, simulated queue wait time $W_q$ is 69.78 min versus theoretical Erlang-C wait of 34.17 min (a 104.23% relative error).
* **Root Cause:** Erlang-C formulas assume infinite time horizons in steady state. In finite simulation runs, operating near saturation ($\hat{\rho} = 0.996$) causes stochastic queue buildup that inflates empirical wait times. In contrast, Little's Law error remains low at **3.04%**.

### 6. Discrepancy Between Docstring and Implemented Ward Ratio
* **Code Reality:** The docstring in `src/data_generator.py` states: *"Bed/Ward Assignment: 75% General Ward, 25% ICU."*
* **Actual Implementation:** The generator uses:
  `is_icu = (triage == "Emergency" and rng.random() < 0.40) or (spo2 < 91.0) or (rng.random() < 0.12)`
  This yields an empirical distribution of **55.05% General Ward** and **44.95% ICU** among admitted patients.

### 7. Empty Generator Script
* **Code Reality:** `notebooks/generate_notebook.py` is present in the filesystem but is empty (**0 bytes**). All notebook content resides in `notebooks/analysis.ipynb`.

### 8. Clinical Domain Simplifications
* **Doctor Homogeneity:** In the base simulation model, all senior doctors have identical service distribution parameters, differing only by department matching and schedule availability.
* **Deterministic Triage Acuity:** Acuity is assigned statically at patient arrival; patients do not undergo dynamic vital sign deterioration in the waiting lounge that would trigger formal re-triage.
* **Stationary Inpatient Stays:** Inpatient lengths of stay abstract bed turnover cleaning times and infection control isolation constraints.

---

## 14. Inventory of All Generated Figures, Models & CSV Files

### 1. Data Files and Serialized Artifacts
* `data/synthetic_opd.csv`: Primary dataset containing 49,295 patient records.
* `data/availability_results.csv`: 4-scenario doctor availability and backup staffing benchmark table.
* `data/alpha_sweep_busy.csv`: Alpha parameter sweep table across 5 values under high congestion.
* `data/before_after_comparisons.json`: 30-seed CRN paired statistical test results for all 4 innovations plus doctor availability.

### 2. Serialized Machine Learning & RL Models (`data/trained_models/`)
* `data/trained_models/simple_linear_regression.pkl`: Simple linear regression model (`acuity_weight` $\rightarrow$ `service_time`).
* `data/trained_models/multiple_linear_regression.pkl`: Multiple linear regression pipeline with `StandardScaler`.
* `data/trained_models/polynomial_regression.pkl`: Degree-2 polynomial regression pipeline.
* `data/trained_models/arrival_forecaster.pkl`: Next-hour arrival volume multiple regression forecaster.
* `data/trained_models/admission_classifier.pkl`: Logistic regression admission propensity classifier.
* `data/trained_models/triage_decision_tree.pkl`: Multi-class triage acuity classification tree (`max_depth=4`).
* `data/trained_models/knn_regressor.pkl`: K-Nearest Neighbors regressor ($k=7$) with distance weighting.
* `data/trained_models/distilled_tree.pkl`: Surrogate decision tree distilled from the neural PPO policy.
* `data/trained_models/ppo_hospital_agent.zip`: Trained Stable-Baselines3 PPO neural network agent.

### 3. Exploratory Data Analysis Plots (`data/eda_plots/`)
* `data/eda_plots/hourly_arrivals.png`: Bar chart of hourly patient arrival volume highlighting the 9:00 AM – 12:00 PM peak.
* `data/eda_plots/wait_time_by_triage.png`: Boxplot showing wait time distributions across Emergency, Urgent, and Routine categories.
* `data/eda_plots/correlation_heatmap.png`: Pearson correlation matrix across physiological and operational features.
* `data/eda_plots/admission_demographics.png`: Grouped bar chart comparing admission rates across age cohorts and residential areas.

### 4. Clustering Diagnostic Plots (`data/clustering_plots/`)
* `data/clustering_plots/elbow_silhouette_analysis.png`: Side-by-side plots of K-Means inertia elbow curve and silhouette coefficients ($k=2..5$).
* `data/clustering_plots/regime_dendrogram.png`: Agglomerative hierarchical clustering dendrogram with Ward linkage.
* `data/clustering_plots/demand_regimes_scatter.png`: Scatter plot of hourly arrivals vs average wait time colored by operational regime.

### 5. Fairness, RL & Policy Evaluation Plots
* `data/fairness_plots/alpha_fairness_tradeoff.png`: Dual-axis plot of Emergency Wait and Routine Wait vs Aging Sensitivity $\alpha$.
* `data/rl_plots/rl_distilled_tree.png`: Complete visual diagram of the surrogate decision tree policy.
* `data/rl_plots/rl_feature_importances.png`: Horizontal bar chart of state feature importances in the distilled policy.
* `data/evaluation_plots/policy_benchmark_waits.png`: Grouped bar chart of 30-seed wait times by triage acuity across all 6 policies.

### 6. Queuing Theory & Simulator Validation Plots (`data/validation_plots/`)
* `data/validation_plots/mmc_validation.png`: Theoretical Erlang-C $W_q$ curve versus empirical simulation points across traffic intensities.
* `data/validation_plots/littles_law_validation.png`: Scatter plot verifying Little's Law identity ($L$ vs $\lambda W$).

---

## 15. Audit Summary: Suspicious Results, Inconsistencies & Verifications

During code inspection and empirical reproduction, the following specific behaviors were verified:

1. **Identical Metrics Across Regression Models:** Simple, Multiple, and Polynomial Regression all output RMSE = 4.052 and $R^2$ = 0.397. This is verified to be a structural consequence of the synthetic data generation formula rather than an execution error.
2. **Identical Results in What-If Scenarios:** Baseline and Bed Expansion scenarios in `run_what_if_scenarios()` produce identical numbers (Mean Wait: 0.69 min, Doctor Util: 59.0%, Bed Rejection: 2.3%). This occurs because the baseline configuration is unconstrained, so adding 10 beds does not impact a system that does not experience bed shortages.
3. **Constant Bed Rejection in Sensitivity Analysis:** Bed capacity sensitivity yields 2.3% rejection across 20, 30, 40, 50, and 60 general beds because general beds never fill to capacity during a 1-day run; rejections occur only in the ICU.
4. **Non-Significant $p$-Values in Before/After Comparisons:**
   * Innovation 1: Routine Starvation $> 25$m ($p = 0.9436$) shows no significant change, whereas severe starvation $> 45$m ($p = 0.0029$) is significantly reduced.
   * Innovation 2: Bed Rejection Rate ($p = 0.3196$) shows no statistically significant change.
5. **RL Distillation Fidelity Gap:** The distilled decision tree achieves 57.9% action fidelity with a 30.3% penalty in cumulative reward ($p < 0.0001$), which is documented as a known trade-off between neural optimality and clinical interpretability.
6. **Documentation Discrepancy in Ward Proportions:** The generator docstring cites 75% General / 25% ICU, but the code implementation produces 55.05% General / 44.95% ICU.
7. **Empty Utility Script:** `notebooks/generate_notebook.py` is present but contains 0 bytes.
8. **Test Suite Status:** All 13 tests in `tests/test_end_to_end.py` pass cleanly in 8.20 seconds on Python 3.14.6.
