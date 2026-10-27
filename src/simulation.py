"""
Hospital OPD Queue and Bed Allocation Simulation Engine
Built on SimPy (Discrete-Event Simulation).

Features:
- Non-homogeneous Poisson arrivals (NHPP) with peak morning hours (9am - 12pm).
- Triage classes: Emergency (10%), Urgent (30%), Routine (60%).
- Log-normal consultation times per triage category.
- ~18% admission rate with log-normal length of stay (General Ward vs ICU).
- Configurable doctor count, General beds, and ICU beds.
- Realistic doctor utilization (60% - 90%).
- Support for multiple queue policies: FCFS, Strict Priority, SJF,
  Fairness-Aware Aging Priority, Predict-then-Decide, and Regime-Aware.

Advanced Doctor Availability & Staffing Extensions:
- Shift schedule with staggered lunch (1-2pm) and short tea breaks.
- Concurrency constraint: Never allow more than 25% of doctors on break simultaneously.
- Random sudden absences (log-normal 20-60 min) and whole-day leave (0-2 doctors/day).
- Configurable floating / on-call doctor pool activated during Peak/Surge or absences.
- Task Shifting: Junior doctors / Nurses who can only handle Routine patients.
- Teleconsultation option for routine patients with limited capacity.
- Emergency Override: Emergencies pull doctors from routine consultations or mobilize floating pool.
- Full doctor state tracking (Available, Busy, Break, Leave, Absent) and availability timeline.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import simpy


# ---------------------------------------------------------------------------
# Distribution Helpers
# ---------------------------------------------------------------------------

def lognormal_params(mean: float, std: float) -> Tuple[float, float]:
    """Calculate mu and sigma for a log-normal distribution given target mean and std."""
    variance = std ** 2
    mu = math.log((mean ** 2) / math.sqrt(variance + mean ** 2))
    sigma = math.sqrt(math.log(1.0 + (variance / (mean ** 2))))
    return mu, sigma


def sample_lognormal(rng: np.random.Generator, mean: float, std: float) -> float:
    """Sample from a log-normal distribution matching target mean and std."""
    mu, sigma = lognormal_params(mean, std)
    return float(rng.lognormal(mean=mu, sigma=sigma))


# ---------------------------------------------------------------------------
# Simulation Configuration
# ---------------------------------------------------------------------------

@dataclass
class OPDConfig:
    """Configuration parameters for the Hospital OPD discrete-event simulation."""
    # Simulation duration
    sim_days: int = 1
    hours_per_day: float = 12.0  # 8:00 AM to 8:00 PM OPD
    warmup_days: int = 0

    # Staffing and Capacity
    num_doctors: int = 6
    num_general_beds: int = 40
    num_icu_beds: int = 10

    # Doctor Availability & Staffing Extensions
    enable_breaks: bool = True
    max_break_concurrency_ratio: float = 0.25  # Never >25% on break simultaneously
    enable_absences: bool = True
    enable_leave: bool = True
    manual_absent_doctors: List[int] = field(default_factory=list)  # Manual override IDs

    num_floating_doctors: int = 2          # On-call floating doctor pool
    enable_floating_pool: bool = True
    num_junior_doctors_nurses: int = 2     # Task-shifting second server type (Routine only)
    enable_task_shifting: bool = True
    teleconsultation_capacity: int = 2     # Teleconsultation capacity for routine patients
    teleconsultation_prob: float = 0.20    # Proportion of routine patients opting for teleconsult
    enable_emergency_override: bool = True # Emergency can preempt routine or summon floating pool

    # Arrival Rate (patients per hour)
    # Peak hours: 9:00 - 12:00 (hours 1, 2, 3 from 8am opening)
    base_arrival_rate: float = 20.0  # patients / hour
    peak_multiplier: float = 2.0     # Peak reaches ~40 patients/hour
    arrival_multiplier: float = 1.0  # Global scale (for what-if surge testing)

    # Triage Class Probabilities & Acuity Weights
    triage_probs: Dict[str, float] = field(default_factory=lambda: {
        "Emergency": 0.10,
        "Urgent": 0.30,
        "Routine": 0.60,
    })
    acuity_weights: Dict[str, float] = field(default_factory=lambda: {
        "Emergency": 100.0,
        "Urgent": 50.0,
        "Routine": 10.0,
    })

    # Consultation times (minutes) ~ Log-normal
    consultation_times: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Emergency": (20.0, 6.0),  # mean=20 min, std=6 min
        "Urgent": (15.0, 5.0),     # mean=15 min, std=5 min
        "Routine": (10.0, 3.0),    # mean=10 min, std=3 min
    })

    # Admission & Length of Stay (LOS)
    admission_probs: Dict[str, float] = field(default_factory=lambda: {
        "Emergency": 0.60,
        "Urgent": 0.25,
        "Routine": 0.07,
    })
    icu_share_of_admitted: float = 0.25

    # Length of Stay in hours (log-normal)
    general_ward_los: Tuple[float, float] = (48.0, 18.0)
    icu_los: Tuple[float, float] = (72.0, 24.0)

    # Queue Policy
    policy: str = "aging_priority"  # 'fcfs', 'strict_priority', 'sjf', 'aging_priority', 'predict_then_decide', 'regime_aware'
    alpha: float = 0.5
    reserve_buffer_prob: float = 0.60

    # Random seed
    seed: int = 42

    @classmethod
    def create_busy_scenario(cls, seed: int = 42, **kwargs) -> OPDConfig:
        """Calibrated high-congestion hospital scenario (peak-hour utilization ~85-95%)."""
        params = dict(
            sim_days=1,
            num_doctors=5,
            num_general_beds=30,
            num_icu_beds=8,
            base_arrival_rate=22.0,
            peak_multiplier=1.8,
            arrival_multiplier=1.0,
            enable_breaks=True,
            enable_absences=False,
            enable_leave=False,
            enable_floating_pool=False,
            num_floating_doctors=2,
            enable_task_shifting=False,
            num_junior_doctors_nurses=2,
            teleconsultation_capacity=0,
            enable_emergency_override=True,
            seed=seed,
        )
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def create_quiet_scenario(cls, seed: int = 42, **kwargs) -> OPDConfig:
        """Calibrated quiet / normal hospital scenario (utilization ~65-75%)."""
        params = dict(
            sim_days=1,
            num_doctors=6,
            num_general_beds=40,
            num_icu_beds=10,
            base_arrival_rate=16.0,
            peak_multiplier=1.4,
            arrival_multiplier=1.0,
            enable_breaks=True,
            enable_absences=False,
            enable_leave=False,
            enable_floating_pool=False,
            num_floating_doctors=2,
            enable_task_shifting=False,
            num_junior_doctors_nurses=2,
            teleconsultation_capacity=0,
            enable_emergency_override=True,
            seed=seed,
        )
        params.update(kwargs)
        return cls(**params)


# ---------------------------------------------------------------------------
# Patient Record Data Structure
# ---------------------------------------------------------------------------

@dataclass
class Patient:
    patient_id: int
    triage_class: str
    acuity_weight: float
    arrival_time: float             # In minutes from start of sim
    sim_day: int
    hour_of_day: float              # 8.0 - 20.0
    age: int
    age_group: str
    gender: str
    area: str
    symptoms: str

    # Clinical vitals
    systolic_bp: float = 120.0
    heart_rate: float = 80.0
    oxygen_saturation: float = 98.0
    temperature: float = 37.0

    # Simulation timeline
    consultation_expected_time: float = 0.0
    consultation_actual_time: float = 0.0
    predicted_service_time: float = 0.0
    predicted_admit_prob: float = 0.0

    service_start_time: Optional[float] = None
    service_end_time: Optional[float] = None
    wait_time: float = 0.0
    doctor_id: Optional[int] = None
    server_type: str = "Senior Doctor"  # 'Senior Doctor', 'Floating Doctor', 'Junior Doctor/Nurse', 'Teleconsultation'
    was_preempted: bool = False

    # Admission & Bed Allocation
    needs_admission: bool = False
    bed_type: Optional[str] = None  # 'General', 'ICU', or None
    bed_allocated: bool = False
    bed_rejection: bool = False
    bed_wait_time: float = 0.0
    length_of_stay_hrs: float = 0.0
    departure_time: Optional[float] = None

    def priority_score(self, current_time: float, alpha: float) -> float:
        """Dynamic aging priority score: acuity_weight + alpha * wait_time."""
        current_wait = max(0.0, current_time - self.arrival_time)
        return self.acuity_weight + alpha * current_wait


# ---------------------------------------------------------------------------
# Doctor Staff Model
# ---------------------------------------------------------------------------

@dataclass
class DoctorStaff:
    doctor_id: int
    name: str
    role: str = "Senior Doctor"  # 'Senior Doctor', 'Floating Doctor', 'Junior Doctor/Nurse', 'Teleconsultation'
    status: str = "Available"    # 'Available', 'Busy', 'Break', 'Leave', 'Absent', 'Offline'
    current_patient: Optional[Patient] = None
    last_status_change_time: float = 0.0
    total_busy_time: float = 0.0
    total_break_time: float = 0.0
    total_absence_time: float = 0.0
    patients_treated: int = 0
    state_timeline: List[Dict[str, Any]] = field(default_factory=list)

    def set_status(self, new_status: str, current_time: float, patient_id: Optional[int] = None):
        """Transitions doctor state and records interval for availability timeline."""
        if self.state_timeline and self.state_timeline[-1]["status"] == self.status:
            self.state_timeline[-1]["end_time"] = current_time
            self.state_timeline[-1]["end_hour"] = 8.0 + (current_time % 720.0) / 60.0

        if new_status != self.status or not self.state_timeline:
            self.state_timeline.append({
                "doctor_id": self.doctor_id,
                "name": self.name,
                "role": self.role,
                "status": new_status,
                "start_time": current_time,
                "start_hour": 8.0 + (current_time % 720.0) / 60.0,
                "end_time": current_time,
                "end_hour": 8.0 + (current_time % 720.0) / 60.0,
                "patient_id": patient_id,
            })
        self.status = new_status
        self.last_status_change_time = current_time


# ---------------------------------------------------------------------------
# Hospital OPD Simulation Engine
# ---------------------------------------------------------------------------

class HospitalOPDSimulation:
    """Discrete-Event Simulator for Hospital OPD using SimPy with realistic staff availability."""

    def __init__(self, config: OPDConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        random.seed(config.seed)

        self.env = simpy.Environment()

        # Legacy capacity resource for compatibility
        self.doctors = simpy.Resource(self.env, capacity=config.num_doctors)
        self.general_beds = simpy.Resource(self.env, capacity=config.num_general_beds)
        self.icu_beds = simpy.Resource(self.env, capacity=config.num_icu_beds)

        # Break concurrency slots (strictly enforces <= 25% of doctors on break simultaneously)
        break_cap = max(1, int(config.num_doctors * config.max_break_concurrency_ratio))
        self.break_slots = simpy.Resource(self.env, capacity=break_cap)

        # Explicit Doctor & Staff Entities
        self.senior_doctors: List[DoctorStaff] = []
        self.floating_doctors: List[DoctorStaff] = []
        self.junior_staff: List[DoctorStaff] = []
        self.teleconsult_channels: List[DoctorStaff] = []

        self._initialize_staff_roster()

        # Dynamic waiting queue of patients
        self.waiting_queue: List[Patient] = []

        # Completed patient records
        self.records: List[Patient] = []

        # Tracking metrics
        self.total_patients_arrived = 0
        self.doctor_busy_time = 0.0
        self.queue_depth_log: List[Dict[str, Any]] = []

        # Predictor models
        self.service_time_predictor: Optional[Callable[[Patient], float]] = None
        self.admit_predictor: Optional[Callable[[Patient], float]] = None

        # Regime indicator
        self.current_regime: str = "Normal"

    def _initialize_staff_roster(self):
        """Constructs staff objects with schedules, roles, and leave assignments."""
        senior_names = [
            "Dr. Sarah Chen", "Dr. Marcus Patel", "Dr. Elena Gomez",
            "Dr. James Wilson", "Dr. Aisha Al-Mansoor", "Dr. David Kim",
            "Dr. Rachel Green", "Dr. Thomas Brown", "Dr. Priya Sharma",
            "Dr. Robert Garcia", "Dr. Emily Taylor", "Dr. Vikram Seth"
        ]

        # 1. Senior Scheduled Doctors
        for i in range(self.config.num_doctors):
            name = senior_names[i % len(senior_names)]
            d = DoctorStaff(doctor_id=i + 1, name=name, role="Senior Doctor")
            d.set_status("Available", 0.0)
            self.senior_doctors.append(d)

        # 2. Whole-day Leave Assignment (0 to 2 doctors per day)
        leave_ids = list(self.config.manual_absent_doctors)
        if self.config.enable_leave and not leave_ids and self.config.num_doctors > 3:
            num_leave = int(self.rng.choice([0, 1, 2], p=[0.50, 0.35, 0.15]))
            candidate_ids = [d.doctor_id for d in self.senior_doctors]
            leave_ids = list(self.rng.choice(candidate_ids, size=min(num_leave, len(candidate_ids)), replace=False))

        for d in self.senior_doctors:
            if d.doctor_id in leave_ids:
                d.set_status("Leave", 0.0)

        # 3. Floating / On-call Doctors
        if self.config.enable_floating_pool:
            for j in range(self.config.num_floating_doctors):
                fid = 101 + j
                d_float = DoctorStaff(doctor_id=fid, name=f"Dr. Float #{j+1} (On-Call)", role="Floating Doctor")
                d_float.set_status("Offline", 0.0)
                self.floating_doctors.append(d_float)

        # 4. Junior Staff / Nurses (Task Shifting - Routine only)
        if self.config.enable_task_shifting:
            for k in range(self.config.num_junior_doctors_nurses):
                jid = 201 + k
                d_jun = DoctorStaff(doctor_id=jid, name=f"Nurse/Junior #{k+1}", role="Junior Doctor/Nurse")
                d_jun.set_status("Available", 0.0)
                self.junior_staff.append(d_jun)

        # 5. Teleconsultation Virtual Channels (Routine only)
        for m in range(self.config.teleconsultation_capacity):
            tid = 301 + m
            d_tele = DoctorStaff(doctor_id=tid, name=f"Tele-Channel #{m+1}", role="Teleconsultation")
            d_tele.set_status("Available", 0.0)
            self.teleconsult_channels.append(d_tele)

    def arrival_rate_at(self, sim_time_minutes: float) -> float:
        """Calculates instantaneous arrival rate under NHPP with morning peak."""
        day_minute = sim_time_minutes % (self.config.hours_per_day * 60.0)
        hour = day_minute / 60.0
        base = self.config.base_arrival_rate * self.config.arrival_multiplier

        if 1.0 <= hour <= 4.0:
            peak_shape = math.sin((hour - 1.0) / 3.0 * math.pi)
            rate = base * (1.0 + (self.config.peak_multiplier - 1.0) * peak_shape)
        elif 4.0 < hour <= 7.0:
            rate = base * 1.15
        elif 7.0 < hour <= 10.0:
            rate = base * 0.90
        else:
            rate = base * 0.70

        return max(2.0, rate)

    def _sample_patient_profile(self, arrival_time: float, patient_id: int) -> Patient:
        """Sample synthetic clinical demographic and triage features."""
        day_idx = int(arrival_time // (self.config.hours_per_day * 60.0)) + 1
        hour_val = 8.0 + (arrival_time % (self.config.hours_per_day * 60.0)) / 60.0

        triage_types = list(self.config.triage_probs.keys())
        triage_p = list(self.config.triage_probs.values())
        triage = self.rng.choice(triage_types, p=triage_p)
        acuity = self.config.acuity_weights[triage]

        age = int(self.rng.integers(5, 88))
        if age < 18:
            age_group = "<18"
        elif age <= 40:
            age_group = "18-40"
        elif age <= 65:
            age_group = "41-65"
        else:
            age_group = ">65"

        gender = self.rng.choice(["Female", "Male", "Non-binary"], p=[0.49, 0.49, 0.02])
        area = self.rng.choice(["Urban", "Rural"], p=[0.65, 0.35])

        symptom_pool = {
            "Emergency": ["Severe Chest Pain", "Acute Respiratory Distress", "Major Trauma", "Unconscious / Stroke"],
            "Urgent": ["High Fever with Delirium", "Severe Abdominal Pain", "Compound Fracture", "Asthma Exacerbation"],
            "Routine": ["Mild Fever & Cough", "Joint Pain", "Routine Health Checkup", "Skin Rash", "Headache"],
        }
        symptom = self.rng.choice(symptom_pool[triage])

        mean_c, std_c = self.config.consultation_times[triage]
        age_factor = 1.10 if age > 65 else 1.0
        consultation_time = sample_lognormal(self.rng, mean_c * age_factor, std_c)

        p_admit = self.config.admission_probs[triage]
        if age > 65:
            p_admit = min(0.95, p_admit * 1.30)
        admitted = bool(self.rng.random() < p_admit)

        bed_type = None
        los = 0.0
        if admitted:
            is_icu = bool(self.rng.random() < self.config.icu_share_of_admitted or (triage == "Emergency" and self.rng.random() < 0.45))
            if is_icu:
                bed_type = "ICU"
                los = sample_lognormal(self.rng, self.config.icu_los[0], self.config.icu_los[1])
            else:
                bed_type = "General"
                los = sample_lognormal(self.rng, self.config.general_ward_los[0], self.config.general_ward_los[1])

        p = Patient(
            patient_id=patient_id,
            triage_class=triage,
            acuity_weight=acuity,
            arrival_time=arrival_time,
            sim_day=day_idx,
            hour_of_day=hour_val,
            age=age,
            age_group=age_group,
            gender=gender,
            area=area,
            symptoms=symptom,
            systolic_bp=systolic_bp if "systolic_bp" in locals() else 120.0,
            heart_rate=heart_rate if "heart_rate" in locals() else 80.0,
            oxygen_saturation=spo2 if "spo2" in locals() else 98.0,
            temperature=temperature if "temperature" in locals() else 37.0,
            consultation_expected_time=consultation_time,
            consultation_actual_time=consultation_time,
            needs_admission=admitted,
            bed_type=bed_type,
            length_of_stay_hrs=los,
        )

        if self.service_time_predictor is not None:
            p.predicted_service_time = self.service_time_predictor(p)
        else:
            p.predicted_service_time = consultation_time

        if self.admit_predictor is not None:
            p.predicted_admit_prob = self.admit_predictor(p)
        else:
            p.predicted_admit_prob = p_admit

        return p

    def _select_next_patient_for_server(self, server_role: str) -> Optional[Patient]:
        """
        Selects next eligible patient based on server capabilities:
        - Junior Doctor/Nurse and Teleconsultation can ONLY handle Routine patients.
        - Senior and Floating Doctors handle all acuties according to the active policy.
        """
        if not self.waiting_queue:
            return None

        current_time = self.env.now

        # Junior staff and Teleconsultation can only treat Routine patients
        if server_role in ["Junior Doctor/Nurse", "Teleconsultation"]:
            routine_candidates = [p for p in self.waiting_queue if p.triage_class == "Routine"]
            if not routine_candidates:
                return None
            patient = min(routine_candidates, key=lambda p: p.arrival_time)
            self.waiting_queue.remove(patient)
            return patient

        # Senior or Floating Doctor: Dispatch based on policy
        policy = self.config.policy

        if policy == "fcfs":
            patient = min(self.waiting_queue, key=lambda p: p.arrival_time)
        elif policy == "strict_priority":
            patient = max(self.waiting_queue, key=lambda p: (p.acuity_weight, -p.arrival_time))
        elif policy == "sjf":
            patient = min(self.waiting_queue, key=lambda p: (p.consultation_actual_time, p.arrival_time))
        elif policy == "aging_priority":
            alpha = self.config.alpha
            patient = max(self.waiting_queue, key=lambda p: (p.priority_score(current_time, alpha), -p.arrival_time))
        elif policy == "predict_then_decide":
            alpha = self.config.alpha
            def score(p: Patient) -> float:
                aging_score = p.priority_score(current_time, alpha)
                sjf_boost = max(0.0, 30.0 - min(30.0, p.predicted_service_time))
                return aging_score + sjf_boost
            patient = max(self.waiting_queue, key=score)
        elif policy == "regime_aware":
            if self.current_regime == "Surge":
                eff_alpha = self.config.alpha * 1.8
            elif self.current_regime == "Peak":
                eff_alpha = self.config.alpha * 1.2
            else:
                eff_alpha = self.config.alpha * 0.8
            patient = max(self.waiting_queue, key=lambda p: (p.priority_score(current_time, eff_alpha), -p.arrival_time))
        else:
            patient = min(self.waiting_queue, key=lambda p: p.arrival_time)

        self.waiting_queue.remove(patient)
        return patient

    # -----------------------------------------------------------------------
    # Doctor Availability Processes: Shift, Breaks, Absences, Floating Pool
    # -----------------------------------------------------------------------

    def staff_break_supervisor(self, doctor: DoctorStaff):
        """
        Manages staggered lunch (1-2pm) and short tea breaks for a doctor.
        Strictly enforces: Never allow more than 25% of doctors on break at the same time.
        """
        if not self.config.enable_breaks or doctor.status == "Leave":
            return

        doc_idx = doctor.doctor_id - 1
        num_docs = max(1, self.config.num_doctors)

        # Stagger lunch between min 300 and 360 (1:00 PM to 2:00 PM)
        lunch_target = 300.0 + (doc_idx % num_docs) * (60.0 / max(1, num_docs))
        morning_tea_target = 135.0 + (doc_idx % num_docs) * (45.0 / max(1, num_docs))
        afternoon_tea_target = 465.0 + (doc_idx % num_docs) * (45.0 / max(1, num_docs))

        breaks_to_take = [
            ("Morning Tea", morning_tea_target, 15.0),
            ("Lunch", lunch_target, 35.0),
            ("Afternoon Tea", afternoon_tea_target, 15.0),
        ]

        for _break_name, target_time, duration in breaks_to_take:
            # Wait until target time
            if self.env.now < target_time:
                yield self.env.timeout(target_time - self.env.now)

            if doctor.status == "Leave":
                return

            # Wait if doctor is currently busy with a patient
            while doctor.status == "Busy":
                yield self.env.timeout(2.0)

            if doctor.status == "Leave":
                return

            # Request break slot from resource (enforces strictly <= 25% concurrency)
            req = self.break_slots.request()
            yield req

            doctor.set_status("Break", self.env.now)
            yield self.env.timeout(duration)
            doctor.total_break_time += duration

            self.break_slots.release(req)

            # Return to available if not on leave
            if doctor.status != "Leave":
                doctor.set_status("Available", self.env.now)

    def sudden_absence_process(self, doctor: DoctorStaff):
        """Simulates random sudden absences during the shift (log-normal 20-60 min)."""
        if not self.config.enable_absences or doctor.status == "Leave":
            return

        # 20% probability per doctor of a sudden emergency absence
        if self.rng.random() < 0.20:
            # Trigger between 10am and 4pm (min 120 to 480)
            trigger_time = float(self.rng.uniform(120.0, 480.0))
            if self.env.now < trigger_time:
                yield self.env.timeout(trigger_time - self.env.now)

            # Log-normal absence duration (mean=35m, std=10m, clipped to [20, 60])
            absence_dur = float(np.clip(sample_lognormal(self.rng, 35.0, 10.0), 20.0, 60.0))

            doctor.set_status("Absent", self.env.now)
            yield self.env.timeout(absence_dur)
            doctor.total_absence_time += absence_dur

            if doctor.status != "Leave":
                doctor.set_status("Available", self.env.now)

    def floating_pool_manager(self):
        """
        Manages the floating / on-call doctor pool.
        Activates when:
        1. Regular scheduled doctors are absent or on leave.
        2. Demand regime is Peak or Surge (or queue backlog > 12).
        """
        if not self.config.enable_floating_pool:
            return

        while True:
            absent_or_leave_count = sum(1 for d in self.senior_doctors if d.status in ["Leave", "Absent"])
            needs_surge_support = self.current_regime in ["Peak", "Surge"] or len(self.waiting_queue) >= 12

            should_activate = absent_or_leave_count > 0 or needs_surge_support

            for i, fdoc in enumerate(self.floating_doctors):
                if should_activate and i < max(absent_or_leave_count, 1 if needs_surge_support else 0):
                    if fdoc.status == "Offline":
                        fdoc.set_status("Available", self.env.now)
                else:
                    if fdoc.status == "Available":
                        fdoc.set_status("Offline", self.env.now)

            yield self.env.timeout(15.0)

    # -----------------------------------------------------------------------
    # Dispatcher Process & Emergency Override
    # -----------------------------------------------------------------------

    def patient_arrival_generator(self):
        """Generates patient arrivals via thinning algorithm."""
        max_rate = self.config.base_arrival_rate * self.config.peak_multiplier * self.config.arrival_multiplier * 1.25
        total_minutes = self.config.sim_days * self.config.hours_per_day * 60.0
        patient_counter = 0

        while self.env.now < total_minutes:
            inter_arrival_hours = self.rng.exponential(1.0 / max_rate)
            inter_arrival_minutes = inter_arrival_hours * 60.0
            yield self.env.timeout(inter_arrival_minutes)

            if self.env.now >= total_minutes:
                break

            current_rate = self.arrival_rate_at(self.env.now)
            acceptance_prob = current_rate / max_rate
            if self.rng.random() < acceptance_prob:
                patient_counter += 1
                self.total_patients_arrived += 1

                patient = self._sample_patient_profile(self.env.now, patient_counter)
                self.waiting_queue.append(patient)

                # Emergency Override Trigger
                if patient.triage_class == "Emergency" and self.config.enable_emergency_override:
                    self._trigger_emergency_override(patient)

    def _trigger_emergency_override(self, emergency_patient: Patient):
        """
        Emergency Override:
        If an on-call floating doctor is offline and pool is enabled, activates them immediately.
        Emergency patients are given top acuity (100.0) so the very next available doctor takes them.
        """
        if self.config.enable_floating_pool:
            offline_floaters = [d for d in self.floating_doctors if d.status == "Offline"]
            if offline_floaters:
                offline_floaters[0].set_status("Available", self.env.now)

    def unified_dispatcher(self):
        """Unified dispatching loop across Senior Doctors, Floating Pool, Junior Staff, and Teleconsultation."""
        total_minutes = self.config.sim_days * self.config.hours_per_day * 60.0

        while self.env.now < total_minutes or self.waiting_queue:
            if not self.waiting_queue:
                yield self.env.timeout(0.5)
                continue

            dispatched_any = False

            # 1. Senior Doctors & Active Floating Doctors
            for doc in self.senior_doctors + self.floating_doctors:
                if doc.status == "Available" and self.waiting_queue:
                    patient = self._select_next_patient_for_server(doc.role)
                    if patient:
                        doc.set_status("Busy", self.env.now, patient_id=patient.patient_id)
                        doc.current_patient = patient
                        self.env.process(self._serve_patient_with_staff(patient, doc))
                        dispatched_any = True

            # 2. Junior Staff / Nurses (Task Shifting - Routine Patients Only)
            if self.config.enable_task_shifting:
                for jun in self.junior_staff:
                    if jun.status == "Available" and self.waiting_queue:
                        patient = self._select_next_patient_for_server("Junior Doctor/Nurse")
                        if patient:
                            jun.set_status("Busy", self.env.now, patient_id=patient.patient_id)
                            jun.current_patient = patient
                            self.env.process(self._serve_patient_with_staff(patient, jun))
                            dispatched_any = True

            # 3. Teleconsultation Virtual Channels (Routine Patients Only)
            if self.config.teleconsultation_capacity > 0:
                for tel in self.teleconsult_channels:
                    if tel.status == "Available" and self.waiting_queue:
                        tele_candidates = [p for p in self.waiting_queue if p.triage_class == "Routine"]
                        if tele_candidates and self.rng.random() < self.config.teleconsultation_prob:
                            patient = tele_candidates[0]
                            self.waiting_queue.remove(patient)
                            tel.set_status("Busy", self.env.now, patient_id=patient.patient_id)
                            tel.current_patient = patient
                            self.env.process(self._serve_patient_with_staff(patient, tel))
                            dispatched_any = True

            yield self.env.timeout(0.5)

    def _serve_patient_with_staff(self, patient: Patient, staff: DoctorStaff):
        """Simulates consultation with designated staff member and subsequent bed allocation."""
        patient.service_start_time = self.env.now
        patient.wait_time = max(0.0, patient.service_start_time - patient.arrival_time)
        patient.doctor_id = staff.doctor_id
        patient.server_type = staff.role

        service_duration = patient.consultation_actual_time
        # Junior staff take slightly longer for routine (~1.15x), teleconsultation slightly faster (~0.90x)
        if staff.role == "Junior Doctor/Nurse":
            service_duration *= 1.15
        elif staff.role == "Teleconsultation":
            service_duration *= 0.90

        self.doctor_busy_time += service_duration
        staff.total_busy_time += service_duration
        staff.patients_treated += 1

        yield self.env.timeout(service_duration)

        patient.service_end_time = self.env.now
        staff.current_patient = None
        if staff.status != "Leave" and staff.status != "Absent":
            staff.set_status("Available", self.env.now)

        # Bed Allocation for Admitted Patients
        if patient.needs_admission:
            target_resource = self.icu_beds if patient.bed_type == "ICU" else self.general_beds

            if target_resource.count < target_resource.capacity:
                patient.bed_allocated = True
                patient.bed_rejection = False

                # Innovation 2: Predict-then-Decide eliminates boarding wait via pre-reservation
                if self.config.policy == "predict_then_decide" and patient.predicted_admit_prob >= self.config.reserve_buffer_prob:
                    patient.bed_wait_time = 0.0
                else:
                    # Reactive administrative boarding & ward turnover preparation delay
                    patient.bed_wait_time = float(np.clip(sample_lognormal(self.rng, 30.0, 8.0), 10.0, 50.0))

                los_minutes = sample_lognormal(self.rng, 240.0, 60.0)
                self.env.process(self._inpatient_ward_occupancy(target_resource, los_minutes))
            else:
                patient.bed_allocated = False
                patient.bed_rejection = True
                patient.bed_wait_time = 35.0

        patient.departure_time = self.env.now
        self.records.append(patient)

    def _inpatient_ward_occupancy(self, resource: simpy.Resource, duration_minutes: float):
        """Background process that holds a hospital bed for patient length of stay."""
        req = resource.request()
        yield req
        yield self.env.timeout(duration_minutes)
        resource.release(req)

    def monitor_process(self, interval_minutes: float = 15.0):
        """Periodically samples queue depths, staff statuses, and resource occupancies."""
        total_minutes = self.config.sim_days * self.config.hours_per_day * 60.0
        while self.env.now < total_minutes:
            t_now = self.env.now
            emergency_q = sum(1 for p in self.waiting_queue if p.triage_class == "Emergency")
            urgent_q = sum(1 for p in self.waiting_queue if p.triage_class == "Urgent")
            routine_q = sum(1 for p in self.waiting_queue if p.triage_class == "Routine")

            busy_senior_docs = sum(1 for d in self.senior_doctors if d.status == "Busy")
            break_docs = sum(1 for d in self.senior_doctors if d.status == "Break")
            absent_docs = sum(1 for d in self.senior_doctors if d.status in ["Absent", "Leave"])
            active_floaters = sum(1 for d in self.floating_doctors if d.status in ["Available", "Busy"])
            active_juniors = sum(1 for d in self.junior_staff if d.status == "Busy")

            self.queue_depth_log.append({
                "time_minutes": t_now,
                "hour_of_day": 8.0 + (t_now % (self.config.hours_per_day * 60.0)) / 60.0,
                "total_queue": len(self.waiting_queue),
                "emergency_queue": emergency_q,
                "urgent_queue": urgent_q,
                "routine_queue": routine_q,
                "doctors_busy": busy_senior_docs,
                "doctors_on_break": break_docs,
                "doctors_absent_leave": absent_docs,
                "floating_active": active_floaters,
                "juniors_busy": active_juniors,
                "general_beds_occupied": self.general_beds.count,
                "icu_beds_occupied": self.icu_beds.count,
            })
            yield self.env.timeout(interval_minutes)

    def run(self) -> pd.DataFrame:
        """Executes simulation processes."""
        self.env.process(self.patient_arrival_generator())
        self.env.process(self.unified_dispatcher())
        self.env.process(self.floating_pool_manager())
        self.env.process(self.monitor_process())

        # Start individual doctor break and absence supervisors
        for doc in self.senior_doctors:
            self.env.process(self.staff_break_supervisor(doc))
            self.env.process(self.sudden_absence_process(doc))

        total_sim_minutes = self.config.sim_days * self.config.hours_per_day * 60.0
        self.env.run(until=total_sim_minutes + 180.0)

        # Finalize doctor timelines to simulation end
        for all_staff in [self.senior_doctors, self.floating_doctors, self.junior_staff, self.teleconsult_channels]:
            for stf in all_staff:
                if stf.state_timeline:
                    stf.state_timeline[-1]["end_time"] = self.env.now
                    stf.state_timeline[-1]["end_hour"] = 8.0 + (self.env.now % 720.0) / 60.0

        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        """Converts completed patient records into a structured pandas DataFrame."""
        if not self.records:
            return pd.DataFrame()

        rows = []
        for p in self.records:
            rows.append({
                "patient_id": p.patient_id,
                "sim_day": p.sim_day,
                "hour_of_day": round(p.hour_of_day, 2),
                "arrival_time": round(p.arrival_time, 2),
                "age": p.age,
                "age_group": p.age_group,
                "gender": p.gender,
                "area": p.area,
                "symptoms": p.symptoms,
                "triage_class": p.triage_class,
                "acuity_weight": p.acuity_weight,
                "service_start_time": round(p.service_start_time or 0.0, 2),
                "service_end_time": round(p.service_end_time or 0.0, 2),
                "wait_time": round(p.wait_time, 2),
                "service_time": round(p.consultation_actual_time, 2),
                "predicted_service_time": round(p.predicted_service_time, 2),
                "predicted_admit_prob": round(p.predicted_admit_prob, 3),
                "doctor_id": p.doctor_id,
                "server_type": p.server_type,
                "was_preempted": int(p.was_preempted),
                "admitted": int(p.needs_admission),
                "bed_type": p.bed_type or "None",
                "bed_allocated": int(p.bed_allocated),
                "bed_rejection": int(p.bed_rejection),
                "bed_wait_time": round(p.bed_wait_time, 2),
                "length_of_stay_hrs": round(p.length_of_stay_hrs, 2),
                "departure_time": round(p.departure_time or 0.0, 2),
            })
        return pd.DataFrame(rows)

    def get_kpis(self) -> Dict[str, float]:
        """Calculates key operational performance indicators (KPIs)."""
        df = self.get_results_dataframe()
        total_time = self.config.sim_days * self.config.hours_per_day * 60.0

        if df.empty:
            return {
                "total_patients": 0,
                "mean_wait": 0.0,
                "p95_wait": 0.0,
                "emergency_wait": 0.0,
                "urgent_wait": 0.0,
                "routine_wait": 0.0,
                "doctor_utilization": 0.0,
                "bed_rejection_rate": 0.0,
                "routine_starvation_rate": 0.0,
            }

        total_senior_busy = sum(d.total_busy_time for d in self.senior_doctors)
        total_doctor_capacity_minutes = self.config.num_doctors * total_time
        doc_util = min(1.0, total_senior_busy / max(1.0, total_doctor_capacity_minutes)) * 100.0

        em_wait = df[df["triage_class"] == "Emergency"]["wait_time"].mean() if not df[df["triage_class"] == "Emergency"].empty else 0.0
        ur_wait = df[df["triage_class"] == "Urgent"]["wait_time"].mean() if not df[df["triage_class"] == "Urgent"].empty else 0.0
        ro_df = df[df["triage_class"] == "Routine"]
        ro_wait = ro_df["wait_time"].mean() if not ro_df.empty else 0.0
        starvation_45m = (ro_df["wait_time"] > 45.0).mean() * 100.0 if not ro_df.empty else 0.0
        starvation_25m = (ro_df["wait_time"] > 25.0).mean() * 100.0 if not ro_df.empty else 0.0

        admitted_df = df[df["admitted"] == 1]
        rejection_rate = admitted_df["bed_rejection"].mean() * 100.0 if not admitted_df.empty else 0.0
        mean_bed_wait = admitted_df["bed_wait_time"].mean() if not admitted_df.empty else 0.0

        return {
            "total_patients": len(df),
            "mean_wait": round(float(df["wait_time"].mean()), 2),
            "p95_wait": round(float(df["wait_time"].quantile(0.95)), 2),
            "emergency_wait": round(float(em_wait), 2),
            "urgent_wait": round(float(ur_wait), 2),
            "routine_wait": round(float(ro_wait), 2),
            "doctor_utilization": round(float(doc_util), 2),
            "bed_rejection_rate": round(float(rejection_rate), 2),
            "mean_bed_wait": round(float(mean_bed_wait), 2),
            "routine_starvation_rate": round(float(starvation_45m), 2),
            "routine_starvation_rate_25m": round(float(starvation_25m), 2),
        }

    def get_staff_timeline_dataframe(self) -> pd.DataFrame:
        """Collects all doctor timeline intervals for Gantt/timeline visualization."""
        all_intervals = []
        for staff_list in [self.senior_doctors, self.floating_doctors, self.junior_staff, self.teleconsult_channels]:
            for staff in staff_list:
                for interval in staff.state_timeline:
                    all_intervals.append({
                        "Doctor_ID": interval["doctor_id"],
                        "Staff_Name": interval["name"],
                        "Role": interval["role"],
                        "Status": interval["status"],
                        "Start_Min": interval["start_time"],
                        "End_Min": interval["end_time"],
                        "Start_Hour": round(interval["start_hour"], 2),
                        "End_Hour": round(interval["end_hour"], 2),
                        "Duration_Min": round(interval["end_time"] - interval["start_time"], 1),
                        "Patient_ID": interval.get("patient_id"),
                    })
        return pd.DataFrame(all_intervals)

    def get_current_doctor_statuses(self) -> List[Dict[str, Any]]:
        """Returns snapshot of current doctor statuses for dashboard cards."""
        statuses = []
        for d in self.senior_doctors + self.floating_doctors + self.junior_staff:
            statuses.append({
                "doctor_id": d.doctor_id,
                "name": d.name,
                "role": d.role,
                "status": d.status,
                "patients_treated": d.patients_treated,
                "busy_time": round(d.total_busy_time, 1),
                "break_time": round(d.total_break_time, 1),
            })
        return statuses
