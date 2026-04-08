# Copyright (c) 2026 Hospital Bed Env Team
# Pure Python simulation engine for the Hospital Environment

import copy
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from hospital_bed_env.models import (
    Patient,
    WardState,
    WardType,
    PatientOutcome,
    PatientDecision,
)


class HospitalSimulator:
    """Core simulation engine that advances the world state."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rng = np.random.RandomState(42)
        
        # State variables
        self.step_idx = 0
        self.wards: Dict[WardType, WardState] = {}
        self.ed_queue: List[Patient] = []
        self.admitted_patients: Dict[str, Patient] = {}
        self.discharged_patients: List[Patient] = []
        self.events: List[str] = []
        
        # Action tracking for reward
        self.last_decisions: List[PatientDecision] = []
        self.wrong_ward_admits: int = 0
        self.deferred_criticals: int = 0
        self.preventable_deaths: int = 0
        self.expected_recoveries: int = 0
        
        # Predefined conditions per ward type
        self.CONDITIONS = {
            WardType.ICU: ["cardiac", "severe_trauma"],
            WardType.GENERAL: ["general", "respiratory"],
            WardType.SURGICAL: ["surgical", "trauma"],
        }

    def reset(self, config: Dict[str, Any], seed: Optional[int] = None):
        """Reset the simulation to initial state."""
        self.config = config
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        self.step_idx = 0
        self.wards = {}
        for w_type, w_cfg in config["wards"].items():
            self.wards[WardType(w_type)] = WardState(
                name=WardType(w_type),
                total_beds=w_cfg["beds"],
                staff_ratio=w_cfg.get("staff_ratio", 1.0)
            )
            
        self.ed_queue = []
        self.admitted_patients = {}
        self.discharged_patients = []
        self.events = ["Simulation initialized."]
        
        self._spawn_patients(config.get("initial_patients", 3), config.get("initial_severity_range", (2, 6)))

    def apply_decisions(self, decisions: List[PatientDecision]) -> Dict[str, Any]:
        """Apply agent decisions and return metrics for the grader."""
        self.events.clear()
        self.last_decisions = decisions
        self.wrong_ward_admits = 0
        self.deferred_criticals = 0
        
        # Map queue for fast lookup
        queue_dict = {p.patient_id: p for p in self.ed_queue}
        
        processed_ids = set()
        
        for decision in decisions:
            if decision.patient_id not in queue_dict:
                continue # Invalid decision, patient not in ED
                
            processed_ids.add(decision.patient_id)
            patient = queue_dict[decision.patient_id]
            
            # Handle discharges first to make room
            if decision.discharge_patient_id:
                self._handle_discharge(decision.discharge_patient_id)
            
            # Handle admitting
            if decision.action.startswith("admit_"):
                target_ward_str = decision.action.split("_")[1]
                target_ward = WardType(target_ward_str)
                self._handle_admit(patient, target_ward)
            
            # Handle explicit deferral
            elif decision.action == "defer":
                self._handle_defer(patient)
                
            elif decision.action == "discharge":
                pid = decision.discharge_patient_id
                if pid and pid in self.admitted_patients:
                    self._handle_discharge(pid)
                # Discharging someone directly from the ED? That's an immediate discharge.
                if patient in self.ed_queue:
                    self.ed_queue.remove(patient)
                    patient.outcome = PatientOutcome.DISCHARGED
                    self.discharged_patients.append(patient)
                    self.events.append(f"Discharged patient {patient.patient_id} from ED.")
        
        # Implicitly defer anyone not processed
        for patient in self.ed_queue:
            if patient.patient_id not in processed_ids:
                self._handle_defer(patient)
                
        return {
            "wrong_ward_admits": self.wrong_ward_admits,
            "deferred_criticals": self.deferred_criticals,
        }

    def advance_time(self) -> Dict[str, Any]:
        """Advance the simulation by one step."""
        self.step_idx += 1
        
        # Reset counters
        self.expected_recoveries = 0
        self.preventable_deaths = 0
        
        # Check for scripted events
        self._check_scripted_events()
        
        # 1. Update waiting patients in ED
        for patient in self.ed_queue:
            patient.hours_waiting += 1
            # Patients deteriorate while waiting (severity INCREASES)
            if patient.severity >= 5:  # Urgent and critical patients deteriorate
                patient.severity += 1
                if patient.severity >= 8:
                    self.events.append(f"Patient {patient.patient_id} deteriorated to critical in ED!")

            # Death occurs when severity exceeds critical threshold
            if patient.severity >= 11:
                self.preventable_deaths += 1
                patient.outcome = PatientOutcome.DIED
                self.events.append(f"Patient {patient.patient_id} died in ED queue.")
        
        # Remove dead patients from ED queue
        self.ed_queue = [p for p in self.ed_queue if p.outcome != PatientOutcome.DIED]
        
        # 2. Update admitted patients
        to_discharge = []
        for pid, patient in self.admitted_patients.items():
            patient.steps_in_ward += 1

            # Staff efficiency affects recovery chance
            ward = self.wards[patient.ward_current]
            staff_mod = ward.staff_ratio

            # SEVERITY SCALE: 1-10 where 10 = most critical, 1 = stable
            # Recovery: severity DECREASES (patient becomes less critical)
            # Deterioration: severity INCREASES (patient becomes more critical)
            # Death: severity reaches 11+ (beyond critical, system failure)
            # Discharge ready: severity <= 3 (stable enough to go home)

            # If in the wrong ward, patient deteriorates
            if patient.ward_current != patient.ward_required:
                if self.rng.random() < 0.5:
                    patient.severity += 1  # Deteriorate (severity goes UP)
                    self.events.append(f"Patient {pid} deteriorated in wrong ward.")
            else:
                # Correct ward -> chance to recover (severity goes DOWN)
                if self.rng.random() < (0.8 * staff_mod):
                    patient.severity -= 1  # Recover (severity goes DOWN)
                    self.expected_recoveries += 1

            # Check for death (severity exceeded critical threshold)
            if patient.severity >= 11:
                self.preventable_deaths += 1
                patient.outcome = PatientOutcome.DIED
                self.events.append(f"Patient {patient.patient_id} died in {patient.ward_current}.")
                to_discharge.append(pid)
            elif patient.severity <= 3:
                # Fully recovered - stable enough for discharge
                patient.outcome = PatientOutcome.RECOVERED
                patient.severity = max(1, patient.severity)  # Clamp to valid range
                self.events.append(f"Patient {patient.patient_id} recovered in {patient.ward_current}.")
        
        # Remove dead patients from wards
        for pid in to_discharge:
            patient = self.admitted_patients[pid]
            ward = self.wards[patient.ward_current]
            ward.patients.remove(pid)
            ward.occupied_beds -= 1
            del self.admitted_patients[pid]
            self.discharged_patients.append(patient)
            
        # 3. Spawn new arrivals based on task config
        arr_rate_min, arr_rate_max = self.config.get("arrival_rate", (0, 0))
        if arr_rate_max > 0:
            num_arrivals = self.rng.randint(arr_rate_min, arr_rate_max + 1)
            if num_arrivals > 0:
                sev_range = self.config.get("arrival_severity", (2, 7))
                self._spawn_patients(num_arrivals, sev_range)
                
        return {
            "recoveries": self.expected_recoveries,
            "preventable_deaths": self.preventable_deaths,
        }

    # --- Private Helpers ---

    def _spawn_patients(self, count: int, severity_range: Tuple[int, int]):
        """Spawn N new patients into the ED queue."""
        wards_list = list(self.wards.keys())
        for _ in range(count):
            ward = wards_list[self.rng.randint(0, len(wards_list))]
            sev = self.rng.randint(severity_range[0], severity_range[1] + 1)
            
            conds = self.CONDITIONS[ward]
            cond = conds[self.rng.randint(0, len(conds))]
            
            p = Patient(
                patient_id=f"P{str(uuid.uuid4())[:4].upper()}",
                severity=sev,
                condition=cond,
                ward_required=ward,
                ward_current=None,
                outcome=PatientOutcome.WAITING
            )
            self.ed_queue.append(p)
            self.events.append(f"New arrival: {p.patient_id} (sev {p.severity}, needs {p.ward_required}).")

    def _handle_discharge(self, patient_id: str):
        if patient_id in self.admitted_patients:
            patient = self.admitted_patients[patient_id]
            ward = self.wards[patient.ward_current]
            ward.occupied_beds -= 1
            ward.patients.remove(patient_id)
            del self.admitted_patients[patient_id]
            
            patient.outcome = PatientOutcome.DISCHARGED
            self.discharged_patients.append(patient)
            self.events.append(f"Discharged patient {patient_id} from {ward.name}.")

    def _handle_admit(self, patient: Patient, target_ward: WardType):
        if target_ward not in self.wards:
            self.events.append(f"Failed to admit {patient.patient_id}: Invalid ward {target_ward}.")
            return
            
        ward = self.wards[target_ward]
        if ward.occupied_beds >= ward.total_beds:
            self.events.append(f"Failed to admit {patient.patient_id} to {target_ward}: No beds.")
            return

        # Success admit
        self.ed_queue.remove(patient)
        patient.ward_current = target_ward
        patient.outcome = PatientOutcome.ADMITTED
        self.admitted_patients[patient.patient_id] = patient
        ward.occupied_beds += 1
        ward.patients.append(patient.patient_id)
        
        if target_ward != patient.ward_required:
            self.wrong_ward_admits += 1
            self.events.append(f"Admitted {patient.patient_id} to WRONG WARD {target_ward}.")
        else:
            self.events.append(f"Admitted {patient.patient_id} to {target_ward}.")

    def _handle_defer(self, patient: Patient):
        if patient.severity >= 8:
            # Check if there was actually a bed available in any valid ward
            ward = self.wards.get(patient.ward_required)
            if ward and ward.occupied_beds < ward.total_beds:
                self.deferred_criticals += 1
                self.events.append(f"Avoidable deferral of critical patient {patient.patient_id}!")

    def _check_scripted_events(self):
        """Check for and apply scripted scenario events like surges or staff shortages."""
        if "events" not in self.config:
            return
            
        for event in self.config["events"]:
            if event["step"] == self.step_idx:
                e_type = event["type"]
                if e_type == "surge":
                    sev_range = event.get("severity", (7, 10))
                    self._spawn_patients(event["count"], sev_range)
                    self.events.append(f"⚠️ SURGE EVENT: {event['count']} patients arrived!")
                elif e_type == "staff_shortage":
                    amt = event.get("amount", 0.5)
                    self.events.append(f"⚠️ STAFF SHORTAGE! ICU capacity halved.")
                    # Specifically halving ICU capacity based on hard task config
                    if WardType.ICU in self.wards:
                        self.wards[WardType.ICU].total_beds = max(1, int(self.wards[WardType.ICU].total_beds * amt))
                        self.wards[WardType.ICU].staff_ratio = 0.5
