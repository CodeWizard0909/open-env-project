# Copyright (c) 2026 Hospital Bed Env Team
# Reward calculation and grading module

from typing import Any, Dict
from hospital_bed_env.models import HospitalState, WardState


class RewardCalculator:
    """Calculates per-step components and normalizes terminal score."""

    def __init__(self):
        # Tracking for terminal evaluation
        self.total_deaths = 0
        self.utilizations = []
        self.sev9_delayed = 0

    def score(self, sim_results: Dict[str, Any], sim) -> float:
        """Compute the per-step reward based on the simulator's output.

        Severity scale: 1=stable, 10=critical, >=11=death
        Recovery: severity decreases
        Deterioration: severity increases
        """
        reward = 0.0

        # Action-based components from apply_decisions
        wrong_ward = sim_results.get("wrong_ward_admits", 0)
        deferred_crit = sim_results.get("deferred_criticals", 0)

        # Advance-time-based components
        recoveries = sim_results.get("recoveries", 0)
        deaths = sim_results.get("preventable_deaths", 0)

        self.total_deaths += deaths

        # 1. Patient admitted to correct ward (+0.20)
        valid_admits = 0
        for decision in sim.last_decisions:
            if decision.action.startswith("admit_") and decision.patient_id in sim.admitted_patients:
                p = sim.admitted_patients[decision.patient_id]
                if p.ward_current == p.ward_required:
                    valid_admits += 1

        reward += valid_admits * 0.20

        # 2. Patient recovery - severity decreased (+0.10)
        reward += recoveries * 0.10

        # 3. Utilization 60-85% (+0.05)
        total_beds = sum(w.total_beds for w in sim.wards.values())
        occ_beds = sum(w.occupied_beds for w in sim.wards.values())
        if total_beds > 0:
            util = occ_beds / total_beds
            self.utilizations.append(util)
            if 0.60 <= util <= 0.85:
                reward += 0.05

        # 4. Preventable death (-0.30)
        reward -= deaths * 0.30

        # 5. Wrong ward admit (-0.15)
        reward -= wrong_ward * 0.15

        # 6. Critical patient (severity >= 8) deferred when bed available (-0.10)
        reward -= deferred_crit * 0.10

        # 7. Patient waits >3 steps in ED (-0.05 per patient)
        long_waiters = sum(1 for p in sim.ed_queue if p.hours_waiting > 3)
        reward -= long_waiters * 0.05

        # Track severity >= 9 patients delayed more than 1 step
        for p in sim.ed_queue:
            if p.severity >= 9 and p.hours_waiting > 1:
                self.sev9_delayed += 1

        return round(reward, 2)

    def terminal_score(self, sim) -> float:
        """Compute the final terminal episode reward."""
        reward = 0.0
        
        # +0.50 Zero preventable deaths
        if self.total_deaths == 0:
            reward += 0.50
            
        # +0.20 Average bed utilization 65-80%
        if self.utilizations:
            avg_util = sum(self.utilizations) / len(self.utilizations)
            if 0.65 <= avg_util <= 0.80:
                reward += 0.20
                
        # +0.10 All severity >=9 patients admitted within 1 step
        if self.sev9_delayed == 0:
            reward += 0.10
            
        # -0.50 Each preventable death
        reward -= self.total_deaths * 0.50
        
        return round(reward, 10)

    def normalize(self, raw_score: float, task_id: str) -> float:
        """Normalize score to [0.0, 1.0] range."""
        # Realistic min/max for the 3 tasks
        ranges = {
            "steady_state": {"min": -2.0, "max": 6.0},
            "surge_event": {"min": -5.0, "max": 10.0},
            "mass_casualty": {"min": -10.0, "max": 15.0},
        }
        
        rng = ranges.get(task_id, {"min": -5.0, "max": 10.0})
        min_p = rng["min"]
        max_p = rng["max"]
        
        score = (raw_score - min_p) / (max_p - min_p)
        return max(0.0, min(1.0, score))
