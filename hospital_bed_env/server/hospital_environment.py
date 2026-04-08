# Copyright (c) 2026 Hospital Bed Env Team
# OpenEnv Environment Wrapper

import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

from hospital_bed_env.models import HospitalAction, HospitalObservation, HospitalState, PatientOutcome
from .simulation import HospitalSimulator
from .grader import RewardCalculator
from .tasks import TASK_REGISTRY


class HospitalEnvironment(Environment[HospitalAction, HospitalObservation, HospitalState]):
    """OpenEnv wrapper connecting the engine to the standard API."""
    
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    action_type = HospitalAction
    observation_type = HospitalObservation
    state_type = HospitalState

    def __init__(
        self,
        task_id: str = "steady_state",
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.task_id = task_id
        if task_id not in TASK_REGISTRY:
            self.task_id = "steady_state"
            
        self._sim = HospitalSimulator(TASK_REGISTRY[self.task_id])
        self._sim.reset(TASK_REGISTRY[self.task_id], seed=TASK_REGISTRY[self.task_id]["seed"])
        self._grader = RewardCalculator()
        self._state = HospitalState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=self.task_id,
            seed=TASK_REGISTRY[self.task_id]["seed"]
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> HospitalObservation:
        """Reset the environment."""
        task_id = kwargs.get("task_id", self.task_id)
        if task_id in TASK_REGISTRY:
            self.task_id = task_id
            
        config = TASK_REGISTRY[self.task_id]
        used_seed = seed if seed is not None else config["seed"]
        
        self._sim.reset(config, seed=used_seed)
        self._grader = RewardCalculator()
        
        self._state = HospitalState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=self.task_id,
            seed=used_seed,
            max_steps=config["max_steps"]
        )
        self._metadata = {}
        
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: HospitalAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HospitalObservation:
        """Execute action, advance time, and compute reward."""
        self._state.step_count += 1
        
        # Apply agent decisions
        sim_results = self._sim.apply_decisions(action.decisions)
        
        # Advance simulation time
        adv_results = self._sim.advance_time()
        sim_results.update(adv_results)
        
        # Compute step reward
        reward = self._grader.score(sim_results, self._sim)
        self._state.cumulative_reward += reward
        
        # Check termination
        done = self._state.step_count >= self._state.max_steps
        
        if done:
            # Add terminal reward
            terminal = self._grader.terminal_score(self._sim)
            reward += terminal
            
            # Grader specifically needs a normalized score [0.0, 1.0] for the Hackathon
            # we provide the normalized score in the metadata
            norm_score = self._grader.normalize(self._state.cumulative_reward + terminal, self.task_id)
            self._state.cumulative_reward += terminal
            self._metadata["normalized_score"] = norm_score

        return self._build_observation(reward=reward, done=done)

    @property
    def state(self) -> HospitalState:
        """Return the current internal state."""
        # Update dynamic state fields
        self._state.step = self._sim.step_idx
        self._state.total_patients_seen = len(self._sim.admitted_patients) + len(self._sim.discharged_patients)
        self._state.total_deaths = self._grader.total_deaths
        self._state.total_recovered = sum(1 for p in self._sim.discharged_patients if p.outcome == PatientOutcome.RECOVERED)
        self._state.utilization_history = list(self._grader.utilizations)
        return self._state

    def _build_observation(self, reward: float, done: bool) -> HospitalObservation:
        """Construct the observation object from the simulator state."""
        
        # Build text summary for LLM
        lines = ["WARD STATUS:"]
        for w in self._sim.wards.values():
            pct = 0.0
            if w.total_beds > 0:
                pct = w.occupied_beds / w.total_beds
            status = "CRITICAL" if pct > 0.9 else "WARNING" if pct > 0.7 else "OK"
            lines.append(f"  {w.name.upper()}: {w.occupied_beds}/{w.total_beds} beds [{status}]")
            
        lines.append("\nWAITING PATIENTS (ED QUEUE):")
        if not self._sim.ed_queue:
            lines.append("  (empty)")
        else:
            for p in self._sim.ed_queue:
                urgency = "CRITICAL" if p.severity >= 8 else "URGENT" if p.severity >= 5 else "ROUTINE"
                lines.append(
                    f"  {p.patient_id} | {urgency} | sev: {p.severity}/10 | "
                    f"condition: {p.condition} | waited: {p.hours_waiting} steps | needs: {p.ward_required.upper()}"
                )
                
        if self._sim.events:
            lines.append("\nRECENT EVENTS:")
            for ev in self._sim.events[-3:]:
                lines.append(f"  {ev}")
                
        return HospitalObservation(
            step=self._state.step_count,
            max_steps=self._state.max_steps,
            wards={w.value: w_state for w, w_state in self._sim.wards.items()},
            ed_queue=list(self._sim.ed_queue),
            admitted=list(self._sim.admitted_patients.values()),
            discharged=list(self._sim.discharged_patients),
            events=list(self._sim.events),
            text_summary="\n".join(lines),
            reward=reward,
            done=done,
            metadata=self._metadata.copy() if hasattr(self, '_metadata') else {}
        )
