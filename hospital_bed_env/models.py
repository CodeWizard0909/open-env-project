# Copyright (c) 2026 Hospital Bed Env Team
# Hospital Bed & Resource Allocation Environment — Pydantic Models

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# =============================================================================
# Enums
# =============================================================================

class WardType(str, Enum):
    """Types of hospital wards."""
    ICU = "icu"
    GENERAL = "general"
    SURGICAL = "surgical"


class PatientOutcome(str, Enum):
    """Possible patient outcomes."""
    WAITING = "waiting"
    ADMITTED = "admitted"
    RECOVERED = "recovered"
    DETERIORATED = "deteriorated"
    DIED = "died"
    DISCHARGED = "discharged"


# =============================================================================
# Domain Models
# =============================================================================

class Patient(BaseModel):
    """A patient in the hospital system.

    Attributes:
        patient_id: Unique identifier
        severity: Patient severity 1-10 (1 = stable, 10 = most critical, >=11 = death)
        condition: Medical condition type
        ward_required: The clinically correct ward for this patient
        ward_current: Current ward placement (None if in ED queue)
        hours_waiting: Steps spent waiting in ED queue
        outcome: Current patient status
        steps_in_ward: Steps spent in current ward
    """
    patient_id: str
    severity: int = Field(ge=1, le=11, description="1=stable, 10=critical, >=11=death")
    condition: str
    ward_required: WardType
    ward_current: Optional[WardType] = None
    hours_waiting: int = Field(default=0, ge=0)
    outcome: PatientOutcome = PatientOutcome.WAITING
    steps_in_ward: int = Field(default=0, ge=0)


class WardState(BaseModel):
    """State of a single hospital ward."""
    name: WardType
    total_beds: int
    occupied_beds: int = 0
    staff_ratio: float = Field(default=1.0, description="Staff effectiveness multiplier")
    patients: List[str] = Field(default_factory=list, description="Patient IDs in this ward")


# =============================================================================
# OpenEnv Action
# =============================================================================

class PatientDecision(BaseModel):
    """A single decision about one patient."""
    patient_id: str
    action: Literal["admit_icu", "admit_general", "admit_surgical", "defer", "discharge"]
    discharge_patient_id: Optional[str] = Field(
        default=None,
        description="If admitting and ward is full, optionally discharge someone to make room"
    )


class HospitalAction(Action, BaseModel):
    """Action space: a list of decisions, one per patient in the ED queue."""
    decisions: List[PatientDecision] = Field(
        default_factory=list,
        description="List of decisions for patients in the ED queue"
    )


# =============================================================================
# OpenEnv Observation
# =============================================================================

class HospitalObservation(Observation, BaseModel):
    """Observation returned by the environment each step.

    Inherits done, reward, metadata from openenv Observation base.
    """
    step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=10, description="Maximum steps in episode")
    wards: Dict[str, WardState] = Field(default_factory=dict)
    ed_queue: List[Patient] = Field(default_factory=list, description="Patients waiting for decision")
    admitted: List[Patient] = Field(default_factory=list, description="Patients currently in wards")
    discharged: List[Patient] = Field(default_factory=list, description="Patients discharged this episode")
    events: List[str] = Field(default_factory=list, description="Human-readable event log")
    text_summary: str = Field(default="", description="LLM-readable text summary of current state")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OpenEnv State
# =============================================================================

class HospitalState(State, BaseModel):
    """Internal environment state for debugging/logging."""
    step: int = 0
    max_steps: int = 10
    task_id: str = "steady_state"
    seed: int = 42
    total_patients_seen: int = 0
    total_deaths: int = 0
    total_recovered: int = 0
    cumulative_reward: float = 0.0
    utilization_history: List[float] = Field(default_factory=list)
