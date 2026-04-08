# Copyright (c) 2026 Hospital Bed Env Team
# OpenEnv WebSocket Client

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import HospitalAction, HospitalObservation, HospitalState


class HospitalEnv(EnvClient[HospitalAction, HospitalObservation, HospitalState]):
    """Client for connecting to the Hospital Bed Environment."""

    def _step_payload(self, action: HospitalAction) -> Dict[str, Any]:
        """Convert a HospitalAction to JSON-serializable dict."""
        # Using model_dump() from Pydantic v2
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[HospitalObservation]:
        """Convert a JSON response to StepResult."""
        obs_payload = payload.get("observation", {})
        obs = HospitalObservation(**obs_payload)
        
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> HospitalState:
        """Convert a JSON response to HospitalState."""
        return HospitalState(**payload)