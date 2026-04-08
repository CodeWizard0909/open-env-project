#!/usr/bin/env python3
"""Baseline inference script for Hospital Bed Environment.

This script runs a simple heuristic agent against all 3 tasks and produces reproducible scores.
The agent uses a priority-based triage system:
1. Always admit critical patients (severity >= 8) first
2. Admit to correct ward if bed available
3. Defer only when no beds available
4. Discharge recovered patients (severity <= 3) to free beds

Severity scale: 1=stable, 10=critical, >=11=death
Recovery: severity decreases toward 1
Deterioration: severity increases toward 11
"""

import json
import os
import sys
from typing import Dict, List, Any
import requests

# API configuration from environment
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")

# Task configurations
TASKS = ["steady_state", "surge_event", "mass_casualty"]


def run_heuristic_agent(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Simple heuristic agent that makes triage decisions.

    Priority order:
    1. Discharge recovered patients (severity <= 3) to free beds
    2. Admit critical patients (severity >= 8) to correct ward
    3. Admit urgent patients (severity 5-7) to correct ward
    4. Defer routine patients (severity < 5) if ward is full
    """
    decisions = []

    # Get ward states
    wards = observation.get("wards", {})
    ed_queue = observation.get("ed_queue", [])
    admitted = observation.get("admitted", [])

    # Calculate bed availability per ward
    bed_availability = {}
    for ward_name, ward_state in wards.items():
        bed_availability[ward_name] = ward_state["total_beds"] - ward_state["occupied_beds"]

    # First: discharge recovered patients to free beds (severity <= 3 = stable)
    for patient in admitted:
        if patient.get("severity", 10) <= 3:
            decisions.append({
                "patient_id": patient["patient_id"],
                "action": "discharge",
                "discharge_patient_id": patient["patient_id"]
            })

    # Sort ED queue by severity (highest first - most critical)
    sorted_queue = sorted(ed_queue, key=lambda p: p.get("severity", 0), reverse=True)

    for patient in sorted_queue:
        pid = patient["patient_id"]
        severity = patient.get("severity", 5)
        ward_required = patient.get("ward_required", "general")

        # Critical patients (severity >= 8) - highest priority
        if severity >= 8:
            if bed_availability.get(ward_required, 0) > 0:
                # Admit to correct ward
                action = f"admit_{ward_required}"
                decisions.append({
                    "patient_id": pid,
                    "action": action,
                    "discharge_patient_id": None
                })
                bed_availability[ward_required] -= 1
            else:
                # No bed - defer (in a real scenario, could transfer from another ward)
                decisions.append({
                    "patient_id": pid,
                    "action": "defer",
                    "discharge_patient_id": None
                })

        # Urgent patients (severity 5-7)
        elif severity >= 5:
            if bed_availability.get(ward_required, 0) > 0:
                action = f"admit_{ward_required}"
                decisions.append({
                    "patient_id": pid,
                    "action": action,
                    "discharge_patient_id": None
                })
                bed_availability[ward_required] -= 1
            else:
                # Defer if no bed
                decisions.append({
                    "patient_id": pid,
                    "action": "defer",
                    "discharge_patient_id": None
                })

        # Routine patients (severity < 5)
        else:
            if bed_availability.get(ward_required, 0) > 0:
                action = f"admit_{ward_required}"
                decisions.append({
                    "patient_id": pid,
                    "action": action,
                    "discharge_patient_id": None
                })
                bed_availability[ward_required] -= 1
            else:
                # Safe to defer routine patients
                decisions.append({
                    "patient_id": pid,
                    "action": "defer",
                    "discharge_patient_id": None
                })

    return decisions


def run_episode(task_id: str, session: requests.Session) -> Dict[str, Any]:
    """Run a single episode against the specified task."""

    # Reset environment
    reset_url = f"{API_BASE_URL}/reset"
    reset_response = session.post(reset_url, json={"task_id": task_id})
    reset_response.raise_for_status()
    data = reset_response.json()
    observation = data.get("observation", {})

    print(json.dumps({"type": "START", "task": task_id, "step": 0}))

    total_reward = 0.0
    step = 0
    max_steps = observation.get("max_steps", 10)

    while not observation.get("done", False):
        step += 1

        # Get agent decisions using heuristic
        decisions = run_heuristic_agent(observation)

        # Step the environment
        step_url = f"{API_BASE_URL}/step"
        step_response = session.post(
            step_url,
            json={"action": {"decisions": decisions}},
            headers={"Content-Type": "application/json"}
        )
        step_response.raise_for_status()
        data = step_response.json()
        observation = data.get("observation", {})

        reward = observation.get("reward", 0.0)
        total_reward += reward

        print(json.dumps({
            "type": "STEP",
            "step": step,
            "reward": round(reward, 3),
            "total_reward": round(total_reward, 3),
            "ed_queue_size": len(observation.get("ed_queue", [])),
            "events": observation.get("events", [])[-2:]
        }))

        if step >= max_steps:
            break

    # Get final state and normalized score
    state_url = f"{API_BASE_URL}/state"
    state_response = session.get(state_url)
    state_response.raise_for_status()
    final_state = state_response.json()

    # Get normalized score from metadata
    normalized_score = 0.0
    if "metadata" in final_state:
        normalized_score = final_state.get("metadata", {}).get("normalized_score", 0.0)

    print(json.dumps({
        "type": "END",
        "task": task_id,
        "total_reward": round(total_reward, 3),
        "normalized_score": round(normalized_score, 3),
        "stats": {
            "total_patients_seen": final_state.get("total_patients_seen", 0),
            "total_deaths": final_state.get("total_deaths", 0),
            "total_recovered": final_state.get("total_recovered", 0)
        }
    }))

    return {
        "task": task_id,
        "total_reward": round(total_reward, 3),
        "normalized_score": round(normalized_score, 3),
        "stats": final_state
    }


def main():
    """Run baseline inference on all tasks."""
    print("=" * 60)
    print("HOSPITAL BED ENVIRONMENT - BASELINE INFERENCE")
    print("=" * 60)
    print(f"API Base URL: {API_BASE_URL}")
    print()

    # Use requests session for HTTP calls
    session = requests.Session()

    results = {}
    for task_id in TASKS:
        print(f"\n{'=' * 40}")
        print(f"Running task: {task_id}")
        print("=" * 40)
        try:
            results[task_id] = run_episode(task_id, session)
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to {API_BASE_URL}")
            print("Make sure the server is running: uvicorn hospital_bed_env.server.app:app --host 0.0.0.0 --port 7860")
            results[task_id] = {"error": "Connection failed"}
            break
        except Exception as e:
            print(f"Error running {task_id}: {e}")
            results[task_id] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for task_id, result in results.items():
        if "error" in result:
            print(f"{task_id}: ERROR - {result['error']}")
        else:
            print(f"{task_id}: score={result['normalized_score']:.3f}, reward={result['total_reward']:.3f}")

    print(json.dumps({"type": "FINAL_SCORES", "scores": results}))


if __name__ == "__main__":
    main()
