# Copyright (c) 2026 Hospital Bed Env Team
# Scenario definitions for the 3 hackathon tasks

TASK_REGISTRY = {
    # ---------------------------------------------------------
    # Easy: 10 steps, general ward only, no surge, seed 42
    # ---------------------------------------------------------
    "steady_state": {
        "task_id": "steady_state",
        "seed": 42,
        "max_steps": 10,
        "arrival_rate": (3, 5),          # 3-5 patients per step
        "arrival_severity": (2, 6),      # Severity 2-6
        "initial_patients": 5,           # Start with 5 waiting
        "initial_severity_range": (2, 6),
        "wards": {
            "general": {"beds": 10, "staff_ratio": 1.0},
        },
        "events": []
    },
    
    # ---------------------------------------------------------
    # Medium: 20 steps, 3 wards, surge at step 9, seed 7
    # ---------------------------------------------------------
    "surge_event": {
        "task_id": "surge_event",
        "seed": 7,
        "max_steps": 20,
        "arrival_rate": (4, 6),
        "arrival_severity": (3, 7),
        "initial_patients": 5,
        "initial_severity_range": (3, 7),
        "wards": {
            "icu": {"beds": 4, "staff_ratio": 1.0},
            "general": {"beds": 10, "staff_ratio": 1.0},
            "surgical": {"beds": 6, "staff_ratio": 1.0},
        },
        "events": [
            {
                "step": 9,
                "type": "surge",
                "count": 10,
                "severity": (7, 10)  # Severity 7-10 spike
            }
        ]
    },
    
    # ---------------------------------------------------------
    # Hard: 30 steps, 3 wards, mass casualty + staff shortage, seed 13
    # ---------------------------------------------------------
    "mass_casualty": {
        "task_id": "mass_casualty",
        "seed": 13,
        "max_steps": 30,
        "arrival_rate": (4, 6),
        "arrival_severity": (3, 7),
        "initial_patients": 6,
        "initial_severity_range": (3, 7),
        "wards": {
            "icu": {"beds": 6, "staff_ratio": 1.0},
            "general": {"beds": 15, "staff_ratio": 1.0},
            "surgical": {"beds": 8, "staff_ratio": 1.0},
        },
        "events": [
            {
                "step": 6,
                "type": "surge",
                "count": 15,         # Mass casualty event
                "severity": (5, 10)
            },
            {
                "step": 15,
                "type": "staff_shortage",
                "amount": 0.5        # ICU capacity halved
            }
        ]
    }
}
