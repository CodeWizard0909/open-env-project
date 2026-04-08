# Hospital Bed & Resource Allocation Environment

A fully-compliant OpenEnv environment for the Meta AI Hackathon (PyTorch + Hugging Face).

## Description

The **Hospital Bed Environment** simulates real-world hospital operations management. As an AI agent, you act as the hospital operations manager, making critical triage decisions:

- **Admit** patients from the Emergency Department (ED) queue to appropriate wards
- **Defer** patients when beds are unavailable (risking deterioration)
- **Transfer** patients between wards when needed
- **Discharge** recovered patients to free up beds

### Real-World Impact

This simulates the actual triage problem that healthcare systems face globally. Poor resource allocation contributes to 15-20% of preventable hospital deaths. The environment captures:

- Time-critical decisions (patients deteriorate while waiting)
- Resource contention (limited beds across multiple wards)
- Partial observability (patients may recover or worsen unpredictably)
- Multi-objective optimization (utilization vs. patient outcomes)

## Action Space

| Action | Description | Parameters |
|--------|-------------|------------|
| `admit_icu` | Admit patient to ICU | `patient_id`, optional `discharge_patient_id` |
| `admit_general` | Admit patient to General ward | `patient_id`, optional `discharge_patient_id` |
| `admit_surgical` | Admit patient to Surgical ward | `patient_id`, optional `discharge_patient_id` |
| `defer` | Keep patient in ED queue | `patient_id` |
| `discharge` | Discharge patient from ward | `patient_id` |

### Action Schema
```json
{
  "decisions": [
    {
      "patient_id": "P1234",
      "action": "admit_icu",
      "discharge_patient_id": null
    }
  ]
}
```

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Current step number |
| `max_steps` | int | Episode length |
| `wards` | dict | Ward states (beds, occupancy, staff ratio) |
| `ed_queue` | list | Patients waiting in ED |
| `admitted` | list | Patients currently in wards |
| `discharged` | list | Patients discharged this episode |
| `events` | list | Recent simulation events |
| `text_summary` | str | LLM-readable state summary |
| `reward` | float | Reward for this step |
| `done` | bool | Episode termination |

### Patient Attributes
- `patient_id`: Unique identifier
- `severity`: 1-10 scale (10 = most critical, 0 = death)
- `condition`: Medical condition type
- `ward_required`: Clinically appropriate ward
- `ward_current`: Current placement (None if in ED)
- `hours_waiting`: Steps spent in ED queue
- `outcome`: WAITING, ADMITTED, RECOVERED, DETERIORATED, DIED, DISCHARGED

## Tasks

### 1. steady_state (Easy)
- **Wards**: 1 General ward (10 beds)
- **Max Steps**: 10
- **Arrival Rate**: 3-5 patients/step
- **Seed**: 42
- **Description**: Single ward with stable patient flow. No surge events. Focus on basic triage.

### 2. surge_event (Medium)
- **Wards**: 3 wards (ICU: 4, General: 10, Surgical: 6 beds)
- **Max Steps**: 20
- **Arrival Rate**: 4-6 patients/step
- **Seed**: 7
- **Special Event**: At step 9, a surge of 10 critical patients (severity 7-10) arrives
- **Description**: Multi-ward triage with mid-episode capacity shock.

### 3. mass_casualty (Hard)
- **Wards**: 3 wards (ICU: 6, General: 15, Surgical: 8 beds)
- **Max Steps**: 30
- **Arrival Rate**: 4-6 patients/step
- **Seed**: 13
- **Special Events**:
  - Step 6: Mass casualty event (15 patients, severity 5-10)
  - Step 15: Staff shortage (ICU capacity halved)
- **Description**: Extreme resource contention with compounding crises.

## Reward Function

### Per-Step Rewards
| Component | Value | Trigger |
|-----------|-------|---------|
| Correct ward admit | +0.20 | Patient admitted to clinically appropriate ward |
| Patient recovery | +0.10 | Admitted patient severity increases |
| Good utilization | +0.05 | Total occupancy 60-85% |
| Wrong ward admit | -0.15 | Patient placed in incorrect ward |
| Deferred critical | -0.10 | Severity >= 8 deferred when bed available |
| Long ED wait | -0.05/patient | Patient waiting > 3 steps in ED |
| Preventable death | -0.30 | Patient dies (severity <= 0) |

### Terminal Bonuses/Penalties
| Component | Value | Trigger |
|-----------|-------|---------|
| Zero deaths | +0.50 | No preventable deaths in episode |
| Good avg utilization | +0.20 | Average occupancy 65-80% |
| Critical response | +0.10 | All severity >= 9 admitted within 1 step |
| Each death | -0.50 | Per preventable death |

### Normalized Score
Raw scores are normalized to [0.0, 1.0] per task:
- `steady_state`: range [-2.0, 6.0]
- `surge_event`: range [-5.0, 10.0]
- `mass_casualty`: range [-10.0, 15.0]

## Setup & Usage

### Local Development

```bash
# Install dependencies
cd hospital_bed_env
pip install -r requirements.txt

# Run the server
uvicorn hospital_bed_env.server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run baseline inference
export API_BASE_URL="http://localhost:7860"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your-key-here"
python inference.py

# Run the validation script to check compliance
python validate.py

# Run unit tests
python -m pytest tests/
```

### Docker Deployment

```bash
# Build
docker build -t hospital-bed-env .

# Run
docker run -p 7860:7860 hospital-bed-env

# Test endpoints
curl http://localhost:7860/reset/steady_state
curl -X POST http://localhost:7860/step/steady_state \
  -H "Content-Type: application/json" \
  -d '{"decisions": []}'
curl http://localhost:7860/state/steady_state
```

### Hugging Face Spaces

1. Create a new Space with Docker runtime
2. Push this repository to your Space
3. The server will automatically start on port 7860

Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/hospital-bed-env`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/reset/{task_id}` | POST | Reset environment, returns initial observation |
| `/step/{task_id}` | POST | Execute action, returns (observation, reward, done, info) |
| `/state/{task_id}` | GET | Get full internal state for debugging |

## Baseline Scores

Using the heuristic priority-triage agent:

| Task | Expected Score | Description |
|------|----------------|-------------|
| `steady_state` | 0.65 - 0.80 | Easy mode, should achieve high scores |
| `surge_event` | 0.45 - 0.65 | Medium mode, surge causes some deaths |
| `mass_casualty` | 0.25 - 0.45 | Hard mode, significant resource contention |

## Project Structure

```
hospital_bed_env/
├── __init__.py           # Package exports
├── models.py             # Pydantic typed models
├── client.py             # OpenEnv client wrapper
├── server/
│   ├── app.py            # FastAPI server entry
│   ├── hospital_environment.py  # OpenEnv wrapper
│   ├── simulation.py     # Core simulation engine
│   ├── grader.py         # Reward calculation
│   └── tasks.py          # Task configurations
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Package configuration
└── README.md             # This file

baselines/
└── heuristic_baseline.py  # Simple triage-priority agent

tests/
├── test_simulation.py     # Unit tests for core logic
└── test_grader.py         # Unit tests for rewards

logs/                      # Execution logs and results
inference.py               # Root LLM-based inference script
validate.py                # Environment compliance checker
```

## OpenEnv Compliance Checklist

- [x] Typed Pydantic models (Observation, Action, State)
- [x] `step(action)` returns (observation, reward, done, info)
- [x] `reset()` returns initial observation
- [x] `state()` returns internal state
- [x] `openenv.yaml` with metadata
- [x] 3 tasks with agent graders (easy → hard)
- [x] Meaningful reward with partial progress signals
- [x] Baseline inference script with reproducible scores
- [x] Working Dockerfile
- [x] Deployable to Hugging Face Spaces

## License

MIT License - Meta AI Hackathon 2026
