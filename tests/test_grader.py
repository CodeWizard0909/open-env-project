import pytest
from hospital_bed_env.server.grader import RewardCalculator
from hospital_bed_env.server.simulation import HospitalSimulator
from hospital_bed_env.server.tasks import TASK_REGISTRY

@pytest.fixture
def grader():
    return RewardCalculator()

@pytest.fixture
def simulator():
    config = TASK_REGISTRY["steady_state"]
    sim = HospitalSimulator(config)
    sim.reset(config, seed=42)
    return sim

def test_initial_score(grader, simulator):
    # No actions yet
    score = grader.score({}, simulator)
    assert score == 0.0

def test_reward_for_admit(grader, simulator):
    # Manually admit a patient to the correct ward
    patient = simulator.ed_queue[0]
    patient.ward_current = patient.ward_required
    simulator.admitted_patients[patient.patient_id] = patient
    simulator.ed_queue.remove(patient)
    
    # Mock the behavior of apply_decisions
    from hospital_bed_env.models import PatientDecision
    simulator.last_decisions = [PatientDecision(patient_id=patient.patient_id, action="admit_general")]
    
    score = grader.score({}, simulator)
    assert score == 0.20 # Reward for correct ward admit

def test_penalty_for_death(grader, simulator):
    metrics = {"preventable_deaths": 1}
    score = grader.score(metrics, simulator)
    assert score == -0.30

def test_terminal_score(grader, simulator):
    # Set stats for terminal score
    grader.total_deaths = 0
    grader.utilizations = [0.75, 0.70] # Average in [0.65, 0.80]
    grader.sev9_delayed = 0
    
    terminal = grader.terminal_score(simulator)
    assert terminal == 0.80 # 0.50 (zero deaths) + 0.20 (util) + 0.10 (sev9)

def test_normalization(grader):
    # Steady state range [-2.0, 6.0]
    assert grader.normalize(-2.0, "steady_state") == 0.0
    assert grader.normalize(6.0, "steady_state") == 1.0
    assert grader.normalize(2.0, "steady_state") == 0.5
