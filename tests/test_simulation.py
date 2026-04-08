import pytest
from hospital_bed_env.server.simulation import HospitalSimulator
from hospital_bed_env.server.tasks import TASK_REGISTRY
from hospital_bed_env.models import PatientDecision, WardType

@pytest.fixture
def simulator():
    config = TASK_REGISTRY["steady_state"]
    sim = HospitalSimulator(config)
    sim.reset(config, seed=42)
    return sim

def test_simulator_initialization(simulator):
    assert simulator.step_idx == 0
    assert len(simulator.wards) == 1
    assert WardType.GENERAL in simulator.wards
    assert len(simulator.ed_queue) == 5  # Based on steady_state config

def test_simulator_step(simulator):
    # Apply empty decisions (defer all)
    metrics = simulator.apply_decisions([])
    assert len(simulator.last_decisions) == 0
    
    # Advance time
    advance_metrics = simulator.advance_time()
    assert simulator.step_idx == 1
    assert len(simulator.ed_queue) >= 5 # 5 initially + arrivals - deaths
    assert "recoveries" in advance_metrics
    assert "preventable_deaths" in advance_metrics

def test_admit_patient(simulator):
    patient = simulator.ed_queue[0]
    patient_id = patient.patient_id
    
    decision = PatientDecision(
        patient_id=patient_id,
        action="admit_general"
    )
    
    simulator.apply_decisions([decision])
    assert patient_id in simulator.admitted_patients
    assert simulator.wards[WardType.GENERAL].occupied_beds == 1
    assert patient_id not in [p.patient_id for p in simulator.ed_queue]

def test_discharge_patient(simulator):
    # First admit someone
    patient = simulator.ed_queue[0]
    pid = patient.patient_id
    simulator.apply_decisions([PatientDecision(patient_id=pid, action="admit_general")])
    
    # Now discharge them by acting on ANOTHER patient in the ED
    ed_patient = simulator.ed_queue[0]
    simulator.apply_decisions([PatientDecision(patient_id=ed_patient.patient_id, action="defer", discharge_patient_id=pid)])
    assert pid not in simulator.admitted_patients
    assert simulator.wards[WardType.GENERAL].occupied_beds == 0
    assert any(p.patient_id == pid for p in simulator.discharged_patients)
