from src.simulation import AirportSimulation



def test_simulation_starts():
    sim = AirportSimulation(seed=1)
    assert len(sim.aircraft) > 0


def test_emergency_trigger():
    sim = AirportSimulation(seed=1)
    sim.trigger_emergency("Engine Failure")
    df = sim.aircraft_frame()
    assert "Emergency" in set(df["status"])


def test_step_advances_time():
    sim = AirportSimulation(seed=1)
    sim.step()
    assert sim.time_step == 1
