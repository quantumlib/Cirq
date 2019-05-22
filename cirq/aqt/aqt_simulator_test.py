import pytest

from cirq.aqt import AQTSimulator


def test_simulator_no_circ():
    with pytest.raises(RuntimeError):
        sim = AQTSimulator(no_qubit=1)
        sim.simulate_samples(1)
