import pytest

from cirq.aqt import AQTSimulator


def test_simulator_no_circ():
    with pytest.raises(RuntimeError):
        sim = AQTSimulator(num_qubits=1)
        sim.simulate_samples(1)
