from nose.tools import raises

from cirq.aqt import AQTSimulator


@raises(RuntimeError)
def test_simulator_no_circ():
    sim = AQTSimulator(no_qubit=1)
    sim.simulate_samples(1)
