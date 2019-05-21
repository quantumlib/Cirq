from cirq.aqt import AQTSampler, AQTSimulator
from nose.tools import raises

@raises(RuntimeError)
def test_simulator_no_circ():
    sim = AQTSimulator(no_qubit=1)
    sim.simulate_samples(1)
