import numpy as np

import cirq.devices as devices
import cirq.experiments.ghz.fidelity as ghz_fidelity
import cirq.experiments.ghz.ghz_1d as ghz_1d
import cirq.sim as sim


def test_measure_ghz_fidelity():
    qubits = devices.LineQubit.range(10)
    sampler = sim.Simulator()
    circuit = ghz_1d.generate_1d_ghz_circuit(qubits)
    rng = np.random.default_rng()
    result = ghz_fidelity.measure_ghz_fidelity(circuit, 20, 20, rng, sampler)
    f, df = result.compute_fidelity(mitigated=False)
    assert f == 1.0
    assert df == 0.0
