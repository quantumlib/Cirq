from typing import Sequence

import cirq
import cirq.work as cw
import numpy as np


class DepolarizingWithDampedReadoutNoiseModel(cirq.NoiseModel):
    """This simulates asymmetric readout error.

    The noise is structured so the T1 decay is applied, then the readout bitflip, then measurement.
    If a circuit contains measurements, they must be in moments that don't also contain gates.
    """

    def __init__(self, depol_prob: float, bitflip_prob: float, decay_prob: float):
        self.qubit_noise_gate = cirq.DepolarizingChannel(depol_prob)
        self.readout_noise_gate = cirq.BitFlipChannel(bitflip_prob)
        self.readout_decay_gate = cirq.AmplitudeDampingChannel(decay_prob)

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']):
        if cirq.devices.noise_model.validate_all_measurements(moment):
            return [
                cirq.Moment(self.readout_decay_gate(q) for q in system_qubits),
                cirq.Moment(self.readout_noise_gate(q) for q in system_qubits),
                moment,
            ]
        else:
            return [moment, cirq.Moment(self.qubit_noise_gate(q) for q in system_qubits)]


def test_calibrate_readout_error():
    sampler = cirq.DensityMatrixSimulator(
        noise=DepolarizingWithDampedReadoutNoiseModel(
            depol_prob=1e-3,
            bitflip_prob=0.03,
            decay_prob=0.03,
        ),
        seed=10,
    )
    readout_calibration = cw.calibrate_readout_error(
        qubits=cirq.LineQubit.range(2),
        sampler=sampler,
        stopping_criteria=cw.RepetitionsStoppingCriteria(100_000),
    )
    means = readout_calibration.means()
    assert len(means) == 2, 'Two qubits'
    assert np.all(means > 0.89)
    assert np.all(means < 0.91)
