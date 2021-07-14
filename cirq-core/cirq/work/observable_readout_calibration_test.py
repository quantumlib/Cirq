import cirq
import cirq.work as cw
import cirq.contrib.noise_models as ccn
import numpy as np


def test_calibrate_readout_error():
    sampler = cirq.DensityMatrixSimulator(
        noise=ccn.DepolarizingWithDampedReadoutNoiseModel(
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
