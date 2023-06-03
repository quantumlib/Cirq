# pylint: disable=wrong-or-nonexistent-copyright-notice
import dataclasses
from typing import Union, Iterable, TYPE_CHECKING

from cirq import circuits, study, ops
from cirq.work.observable_measurement import measure_grouped_settings, StoppingCriteria
from cirq.work.observable_settings import InitObsSetting, zeros_state

if TYPE_CHECKING:
    import cirq


def calibrate_readout_error(
    qubits: Iterable[ops.Qid],
    sampler: Union['cirq.Simulator', 'cirq.Sampler'],
    stopping_criteria: StoppingCriteria,
):
    # We know there won't be any fancy sweeps or observables so we can
    # get away with more repetitions per job
    stopping_criteria = dataclasses.replace(
        stopping_criteria, repetitions_per_chunk=100_000  # type: ignore[type-var]
    )

    # Simultaneous readout characterization:
    # We can measure all qubits simultaneously (i.e. _max_setting is ZZZ..ZZ
    # for all qubits). We will extract individual qubit quantities, so there
    # are `n_qubits` InitObsSetting, each responsible for one <Z>.
    #
    # Readout symmetrization means we just need to measure the "identity"
    # circuit. In reality, this corresponds to measuring I for half the time
    # and X for the other half.
    init_state = zeros_state(qubits)
    max_setting = InitObsSetting(
        init_state=init_state, observable=ops.PauliString({q: ops.Z for q in qubits})
    )
    grouped_settings = {
        max_setting: [
            InitObsSetting(init_state=init_state, observable=ops.PauliString({q: ops.Z}))
            for q in qubits
        ]
    }

    results = measure_grouped_settings(
        circuit=circuits.Circuit(),
        grouped_settings=grouped_settings,
        sampler=sampler,
        stopping_criteria=stopping_criteria,
        circuit_sweep=study.UnitSweep,
        readout_symmetrization=True,
    )
    (result,) = list(results)
    return result
