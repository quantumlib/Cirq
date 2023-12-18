import cirq
from typing import Sequence
import numpy as np
from cirq.experiments.qubit_characterizations import (
    RandomizedBenchMarkResult,
    _random_single_q_clifford,
    _single_qubit_cliffords,
    _gate_seq_to_mats,
)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import datetime


def _create_rb_circuit(
    qubits: tuple[cirq.GridQubit], num_cfds: int, c1: list, cfd_mats: np.array
) -> cirq.Circuit:
    circuits_to_zip = [_random_single_q_clifford(qubit, num_cfds, c1, cfd_mats) for qubit in qubits]
    circuit = cirq.Circuit.zip(*circuits_to_zip)
    measure_moment = cirq.Moment(
        cirq.measure_each(*qubits, key_func=lambda q: 'q{}_{}'.format(q.row, q.col))
    )
    circuit_with_meas = cirq.Circuit.from_moments(*(circuit.moments + [measure_moment]))
    return circuit_with_meas


def single_qubit_randomized_benchmarking(
    sampler: cirq.Sampler,
    use_xy_basis: bool = True,
    *,
    qubits: tuple[cirq.GridQubit] | None = None,
    num_clifford_range: Sequence[int] = [5, 18, 70, 265, 1000],
    num_circuits: int = 10,
    repetitions: int = 600,
) -> list[RandomizedBenchMarkResult]:
    if qubits is None:
        device = sampler.processor.get_device()
        qubits = tuple(sorted(list(device.metadata.qubit_set)))

    cliffords = _single_qubit_cliffords()
    c1 = cliffords.c1_in_xy if use_xy_basis else cliffords.c1_in_xz
    cfd_mats = np.array([_gate_seq_to_mats(gates) for gates in c1])

    # create circuits
    circuits = []
    for num_cfds in num_clifford_range:
        for _ in range(num_circuits):
            circuits.append(_create_rb_circuit(qubits, num_cfds, c1, cfd_mats))

    # run circuits
    results_all = sampler.run_batch(circuits, repetitions=repetitions)
    gnd_probs = {q: [] for q in qubits}
    idx = 0
    for num_cfds in num_clifford_range:
        excited_probs_l = {q: [] for q in qubits}
        for _ in range(num_circuits):
            results = results_all[idx][0]
            for qubit in qubits:
                excited_probs_l[qubit].append(
                    np.mean(results.measurements['q{}_{}'.format(qubit.row, qubit.col)])
                )
            idx += 1
        for qubit in qubits:
            gnd_probs[qubit].append(1.0 - np.mean(excited_probs_l[qubit]))
    return {q: RandomizedBenchMarkResult(num_clifford_range, gnd_probs[q]) for q in qubits}


def compute_pauli_errors(rb_result: dict) -> dict:
    exp_fit = lambda x, A, B, p: A * p**x + B
    rb_errors = {}
    d = 2
    for qubit in rb_result:
        data = np.array(result[qubit].data)
        x = data[:, 0]
        y = data[:, 1]
        fit = curve_fit(exp_fit, x, y)
        p = fit[0][2]
        pauli_error = (1.0 - 1.0 / (d * d)) * (1.0 - p)
        rb_errors[qubit] = pauli_error
    return rb_errors


def plot_error_rates(rb_result: dict, ax: plt.Axes | None = None) -> plt.Axes:
    errors = compute_pauli_errors(rb_result)
    heatmap = cirq.Heatmap(errors)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    _ = heatmap.plot(ax, vmin=0, vmax=0.01)
    return ax


if __name__ == "__main__":
    # The Google Cloud Project id to use.
    project_id = "i-dont-have-experimental"  # @param {type:"string"}

    from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook

    # For real engine instances, delete 'virtual=True' below.
    qcs_objects = get_qcs_objects_for_notebook(project_id)

    project_id = qcs_objects.project_id
    engine = qcs_objects.engine
    if not qcs_objects.signed_in:
        print(
            "ERROR: Please setup project_id in this cell or set the `GOOGLE_CLOUD_PROJECT` env var to your project id."
        )
        print("Using noisy simulator instead.")

    processor_id = "bodega_sim_gmon18"  # @param {type:"string"}
    processor = engine.get_processor(processor_id)

    sampler = processor.get_sampler(
        # run_name =
        # device_config_name =
    )
    start = datetime.datetime.now().timestamp()
    result = single_qubit_randomized_benchmarking(sampler)
    end = datetime.datetime.now().timestamp()
    print(end - start)

    plot_error_rates(result)
