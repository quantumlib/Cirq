from typing import Sequence, List
import itertools
import numpy as np

import cirq.contrib.noise_models as ccn
import cirq
from cirq.work.observable_measurement import group_settings_greedy, \
    measure_grouped_settings, VarianceStoppingCriteria, flatten_grouped_results, \
    ObservableMeasuredResult
import cirq.google as cg
import os

import scipy.linalg

TRY_TO_USE_QUANTUM_ENGINE = True


def get_sampler():
    if TRY_TO_USE_QUANTUM_ENGINE and 'GOOGLE_CLOUD_PROJECT' in os.environ:
        # coverage: ignore
        print("Using quantum engine")
        return cg.get_engine_sampler('rainbow', 'sqrt_iswap')

    print("Using noisy simulator")
    return cirq.DensityMatrixSimulator(noise=ccn.DepolarizingWithDampedReadoutNoiseModel(
        depol_prob=0.005, bitflip_prob=0.03, decay_prob=0.08))


def process_tomo_settings(qubits: Sequence[cirq.Qid]):
    """Iterate over [+-XYZ] x [XYZ] initializations and measurements.

    This creates a tomographically (over-)complete set of measurements for
    quantum process tomography.
    """
    for states in itertools.product(cirq.PAULI_STATES, repeat=len(qubits)):
        init_state = cirq.ProductState({q: st for q, st in zip(qubits, states)})

        for paulis in itertools.product([None, cirq.X, cirq.Y, cirq.Z], repeat=len(qubits)):
            observable = cirq.PauliString(
                {q: op for q, op in zip(qubits, paulis) if op is not None})

            yield cirq.InitObsSetting(
                init_state=init_state,
                observable=observable,
            )


def vec(matrix: np.ndarray) -> np.ndarray:
    """Column stack"""
    return matrix.T.reshape((-1, 1))


def unvec(vector: np.ndarray) -> np.ndarray:
    """Opposite of `vec`."""
    assert vector.ndim == 1, vector.shape
    dim = np.sqrt(len(vector))
    assert int(dim) == dim, dim
    dim = int(dim)
    matrix = vector.reshape(dim, dim)
    return matrix


def linear_inv_process_estimate(results: List[ObservableMeasuredResult],
                                qubits: List[cirq.Qid]) -> np.ndarray:
    """
    Estimate a quantum process using linear inversion.

    This is the simplest process tomography post processing and will
    likely give bad results for real data. Returns the choi matrix
    representation of the process.
    """
    measurement_matrix = np.asarray([
        vec(np.kron(
            result.init_state.projector(qubit_order=qubits),
            result.observable.dense(qubits=qubits)._unitary_().T,
        )).conj().T.squeeze()
        for result in results
    ])
    expectations = np.array([result.mean for result in results])
    rho = scipy.linalg.pinv(measurement_matrix) @ expectations
    return unvec(rho)


def get_circuit():
    """A process to tomographize"""
    qubits = [cirq.GridQubit(6, 5)]
    circuit = cirq.Circuit(cirq.YPowGate(exponent=0.5).on(qubits[0]))

    # # Uncomment to try random unitaries:
    # special_unitary = cirq.testing.random_special_unitary(2, random_state=52)
    # circuit = cirq.Circuit(cirq.MatrixGate(special_unitary).on(qubits[0]))
    # circuit = cg.optimized_for_sycamore(circuit)
    return circuit, qubits


def simulate():
    """Validate the inversion routine by simulating exact observable values."""
    circuit, qubits = get_circuit()
    settings = list(process_tomo_settings(qubits))

    print("Circuit:")
    print(circuit)

    results = []
    for setting in settings:
        psi = cirq.final_wavefunction(circuit, initial_state=setting.init_state)
        o = setting.observable.expectation_from_wavefunction(
            psi, qubit_map={qubits[0]: 0}, check_preconditions=False)
        assert np.isclose(o.imag, 0, atol=1e-5)
        o = o.real
        results.append(ObservableMeasuredResult(
            setting=setting,
            mean=o,
            variance=0,
            repetitions=float('inf'),
            circuit_params=dict(),
        ))

    print('\nFlattened Results:')
    for result in results:
        print(f'{str(result.setting):23s}{result.mean:+1.4f}')

    print("\nInverting results...")
    chi_est = linear_inv_process_estimate(results, qubits)
    chi_est = np.real_if_close(chi_est)
    print(np.round(chi_est, 3))

    print('-' * 30)
    u_true = vec(cirq.unitary(circuit))
    chi_true = u_true @ u_true.conj().T
    chi_true = np.real_if_close(chi_true)
    print(np.round(chi_true, 3))

    print("Frobenius norm:", np.linalg.norm(chi_true - chi_est))
    np.testing.assert_allclose(chi_est, chi_true, atol=1e-5)


def collect_data():
    circuit, qubits = get_circuit()
    settings = list(process_tomo_settings(qubits))

    print("Circuit:")
    print(circuit)
    print("\nSettings:")
    for setting in settings:
        print(setting)
    print(f'N = {len(settings)}')

    grouped_settings = group_settings_greedy(settings)
    print("\nGrouped Settings:")
    for max_setting, simul_settings in grouped_settings.items():
        print(str(max_setting) + ':', '[' + ', '.join(str(s) for s in simul_settings) + ']')
    print(f'N = {len(grouped_settings)}')

    grouped_results = measure_grouped_settings(
        circuit=circuit,
        grouped_settings=grouped_settings,
        sampler=get_sampler(),
        stopping_criteria=VarianceStoppingCriteria(1e-5),
        checkpoint=True,
        checkpoint_fn='./tomo.json'
    )

    print('\nGrouped Results:')
    for result in grouped_results:
        print(str(result.max_setting))
        print('-' * 30)
        for setting in result.simul_settings:
            print(
                f'  {str(setting):23s}{result.mean(setting):+1.4f} +- {result.stddev(setting):1.4f}')
        print()

    print('\nFlattened Results:')
    results = flatten_grouped_results(grouped_results)
    for result in results:
        print(f'{str(result.setting):23s}{result.mean:+1.4f} +- {result.stddev:1.4}')

    print("\nInverting results...")
    chi_est = linear_inv_process_estimate(results, qubits)
    chi_est = np.real_if_close(chi_est)
    print(np.round(chi_est, 3))

    print('-' * 30)
    u_true = vec(cirq.unitary(circuit))
    chi_true = u_true @ u_true.conj().T
    chi_true = np.real_if_close(chi_true)
    print(np.round(chi_true, 3))

    print("Frobenius norm:", np.linalg.norm(chi_true - chi_est))


def recover():
    """If data collection dies, the use of checkpoint files allows
    you to recover your work. You can also use the final checkpoint file
    as a rough-and-ready final copy of the data for further analysis.
    """
    grouped_results = cirq.read_json('./tomo.json')
    qubits = grouped_results[0].qubit_to_index.keys()
    results = flatten_grouped_results(grouped_results)

    chi_est = linear_inv_process_estimate(results, qubits)
    chi_est = np.real_if_close(chi_est)
    print(np.round(chi_est, 3))


def main():
    simulate()
    collect_data()
    recover()


if __name__ == '__main__':
    main()
