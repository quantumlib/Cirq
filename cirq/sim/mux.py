# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampling/simulation methods that delegate to appropriate simulators.

Filename is a reference to multiplexing.
"""

from typing import List, Optional, Type, Union, Sequence, cast, TYPE_CHECKING

import numpy as np

from cirq import circuits, protocols, study, devices, ops, value
from cirq._doc import document
from cirq.sim import (sparse_simulator, density_matrix_simulator,
                      state_vector_simulator)
from cirq.sim.clifford import clifford_simulator
from cirq._compat import deprecated

if TYPE_CHECKING:
    import cirq

CIRCUIT_LIKE = Union[circuits.Circuit, ops.Gate, ops.OP_TREE]
document(
    CIRCUIT_LIKE,  # type: ignore
    """A `circuits.Circuit` or a value that can be trivially converted into it:
        a gate, an operation, and a list or tree of operations.
    """)


def _is_clifford_circuit(program: 'cirq.Circuit') -> bool:
    return all(
        clifford_simulator.CliffordSimulator.is_supported_operation(op)
        for op in program.all_operations())


def sample(program: 'cirq.Circuit',
           *,
           noise: 'cirq.NOISE_MODEL_LIKE' = None,
           param_resolver: Optional[study.ParamResolver] = None,
           repetitions: int = 1,
           dtype: Type[np.number] = np.complex64,
           seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None) -> study.TrialResult:
    """Simulates sampling from the given circuit.

    Args:
        program: The circuit to sample from.
        noise: Noise model to use while running the simulation.
        param_resolver: Parameters to run with the program.
        repetitions: The number of samples to take.
        dtype: The `numpy.dtype` used by the simulation. Typically one of
            `numpy.complex64` or `numpy.complex128`.
            Favors speed over precision by default, i.e. uses `numpy.complex64`.
        seed: The random seed to use for this simulator.
    """
    noise_model = devices.NoiseModel.from_noise_model_like(noise)

    # State vector simulation is much faster, but only works if no randomness.
    if noise_model == devices.NO_NOISE:
        if _is_clifford_circuit(program):
            # If all non-measurement operations are clifford, use the Clifford
            # simulator.
            return clifford_simulator.CliffordSimulator(seed=seed).run(
                program, param_resolver=param_resolver, repetitions=repetitions)
        if protocols.has_unitary(program):
            return sparse_simulator.Simulator(dtype=dtype, seed=seed).run(
                program=program,
                param_resolver=param_resolver,
                repetitions=repetitions)

    return density_matrix_simulator.DensityMatrixSimulator(
        dtype=dtype, noise=noise_model,
        seed=seed).run(program=program,
                       param_resolver=param_resolver,
                       repetitions=repetitions)


def _to_circuit(program: 'cirq.CIRCUIT_LIKE') -> 'cirq.Circuit':
    result = None
    if isinstance(program, circuits.Circuit):
        # No change needed.
        result = program
    elif isinstance(program, ops.Gate):
        result = circuits.Circuit(
            program.on(*devices.LineQid.for_gate(program)))
    else:
        # It should be an OP_TREE.
        result = circuits.Circuit(program)
    return cast('cirq.Circuit', result)


def final_state_vector(
        program: 'cirq.CIRCUIT_LIKE',
        *,
        initial_state: Union[int, Sequence[Union[int, float, complex]], np.
                             ndarray] = 0,
        param_resolver: study.ParamResolverOrSimilarType = None,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        dtype: Type[np.number] = np.complex64,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None) -> 'np.ndarray':
    """Returns the state vector resulting from acting operations on a state.

    By default the input state is the computational basis zero state, in which
    case the output is just the first column of the implied unitary matrix.

    Args:
        program: The circuit, gate, operation, or tree of operations
            to apply to the initial state in order to produce the result.
        param_resolver: Parameters to run with the program.
        qubit_order: Determines the canonical ordering of the qubits. This
            is often used in specifying the initial state, i.e. the
            ordering of the computational basis states.
        initial_state: If an int, the state is set to the computational
            basis state corresponding to this state. Otherwise  if this
            is a np.ndarray it is the full initial state. In this case it
            must be the correct size, be normalized (an L2 norm of 1), and
            be safely castable to an appropriate dtype for the simulator.
        dtype: The `numpy.dtype` used by the simulation. Typically one of
            `numpy.complex64` or `numpy.complex128`.
        seed: The random seed to use for this simulator.

    Returns:
        The state vector resulting from applying the given unitary operations to
        the desired initial state. Specifically, a numpy array containing the
        the amplitudes in np.kron order, where the order of arguments to kron
        is determined by the qubit order argument (which defaults to just
        sorting the qubits that are present into an ascending order).
    """

    if not isinstance(initial_state, int):
        initial_state = np.asarray(initial_state, dtype=dtype)

    circuit_like = _to_circuit(program)

    if not protocols.has_unitary(
            protocols.resolve_parameters(circuit_like, param_resolver)):
        raise ValueError(
            "Program doesn't have a single well defined final state vector "
            "because it is not unitary. "
            "Maybe you wanted `cirq.final_density_matrix`?\n"
            "\n"
            "Program: {!r}".format(circuit_like))

    result = sparse_simulator.Simulator(dtype=dtype, seed=seed).simulate(
        program=circuit_like,
        initial_state=initial_state,
        qubit_order=qubit_order,
        param_resolver=param_resolver)

    return cast(sparse_simulator.SparseSimulatorStep, result).state_vector()


final_wavefunction = deprecated(
    deadline='v0.10.0',
    fix='Use `cirq.final_state_vector` instead.')(final_state_vector)


def sample_sweep(program: 'cirq.Circuit',
                 params: study.Sweepable,
                 *,
                 noise: 'cirq.NOISE_MODEL_LIKE' = None,
                 repetitions: int = 1,
                 dtype: Type[np.number] = np.complex64,
                 seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
                ) -> List[study.TrialResult]:
    """Runs the supplied Circuit, mimicking quantum hardware.

    In contrast to run, this allows for sweeping over different parameter
    values.

    Args:
        program: The circuit to simulate.
        params: Parameters to run with the program.
        noise: Noise model to use while running the simulation.
        repetitions: The number of repetitions to simulate, per set of
            parameter values.
        dtype: The `numpy.dtype` used by the simulation. Typically one of
            `numpy.complex64` or `numpy.complex128`.
            Favors speed over precision by default, i.e. uses `numpy.complex64`.
        seed: The random seed to use for this simulator.

    Returns:
        TrialResult list for this run; one for each possible parameter
        resolver.
    """
    prng = value.parse_random_state(seed)

    trial_results = []  # type: List[study.TrialResult]
    for param_resolver in study.to_resolvers(params):
        measurements = sample(program,
                              noise=noise,
                              param_resolver=param_resolver,
                              repetitions=repetitions,
                              dtype=dtype,
                              seed=prng)
        trial_results.append(measurements)
    return trial_results


def final_density_matrix(
        program: 'cirq.CIRCUIT_LIKE',
        *,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        initial_state: Union[int, Sequence[Union[int, float, complex]], np.
                             ndarray] = 0,
        param_resolver: study.ParamResolverOrSimilarType = None,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        dtype: Type[np.number] = np.complex64,
        seed: Optional[Union[int, np.random.RandomState]] = None,
        ignore_measurement_results: bool = True) -> 'np.ndarray':
    """Returns the density matrix resulting from simulating the circuit.

    Note that, unlike `cirq.final_state_vector`, terminal measurements
    are not omitted. Instead, all measurements are treated as sources
    of decoherence (i.e. measurements do not collapse, they dephase). See
    ignore_measurement_results for details.

    Args:
        program: The circuit, gate, operation, or tree of operations
            to apply to the initial state in order to produce the result.
        noise: Noise model to use while running the simulation.
        param_resolver: Parameters to run with the program.
        qubit_order: Determines the canonical ordering of the qubits. This
            is often used in specifying the initial state, i.e. the
            ordering of the computational basis states.
        initial_state: If an int, the state is set to the computational
            basis state corresponding to this state. Otherwise  if this
            is a np.ndarray it is the full initial state. In this case it
            must be the correct size, be normalized (an L2 norm of 1), and
            be safely castable to an appropriate dtype for the simulator.
        dtype: The `numpy.dtype` used by the simulation. Typically one of
            `numpy.complex64` or `numpy.complex128`.
        seed: The random seed to use for this simulator.
        ignore_measurement_results: Defaults to True. When True, the returned
            density matrix is not conditioned on any measurement results.
            For example, this effectively replaces computational basis
            measurement with dephasing noise. The result density matrix in this
            case should be unique. When False, the result will be conditioned on
            sampled (but unreported) measurement results. In this case the
            result may vary from call to call.

    Returns:
        The density matrix for the state which results from applying the given
        operations to the desired initial state.

    """
    initial_state_like = None
    if not isinstance(initial_state, int):
        initial_state_like = np.asarray(initial_state, dtype=dtype)
    else:
        initial_state_like = initial_state

    noise_model = devices.NoiseModel.from_noise_model_like(noise)
    circuit_like = _to_circuit(program)

    can_do_unitary_simulation = True
    if not protocols.has_unitary(circuit_like):
        can_do_unitary_simulation = False
    if isinstance(circuit_like, circuits.Circuit):
        if cast(circuits.Circuit, circuit_like).has_measurements():
            # Including terminal measurements.
            can_do_unitary_simulation = False
    if noise_model != devices.NO_NOISE:
        can_do_unitary_simulation = False

    if can_do_unitary_simulation:
        # pure case: use SparseSimulator
        result = sparse_simulator.Simulator(dtype=dtype, seed=seed).simulate(
            program=circuit_like,
            initial_state=initial_state_like,
            qubit_order=qubit_order,
            param_resolver=param_resolver)
        return cast(state_vector_simulator.StateVectorTrialResult,
                    result).density_matrix_of()
    else:
        # noisy case: use DensityMatrixSimulator with dephasing
        result = density_matrix_simulator.DensityMatrixSimulator(
            dtype=dtype,
            noise=noise,
            seed=seed,
            ignore_measurement_results=(ignore_measurement_results)).simulate(
                program=circuit_like,
                initial_state=initial_state_like,
                qubit_order=qubit_order,
                param_resolver=param_resolver)
        return cast(density_matrix_simulator.DensityMatrixTrialResult,
                    result).final_density_matrix
