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

from cirq import circuits, protocols, study, schedules, devices, ops
from cirq.sim import (sparse_simulator, density_matrix_simulator,
                      wave_function_simulator)

if TYPE_CHECKING:
    import cirq


def sample(program: Union[circuits.Circuit, schedules.Schedule],
           *,
           noise: 'cirq.NOISE_MODEL_LIKE' = None,
           param_resolver: Optional[study.ParamResolver] = None,
           repetitions: int = 1,
           dtype: Type[np.number] = np.complex64,
           seed: Optional[Union[int, np.random.RandomState]] = None
          ) -> study.TrialResult:
    """Simulates sampling from the given circuit or schedule.

    Args:
        program: The circuit or schedule to sample from.
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
    if noise_model == devices.NO_NOISE and protocols.has_unitary(program):
        return sparse_simulator.Simulator(dtype=dtype, seed=seed).run(
            program=program,
            param_resolver=param_resolver,
            repetitions=repetitions)

    return density_matrix_simulator.DensityMatrixSimulator(
        dtype=dtype, noise=noise_model,
        seed=seed).run(program=program,
                       param_resolver=param_resolver,
                       repetitions=repetitions)


def _to_circuit_like(program: Union[circuits.Circuit, ops.Gate, ops.
                                    OP_TREE, schedules.Schedule]
                    ) -> 'Union[circuits.Circuit, schedules.Schedule]':
    result = None
    if isinstance(program, (schedules.Schedule, circuits.Circuit)):
        # No change needed.
        result = program
    elif isinstance(program, ops.Gate):
        result = circuits.Circuit(
            program.on(*devices.LineQid.for_gate(program)))
    else:
        # It should be an OP_TREE.
        result = circuits.Circuit(program)
    return cast(Union[circuits.Circuit, schedules.Schedule], result)


def final_wavefunction(
        program: Union[circuits.Circuit, ops.Gate, ops.OP_TREE, schedules.
                       Schedule],
        *,
        initial_state: Union[int, Sequence[Union[int, float, complex]], np.
                             ndarray] = 0,
        param_resolver: study.ParamResolverOrSimilarType = None,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        dtype: Type[np.number] = np.complex64,
        seed: Optional[Union[int, np.random.RandomState]] = None
) -> 'np.ndarray':
    """Returns the state vector resulting from acting operations on a state.

    By default the input state is the computational basis zero state, in which
    case the output is just the first column of the implied unitary matrix.

    Args:
        program: The circuit, schedule, gate, operation, or tree of operations
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
        The wavefunction resulting from applying the given unitary operations to
        the desired initial state. Specifically, a numpy array containing the
        the amplitudes in np.kron order, where the order of arguments to kron
        is determined by the qubit order argument (which defaults to just
        sorting the qubits that are present into an ascending order).
    """

    if not isinstance(initial_state, int):
        initial_state = np.asarray(initial_state, dtype=dtype)

    if isinstance(program, (schedules.Schedule, circuits.Circuit)):
        # No change needed.
        pass
    elif isinstance(program, ops.Gate):
        program = circuits.Circuit(
            program.on(*devices.LineQid.for_gate(program)))
    else:
        # It should be an OP_TREE.
        program = circuits.Circuit(program)

    if not protocols.has_unitary(
            protocols.resolve_parameters(program, param_resolver)):
        raise ValueError(
            "Program doesn't have a single well defined final wavefunction "
            "because it is not unitary. "
            "Maybe you wanted `cirq.sample_wavefunction`?\n"
            "\n"
            "Program: {!r}".format(program))

    result = sparse_simulator.Simulator(dtype=dtype, seed=seed).simulate(
        program=cast(Union[circuits.Circuit, schedules.Schedule], program),
        initial_state=initial_state,
        qubit_order=qubit_order,
        param_resolver=param_resolver)

    return cast(sparse_simulator.SparseSimulatorStep, result).state_vector()


def sample_sweep(program: Union[circuits.Circuit, schedules.Schedule],
                 params: study.Sweepable,
                 *,
                 noise: 'cirq.NOISE_MODEL_LIKE' = None,
                 repetitions: int = 1,
                 dtype: Type[np.number] = np.complex64,
                 seed: Optional[Union[int, np.random.RandomState]] = None
                ) -> List[study.TrialResult]:
    """Runs the supplied Circuit or Schedule, mimicking quantum hardware.

    In contrast to run, this allows for sweeping over different parameter
    values.

    Args:
        program: The circuit or schedule to simulate.
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
    if seed is None:
        prng = None
    elif isinstance(seed, np.random.RandomState):
        prng = seed
    else:
        prng = np.random.RandomState(seed)

    circuit = (program.to_circuit()
               if isinstance(program, schedules.Schedule) else program)

    trial_results = []  # type: List[study.TrialResult]
    for param_resolver in study.to_resolvers(params):
        measurements = sample(circuit,
                              noise=noise,
                              param_resolver=param_resolver,
                              repetitions=repetitions,
                              dtype=dtype,
                              seed=prng)
        trial_results.append(measurements)
    return trial_results


def final_density_matrix(
        program: Union[circuits.Circuit, ops.Gate, ops.OP_TREE, schedules.
                       Schedule],
        *,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        initial_state: Union[int, Sequence[Union[int, float, complex]], np.
                             ndarray] = 0,
        param_resolver: study.ParamResolverOrSimilarType = None,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        dtype: Type[np.number] = np.complex64,
        seed: Optional[Union[int, np.random.RandomState]] = None
) -> 'np.ndarray':
    """Returns the density matrix resulting from simulating the circuit.
    The terminal measurements would affect the returned result.

    Args:
        program: The circuit, schedule, gate, operation, or tree of operations
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
    circuit_like = _to_circuit_like(program)

    if not noise_model == devices.NO_NOISE and not protocols.has_unitary(
            circuit_like):
        raise ValueError("Noise specified more than once.")
    elif noise_model == devices.NO_NOISE and protocols.has_unitary(
            circuit_like):
        # pure case: use SparseSimulator
        result = sparse_simulator.Simulator(dtype=dtype, seed=seed).simulate(
            program=circuit_like,
            initial_state=initial_state_like,
            qubit_order=qubit_order,
            param_resolver=param_resolver)
        return cast(wave_function_simulator.WaveFunctionTrialResult,
                    result).density_matrix_of()
    else:
        # noisy case: use DensityMatrixSimulator
        result = density_matrix_simulator.DensityMatrixSimulator(
            dtype=dtype, noise=noise,
            seed=seed).simulate(program=circuit_like,
                                initial_state=initial_state_like,
                                qubit_order=qubit_order,
                                param_resolver=param_resolver)
        return cast(density_matrix_simulator.DensityMatrixTrialResult,
                    result).final_density_matrix
