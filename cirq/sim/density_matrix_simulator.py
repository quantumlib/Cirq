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

"""Simulator for density matrices that simulates noisy quantum circuits."""

import collections

from typing import cast, Dict, Iterator, List, Type, Union

import numpy as np

from cirq import circuits, linalg, ops, protocols, study, value
from cirq.sim import density_matrix_utils, simulator

class DensityMatrixSimulator(simulator.SimulatesSamples,
                             simulator.SimulatesIntermediateState):
    """A simulator for density matrices and noisy quantum circuits.

    This simulator can be applied on circuits that are made up of operations
    that have:
        * a `_channel_` method
        * a `_mixture_` method for a probabilistic combination of unitary gates.
        * a `_unitary_` method
        * a `_has_unitary_` and `_apply_unitary_` method.
        * measurements
        * a `_decompose_` that eventually yields one of the above
    That is, the circuit must have elements that follow on of the protocols:
        * `cirq.SupportsChannel`
        * `cirq.SupportsMixture`
        * `cirq.SupportsApplyUnitary`
        * `cirq.SupportsUnitary`
        * `cirq.SupportsDecompose`
    or is a measurement.

    This simulator supports three types of simulation.

    Run simulations which mimic running on actual quantum hardware. These
    simulations do not give access to the density matrix (like actual hardware).
    There are two variations of run methods, one which takes in a single
    (optional) way to resolve parameterized circuits, and a second which
    takes in a list or sweep of parameter resolver:

        run(circuit, param_resolver, repetitions)

        run_sweep(circuit, params, repetitions)

    These methods return `TrialResult`s which contain both the measurement
    results, but also the parameters used for the parameterized
    circuit operations. The initial state of a run is always the all 0s state
    in the computational basis.

    By contrast the simulate methods of the simulator give access to the density
    matrix of the simulation at the end of the simulation of the circuit.
    Note that if the circuit contains measurements then the density matrix
    is that result for those particular measurement results. For example
    if there is one measurement, then the simulation may result in the
    measurement result for this measurement, and the density matrix will
    be that conditional on that result. It will not be the density matrix formed
    by summing over the different measurements and their probabilities.
    The simulate methods take in two parameters that the run methods do not: a
    qubit order and an initial state. The qubit order is necessary because an
    ordering must be chosen for the kronecker product (see
    `DensityMatrixTrialResult` for details of this ordering). The initial
    state can be either the full density matrix, the full wave function (for
    pure states), or an integer which represents the initial state of being
    in a computational basis state for the binary representation of that
    integer. Similar to run methods, there are two simulate methods that run
    for single simulations or for sweeps across different parameters:

        simulate(circuit, param_resolver, qubit_order, initial_state)

        simulate_sweep(circuit, params, qubit_order, initial_state)

    The simulate methods in contrast to the run methods do not perform
    repetitions. The result of these simulations is a
    `DensityMatrixTrialResult` which contains, in addition to measurement
    results and information about the parameters that were used in the
    simulation, access to the density matrix via the `density_matrix` method.

    If one wishes to perform simulations that have access to the
    density matrix as one steps through running the circuit there is a generator
    which can be iterated over and each step is an object that gives access
    to the density matrix.  This stepping through a `Circuit` is done on a
    `Moment` by `Moment` manner.

        simulate_moment_steps(circuit, param_resolver, qubit_order,
                              initial_state)

    One can iterate over the moments via

        for step_result in simulate_moments(circuit):
           # do something with the density matrix via
           # step_result.density_matrix()
    """

    def __init__(self, dtype: Type[np.number] = np.complex64):
        """Density matrix simulator.

         Args:
            dtype: The `numpy.dtype` used by the simulation. One of
            `numpy.complex64` or `numpy.complex128`
        """
        if dtype not in {np.complex64, np.complex128}:
            raise ValueError(
                'dtype must be complex64 or complex128, was {}'.format(dtype))

        self._dtype = dtype

    def _run(self,
            circuit: circuits.Circuit,
            param_resolver: study.ParamResolver,
            repetitions: int) -> Dict[str, np.ndarray]:
        """See definition in `cirq.SimulatesSamples`."""
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        measurements = {}  # type: Dict[str, List[np.ndarray]]
        # TODO: optimize for all terminal measurements.
        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                resolved_circuit,
                qubit_order=ops.QubitOrder.DEFAULT,
                initial_state=0,
                perform_measurements=True)
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    if not k in measurements:
                        measurements[k] = []
                    measurements[k].append(np.array(v, dtype=bool))
        return {k: np.array(v) for k, v in measurements.items()}


    def _simulator_iterator(self,
            circuit: circuits.Circuit,
            param_resolver: study.ParamResolver,
            qubit_order: ops.QubitOrderOrList,
            initial_state: Union[int, np.ndarray]) -> Iterator:
        """See definition in `cirq.SimulatesIntermediateState`.

        If the initial state is an int, the state is set to the computational
        basis state corresponding to this state. Otherwise  if the initial
        state is a np.ndarray it is the full initial state, either a pure state
        or the full density matrix.  If it is the pure state it must be the
        correct size, be normalized (an L2 norm of 1), and be safely castable
        to an appropriate dtype for the simulator.  If it is a mixed state
        it must be correctly sized and positive semidefinite with trace one.
        """
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        actual_initial_state = 0 if initial_state is None else initial_state
        return self._base_iterator(resolved_circuit,
                                   qubit_order,
                                   actual_initial_state)

    def _base_iterator(
            self,
            circuit: circuits.Circuit,
            qubit_order: ops.QubitOrderOrList,
            initial_state: Union[int, np.ndarray],
            perform_measurements: bool = True) -> Iterator:
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            circuit.all_qubits())
        num_qubits = len(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        matrix = density_matrix_utils.to_valid_density_matrix(
            initial_state, num_qubits, self._dtype)
        if len(circuit) == 0:
            yield DensityMatrixStepResult(matrix, {}, qubit_map, self._dtype)
        matrix = np.reshape(matrix, (2,) * num_qubits * 2)

        def on_stuck(bad_op: ops.Operation):
            return TypeError(
                "Can't simulate operations that don't implement "
                "SupportsUnitary, SupportsApplyUnitary, SupportsMixture, "
                "SupportsChannel or is a measurement: {!r}".format(bad_op))

        def keep(potential_op: ops.Operation) -> bool:
            return (protocols.has_channel(potential_op)
                    or ops.MeasurementGate.is_measurement(potential_op)
                    or isinstance(potential_op,
                                  (ops.SamplesDisplay, ops.WaveFunctionDisplay))
                    )

        matrix = np.reshape(matrix, (2,) * num_qubits * 2)
        for moment in circuit:
            measurements = collections.defaultdict(
                list)  # type: Dict[str, List[bool]]

            channel_ops_and_measurements = protocols.decompose(
                moment.operations,
                keep=keep,
                on_stuck_raise=on_stuck)

            for op in channel_ops_and_measurements:
                indices = [qubit_map[qubit] for qubit in op.qubits]
                if isinstance(op,
                              (ops.SamplesDisplay, ops.WaveFunctionDisplay)):
                    continue
                elif ops.MeasurementGate.is_measurement(op):
                    meas = cast(ops.MeasurementGate,
                                cast(ops.GateOperation, op).gate)
                    if perform_measurements:
                        invert_mask = meas.invert_mask or num_qubits * (False,)
                        # Measure updates inline.
                        bits, _ = density_matrix_utils.measure_density_matrix(
                            matrix, indices, matrix)
                        corrected = [bit ^ mask for bit, mask in
                                     zip(bits, invert_mask)]
                        key = protocols.measurement_key(meas)
                        measurements[key].extend(corrected)
                else:
                    # TODO: Use apply_channel similar to apply_unitary.
                    gate = cast(ops.GateOperation, op).gate
                    channel = protocols.channel(gate)
                    sum_buffer = np.zeros((2,) * 2 * num_qubits,
                                          dtype=self._dtype)
                    for krauss in channel:
                        krauss_tensor = np.reshape(krauss.astype(self._dtype),
                                                   (2,) * gate.num_qubits() * 2)
                        buffer = linalg.targeted_left_multiply(krauss_tensor,
                                                               matrix,
                                                               indices)
                        # No need to transpose as we are acting on the tensor
                        # representation of matrix, so transpose is done for us.
                        buffer = linalg.targeted_left_multiply(
                            np.conjugate(krauss_tensor),
                            buffer,
                            [num_qubits + x for x in indices])
                        sum_buffer += buffer
                    np.copyto(dst=matrix, src=sum_buffer)
            yield DensityMatrixStepResult(
                    density_matrix=matrix,
                    measurements=measurements,
                    qubit_map=qubit_map,
                    dtype=self._dtype)

    def _create_simulator_trial_result(self,
            params: study.ParamResolver,
            measurements: Dict[str, np.ndarray],
            final_simulator_state: 'DensityMatrixSimulatorState') \
            -> 'DensityMatrixTrialResult':
        return DensityMatrixTrialResult(
            params=params,
            measurements=measurements,
            final_simulator_state=final_simulator_state)


class DensityMatrixStepResult(simulator.StepResult):
    """A single step in the simulation of the DensityMatrixSimulator.

    Attributes:
        qubit_map: A map from the Qubits in the Circuit to the the index
            of this qubit for a canonical ordering. This canonical ordering
            is used to define the state vector (see the state_vector()
            method).
        measurements: A dictionary from measurement gate key to measurement
            results, ordered by the qubits that the measurement operates on.
    """

    def __init__(self,
            density_matrix: np.ndarray,
            measurements: Dict[str, np.ndarray],
            qubit_map: Dict[ops.QubitId, int],
            dtype: Type[np.number] = np.complex64):
        """DensityMatrixStepResult.

        Args:
            density_matrix: The density matrix at this step. Can be mutated.
            measurements: The measurements for this step of the simulation.
            qubit_map: A map from qubit id to index used to define the
                ordering of the basis in density_matrix.
            dtype: The numpy dtype for the density matrix.
        """
        super().__init__(measurements)
        self._density_matrix = density_matrix
        self._qubit_map = qubit_map
        self._dtype = dtype

    def simulator_state(self) -> 'DensityMatrixSimulatorState':
        return DensityMatrixSimulatorState(self._density_matrix,
                                           self._qubit_map)

    def set_density_matrix(self, density_matrix_repr: Union[int, np.ndarray]):
        """Set the density matrix to a new density matrix.

        Args:
            density_matrix_repr: If this is an int, the density matrix is set to
            the computational basis state corresponding to this state. Otherwise
            if this is a np.ndarray it is the full state, either a pure state
            or the full density matrix.  If it is the pure state it must be the
            correct size, be normalized (an L2 norm of 1), and be safely
            castable to an appropriate dtype for the simulator.  If it is a
            mixed state it must be correctly sized and positive semidefinite
            with trace one.
        """
        density_matrix = density_matrix_utils.to_valid_density_matrix(
            density_matrix_repr, len(self._qubit_map), self._dtype)
        density_matrix = np.reshape(density_matrix,
                                    self.simulator_state().density_matrix.shape)
        np.copyto(dst=self.simulator_state().density_matrix, src=density_matrix)

    def density_matrix(self):
        """Returns the density matrix at this step in the simulation.

        The density matrix that is stored in this result is returned in the
        computational basis with these basis states defined by the qubit_map.
        In particular the value in the qubit_map is the index of the qubit,
        and these are translated into binary vectors where the last qubit is
        the 1s bit of the index, the second-to-last is the 2s bit of the index,
        and so forth (i.e. big endian ordering). The density matrix is a
        `2 ** num_qubits` square matrix, with rows and columns ordered by
        the computational basis as just described.

        Example:
             qubit_map: {QubitA: 0, QubitB: 1, QubitC: 2}
             Then the returned density matrix will have (row and column) indices
             mapped to qubit basis states like the following table

                    | QubitA | QubitB | QubitC
                :-: | :----: | :----: | :----:
                 0  |   0    |   0    |   0
                 1  |   0    |   0    |   1
                 2  |   0    |   1    |   0
                 3  |   0    |   1    |   1
                 4  |   1    |   0    |   0
                 5  |   1    |   0    |   1
                 6  |   1    |   1    |   0
                 7  |   1    |   1    |   1
        """
        size = 2 ** len(self._qubit_map)
        return np.reshape(self._density_matrix, (size, size))

    def sample(self,
            qubits: List[ops.QubitId],
            repetitions: int = 1) -> np.ndarray:
        indices = [self._qubit_map[q] for q in qubits]
        return density_matrix_utils.sample_density_matrix(
            self.simulator_state().density_matrix,
            indices, repetitions)


@value.value_equality(unhashable=True)
class DensityMatrixSimulatorState():
    """The simulator state for DensityMatrixSimulator

    Args:
        density_matrix: The density matrix of the simulation.
        qubit_map: A map from qubit id to index used to define the
            ordering of the basis in density_matrix.
    """

    def __init__(self,
            density_matrix: np.ndarray,
            qubit_map: Dict[ops.QubitId, int]):
        self.density_matrix = density_matrix
        self.qubit_map = qubit_map

    def _value_equality_values_(self):
        return (self.density_matrix.tolist(), self.qubit_map)

    def __repr__(self):
        return ("cirq.DensityMatrixSimulatorState("
                "density_matrix=np.array({!r}), "
                "qubit_map={!r})".format(self.density_matrix.tolist(),
                                         self.qubit_map))


@value.value_equality(unhashable=True)
class DensityMatrixTrialResult(simulator.SimulationTrialResult):
    """A `SimulationTrialResult` for `DensityMatrixSimulator` runs.

    The density matrix that is stored in this result is returned in the
    computational basis with these basis states defined by the qubit_map.
    In particular the value in the qubit_map is the index of the qubit,
    and these are translated into binary vectors where the last qubit is
    the 1s bit of the index, the second-to-last is the 2s bit of the index,
    and so forth (i.e. big endian ordering). The density matrix is a
    `2 ** num_qubits` square matrix, with rows and columns ordered by
    the computational basis as just described.

    Example:
         qubit_map: {QubitA: 0, QubitB: 1, QubitC: 2}
         Then the returned density matrix will have (row and column) indices
         mapped to qubit basis states like the following table

                | QubitA | QubitB | QubitC
            :-: | :----: | :----: | :----:
             0  |   0    |   0    |   0
             1  |   0    |   0    |   1
             2  |   0    |   1    |   0
             3  |   0    |   1    |   1
             4  |   1    |   0    |   0
             5  |   1    |   0    |   1
             6  |   1    |   1    |   0
             7  |   1    |   1    |   1

    Attributes:
        params: A ParamResolver of settings used for this result.
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are a numpy ndarray of actual boolean
            measurement results (ordered by the qubits acted on by the
            measurement gate.)
        final_simulator_state: The final simulator state of the system after the
            trial finishes.
        final_density_matrix: The final density matrix of the system.
    """

    def __init__(self,
            params: study.ParamResolver,
            measurements: Dict[str, np.ndarray],
            final_simulator_state: DensityMatrixSimulatorState) -> None:
        super().__init__(params=params,
                         measurements=measurements,
                         final_simulator_state=final_simulator_state)
        size = 2 ** len(final_simulator_state.qubit_map)
        self.final_density_matrix = np.reshape(
            final_simulator_state.density_matrix, (size, size))

    def _value_equality_values_(self):
        measurements = {k: v.tolist() for k, v in
                        sorted(self.measurements.items())}
        return (self.params, measurements, self.final_simulator_state)

    def __repr__(self):
        return ("cirq.DensityMatrixTrialResult(params={!r}, measurements={!r}, "
                "final_simulator_state={!r})"
                .format(self.params, self.measurements,
                        self.final_simulator_state))
