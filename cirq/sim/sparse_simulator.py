# Copyright 2018 The Cirq Developers
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

"""A simulator that uses numpy's einsum or sparse matrix operations."""

import collections

from typing import Dict, Iterator, List, Tuple, Type, TYPE_CHECKING

import numpy as np

from cirq import circuits, linalg, ops, protocols, study, value
from cirq.sim import simulator, wave_function, wave_function_simulator

if TYPE_CHECKING:
    import cirq


class _FlipGate(ops.SingleQubitGate):
    """A unitary gate that flips the |0> state with another state.

    Used by `Simulator` to reset a qubit.
    """

    def __init__(self, dimension: int, reset_value: int):
        assert 0 < reset_value < dimension
        self.dimension = dimension
        self.reset_value = reset_value

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (self.dimension,)

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        args.available_buffer[..., 0] = args.target_tensor[..., self.
                                                           reset_value]
        args.available_buffer[..., self.
                              reset_value] = args.target_tensor[..., 0]
        return args.available_buffer


# Mutable named tuple to hold state and a buffer.
class _StateAndBuffer():
    def __init__(self, state, buffer):
        self.state = state
        self.buffer = buffer


class Simulator(simulator.SimulatesSamples,
                wave_function_simulator.SimulatesIntermediateWaveFunction):
    """A sparse matrix wave function simulator that uses numpy.

    This simulator can be applied on circuits that are made up of operations
    that have a `_unitary_` method, or `_has_unitary_` and
    `_apply_unitary_`, `_mixture_` methods, are measurements, or support a
    `_decompose_` method that returns operations satisfying these same
    conditions. That is to say, the operations should follow the
    `cirq.SupportsConsistentApplyUnitary` protocol, the `cirq.SupportsUnitary`
    protocol, the `cirq.SupportsMixture` protocol, or the
    `cirq.CompositeOperation` protocol. It is also permitted for the circuit
    to contain measurements which are operations that support
    `cirq.SupportsChannel` and `cirq.SupportsMeasurementKey`

    This simulator supports three types of simulation.

    Run simulations which mimic running on actual quantum hardware. These
    simulations do not give access to the wave function (like actual hardware).
    There are two variations of run methods, one which takes in a single
    (optional) way to resolve parameterized circuits, and a second which
    takes in a list or sweep of parameter resolver:

        run(circuit, param_resolver, repetitions)

        run_sweep(circuit, params, repetitions)

    The simulation performs optimizations if the number of repetitions is
    greater than one and all measurements in the circuit are terminal (at the
    end of the circuit). These methods return `TrialResult`s which contain both
    the measurement results, but also the parameters used for the parameterized
    circuit operations. The initial state of a run is always the all 0s state
    in the computational basis.

    By contrast the simulate methods of the simulator give access to the
    wave function of the simulation at the end of the simulation of the circuit.
    These methods take in two parameters that the run methods do not: a
    qubit order and an initial state. The qubit order is necessary because an
    ordering must be chosen for the kronecker product (see
    `SparseSimulationTrialResult` for details of this ordering). The initial
    state can be either the full wave function, or an integer which represents
    the initial state of being in a computational basis state for the binary
    representation of that integer. Similar to run methods, there are two
    simulate methods that run for single runs or for sweeps across different
    parameters:

        simulate(circuit, param_resolver, qubit_order, initial_state)

        simulate_sweep(circuit, params, qubit_order, initial_state)

    The simulate methods in contrast to the run methods do not perform
    repetitions. The result of these simulations is a
    `SparseSimulationTrialResult` which contains, in addition to measurement
    results and information about the parameters that were used in the
    simulation,access to the state via the `state` method and `StateVectorMixin`
    methods.

    If one wishes to perform simulations that have access to the
    wave function as one steps through running the circuit there is a generator
    which can be iterated over and each step is an object that gives access
    to the wave function.  This stepping through a `Circuit` is done on a
    `Moment` by `Moment` manner.

        simulate_moment_steps(circuit, param_resolver, qubit_order,
                              initial_state)

    One can iterate over the moments via

        for step_result in simulate_moments(circuit):
           # do something with the wave function via step_result.state

    Note also that simulations can be stochastic, i.e. return different results
    for different runs.  The first version of this occurs for measurements,
    where the results of the measurement are recorded.  This can also
    occur when the circuit has mixtures of unitaries.

    See `Simulator` for the definitions of the supported methods.
    """

    def __init__(self,
                 *,
                 dtype: Type[np.number] = np.complex64,
                 seed: value.RANDOM_STATE_LIKE = None):
        """A sparse matrix simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation. One of
                `numpy.complex64` or `numpy.complex128`.
            seed: The random seed to use for this simulator.
        """
        if np.dtype(dtype).kind != 'c':
            raise ValueError(
                'dtype must be a complex type but was {}'.format(dtype))
        self._dtype = dtype
        self._prng = value.parse_random_state(seed)

    def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int) -> Dict[str, List[np.ndarray]]:
        """See definition in `cirq.SimulatesSamples`."""
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        self._check_all_resolved(resolved_circuit)

        def measure_or_mixture(op):
            return protocols.is_measurement(op) or protocols.has_mixture(op)
        if circuit.are_all_matches_terminal(measure_or_mixture):
            return self._run_sweep_sample(resolved_circuit, repetitions)
        return self._run_sweep_repeat(resolved_circuit, repetitions)

    def _run_sweep_sample(
        self,
        circuit: circuits.Circuit,
        repetitions: int) -> Dict[str, List[np.ndarray]]:
        for step_result in self._base_iterator(
                circuit=circuit,
                qubit_order=ops.QubitOrder.DEFAULT,
                initial_state=0,
                perform_measurements=False):
            pass
        # We can ignore the mixtures since this is a run method which
        # does not return the state.
        measurement_ops = [op for _, op, _ in
                           circuit.findall_operations_with_gate_type(
                                   ops.MeasurementGate)]
        return step_result.sample_measurement_ops(measurement_ops,
                                                  repetitions,
                                                  seed=self._prng)

    def _run_sweep_repeat(
        self,
        circuit: circuits.Circuit,
        repetitions: int) -> Dict[str, List[np.ndarray]]:
        measurements = {}  # type: Dict[str, List[np.ndarray]]
        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                    circuit,
                    qubit_order=ops.QubitOrder.DEFAULT,
                    initial_state=0)

            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    if not k in measurements:
                        measurements[k] = []
                    measurements[k].append(np.array(v, dtype=np.uint8))
        return {k: np.array(v) for k, v in measurements.items()}

    def _simulator_iterator(
            self,
            circuit: circuits.Circuit,
            param_resolver: study.ParamResolver,
            qubit_order: ops.QubitOrderOrList,
            initial_state: 'cirq.STATE_VECTOR_LIKE',
    ) -> Iterator:
        """See definition in `cirq.SimulatesIntermediateState`.

        If the initial state is an int, the state is set to the computational
        basis state corresponding to this state. Otherwise  if the initial
        state is a np.ndarray it is the full initial state. In this case it
        must be the correct size, be normalized (an L2 norm of 1), and
        be safely castable to an appropriate dtype for the simulator.
        """
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        self._check_all_resolved(resolved_circuit)
        actual_initial_state = 0 if initial_state is None else initial_state
        return self._base_iterator(resolved_circuit,
                                   qubit_order,
                                   actual_initial_state,
                                   perform_measurements=True)

    def _base_iterator(
            self,
            circuit: circuits.Circuit,
            qubit_order: ops.QubitOrderOrList,
            initial_state: 'cirq.STATE_VECTOR_LIKE',
            perform_measurements: bool = True,
    ) -> Iterator:
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
                circuit.all_qubits())
        num_qubits = len(qubits)
        qid_shape = protocols.qid_shape(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        state = wave_function.to_valid_state_vector(initial_state,
                                                    num_qubits,
                                                    qid_shape=qid_shape,
                                                    dtype=self._dtype)
        if len(circuit) == 0:
            yield SparseSimulatorStep(state, {}, qubit_map, self._dtype)

        def on_stuck(bad_op: ops.Operation):
            return TypeError(
                "Can't simulate unknown operations that don't specify a "
                "_unitary_ method, a _decompose_ method, "
                "(_has_unitary_ + _apply_unitary_) methods,"
                "(_has_mixture_ + _mixture_) methods, or are measurements."
                ": {!r}".format(bad_op))

        def keep(potential_op: ops.Operation) -> bool:
            # The order of this is optimized to call has_xxx methods first.
            return (protocols.has_unitary(potential_op) or
                    protocols.has_mixture(potential_op) or
                    protocols.is_measurement(potential_op) or
                    isinstance(potential_op.gate, ops.ResetChannel))

        data = _StateAndBuffer(state=np.reshape(state, qid_shape),
                               buffer=np.empty(qid_shape, dtype=self._dtype))
        for moment in circuit:
            measurements = collections.defaultdict(
                list)  # type: Dict[str, List[int]]

            unitary_ops_and_measurements = protocols.decompose(
                moment, keep=keep, on_stuck_raise=on_stuck)

            for op in unitary_ops_and_measurements:
                indices = [qubit_map[qubit] for qubit in op.qubits]
                if isinstance(op.gate, ops.ResetChannel):
                    self._simulate_reset(op, data, indices)
                elif protocols.has_unitary(op):
                    self._simulate_unitary(op, data, indices)
                elif protocols.is_measurement(op):
                    # Do measurements second, since there may be mixtures that
                    # operate as measurements.
                    # TODO: support measurement outside the computational basis.
                    if perform_measurements:
                        self._simulate_measurement(op, data, indices,
                                                   measurements, num_qubits)
                elif protocols.has_mixture(op):
                    self._simulate_mixture(op, data, indices)

            yield SparseSimulatorStep(
                state_vector=data.state,
                measurements=measurements,
                qubit_map=qubit_map,
                dtype=self._dtype)

    def _simulate_unitary(self, op: ops.Operation, data: _StateAndBuffer,
            indices: List[int]) -> None:
        """Simulate an op that has a unitary."""
        result = protocols.apply_unitary(
                op,
                args=protocols.ApplyUnitaryArgs(
                        data.state,
                        data.buffer,
                        indices))
        if result is data.buffer:
            data.buffer = data.state
        data.state = result

    def _simulate_reset(self, op: ops.Operation, data: _StateAndBuffer,
                        indices: List[int]) -> None:
        """Simulate an op that is a reset to the |0> state."""
        if isinstance(op.gate, ops.ResetChannel):
            reset = op.gate
            # Do a silent measurement.
            bits, _ = wave_function.measure_state_vector(
                data.state, indices, out=data.state, qid_shape=data.state.shape)
            # Apply bit flip(s) to change the reset the bits to 0.
            for b, i, d in zip(bits, indices, protocols.qid_shape(reset)):
                if b == 0:
                    continue  # Already zero, no reset needed
                reset_unitary = _FlipGate(d, reset_value=b)(*op.qubits)
                self._simulate_unitary(reset_unitary, data, [i])

    def _simulate_measurement(self, op: ops.Operation, data: _StateAndBuffer,
                              indices: List[int],
                              measurements: Dict[str, List[int]],
                              num_qubits: int) -> None:
        """Simulate an op that is a measurement in the computational basis."""
        # TODO: support measurement outside computational basis.
        if isinstance(op.gate, ops.MeasurementGate):
            meas = op.gate
            invert_mask = meas.full_invert_mask()
            # Measure updates inline.
            bits, _ = wave_function.measure_state_vector(
                data.state,
                indices,
                out=data.state,
                qid_shape=data.state.shape,
                seed=self._prng)
            corrected = [
                bit ^ (bit < 2 and mask)
                for bit, mask in zip(bits, invert_mask)
            ]
            key = protocols.measurement_key(meas)
            measurements[key].extend(corrected)

    def _simulate_mixture(self, op: ops.Operation, data: _StateAndBuffer,
            indices: List[int]) -> None:
        """Simulate an op that is a mixtures of unitaries."""
        probs, unitaries = zip(*protocols.mixture(op))
        # We work around numpy barfing on choosing from a list of
        # numpy arrays (which is not `one-dimensional`) by selecting
        # the index of the unitary.
        index = self._prng.choice(range(len(unitaries)), p=probs)
        shape = protocols.qid_shape(op) * 2
        unitary = unitaries[index].astype(self._dtype).reshape(shape)
        result = linalg.targeted_left_multiply(unitary, data.state, indices,
                                               out=data.buffer)
        data.buffer = data.state
        data.state = result

    def _check_all_resolved(self, circuit):
        """Raises if the circuit contains unresolved symbols."""
        if protocols.is_parameterized(circuit):
            unresolved = [
                op for moment in circuit for op in moment
                if protocols.is_parameterized(op)
            ]
            raise ValueError(
                'Circuit contains ops whose symbols were not specified in '
                'parameter sweep. Ops: {}'.format(unresolved))


class SparseSimulatorStep(wave_function.StateVectorMixin,
                          wave_function_simulator.WaveFunctionStepResult):
    """A `StepResult` that includes `StateVectorMixin` methods."""

    def __init__(self, state_vector, measurements, qubit_map, dtype):
        """Results of a step of the simulator.

        Args:
            qubit_map: A map from the Qubits in the Circuit to the the index
                of this qubit for a canonical ordering. This canonical ordering
                is used to define the state vector (see the state_vector()
                method).
            measurements: A dictionary from measurement gate key to measurement
                results, ordered by the qubits that the measurement operates on.
        """
        super().__init__(measurements=measurements, qubit_map=qubit_map)
        self._dtype = dtype
        size = np.prod(protocols.qid_shape(self), dtype=int)
        self._state_vector = np.reshape(state_vector, size)

    def _simulator_state(self
                        ) -> wave_function_simulator.WaveFunctionSimulatorState:
        return wave_function_simulator.WaveFunctionSimulatorState(
            qubit_map=self.qubit_map,
            state_vector=self._state_vector)

    def state_vector(self):
        """Return the wave function at this point in the computation.

        The state is returned in the computational basis with these basis
        states defined by the qubit_map. In particular the value in the
        qubit_map is the index of the qubit, and these are translated into
        binary vectors where the last qubit is the 1s bit of the index, the
        second-to-last is the 2s bit of the index, and so forth (i.e. big
        endian ordering).

        Example:
             qubit_map: {QubitA: 0, QubitB: 1, QubitC: 2}
             Then the returned vector will have indices mapped to qubit basis
             states like the following table

                |     | QubitA | QubitB | QubitC |
                | :-: | :----: | :----: | :----: |
                |  0  |   0    |   0    |   0    |
                |  1  |   0    |   0    |   1    |
                |  2  |   0    |   1    |   0    |
                |  3  |   0    |   1    |   1    |
                |  4  |   1    |   0    |   0    |
                |  5  |   1    |   0    |   1    |
                |  6  |   1    |   1    |   0    |
                |  7  |   1    |   1    |   1    |
        """
        return self._simulator_state().state_vector

    def set_state_vector(self, state: 'cirq.STATE_VECTOR_LIKE'):
        update_state = wave_function.to_valid_state_vector(
            state,
            len(self.qubit_map),
            qid_shape=protocols.qid_shape(self, None),
            dtype=self._dtype)
        np.copyto(self._state_vector, update_state)

    def sample(self,
               qubits: List[ops.Qid],
               repetitions: int = 1,
               seed: value.RANDOM_STATE_LIKE = None) -> np.ndarray:
        indices = [self.qubit_map[qubit] for qubit in qubits]
        return wave_function.sample_state_vector(self._state_vector,
                                                 indices,
                                                 qid_shape=protocols.qid_shape(
                                                     self, None),
                                                 repetitions=repetitions,
                                                 seed=seed)
