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

from typing import Dict, Iterator, List, Union

import numpy as np

from cirq import circuits, linalg, ops, protocols, study
from cirq.sim import simulator, wave_function, wave_function_simulator


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
    `cirq.SupportsApplyUnitary` protocol, the `cirq.SupportsUnitary` protocol,
    the `cirq.SupportsMixture` protocol, or the `cirq.CompositeOperation`
    protocol. It is also permitted for the circuit to contain measurements
    which are operations that support `cirq.SupportsChannel` and
    `cirq.SupportsMeasurementKey`

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

    Finally, one can compute the values of displays (instances of
    `SamplesDisplay` or `WaveFunctionDisplay`) in the circuit:

        compute_displays(circuit, param_resolver, qubit_order, initial_state)

        compute_displays_sweep(circuit, params, qubit_order, initial_state)

    The result of computing display values is stored in a
    `ComputeDisplaysResult`.

    See `Simulator` for the definitions of the supported methods.
    """

    def __init__(self, *, dtype=np.complex64):
        """A sparse matrix simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation. One of
            `numpy.complex64` or `numpy.complex128`
        """
        if np.dtype(dtype).kind != 'c':
            raise ValueError(
                'dtype must be a complex type but was {}'.format(dtype))
        self._dtype = dtype

    def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int) -> Dict[str, List[np.ndarray]]:
        """See definition in `cirq.SimulatesSamples`."""
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
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
        return step_result.sample_measurement_ops(measurement_ops, repetitions)

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
                    measurements[k].append(np.array(v, dtype=bool))
        return {k: np.array(v) for k, v in measurements.items()}

    def _simulator_iterator(
            self,
            circuit: circuits.Circuit,
            param_resolver: study.ParamResolver,
            qubit_order: ops.QubitOrderOrList,
            initial_state: Union[int, np.ndarray],
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
        actual_initial_state = 0 if initial_state is None else initial_state
        return self._base_iterator(resolved_circuit,
                                   qubit_order,
                                   actual_initial_state,
                                   perform_measurements=True)

    def _base_iterator(
            self,
            circuit: circuits.Circuit,
            qubit_order: ops.QubitOrderOrList,
            initial_state: Union[int, np.ndarray],
            perform_measurements: bool=True,
    ) -> Iterator:
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
                circuit.all_qubits())
        num_qubits = len(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        state = wave_function.to_valid_state_vector(initial_state,
                                                    num_qubits,
                                                    self._dtype)
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
            return (protocols.has_unitary(potential_op)
                    or protocols.has_mixture(potential_op)
                    or protocols.is_measurement(potential_op))

        data = _StateAndBuffer(
                state=np.reshape(state, (2,) * num_qubits),
                buffer=np.empty((2,) * num_qubits, dtype=self._dtype))
        for moment in circuit:
            measurements = collections.defaultdict(
                    list)  # type: Dict[str, List[bool]]

            non_display_ops = (op for op in moment
                               if not isinstance(op, (ops.SamplesDisplay,
                                                      ops.WaveFunctionDisplay,
                                                      ops.DensityMatrixDisplay
                                                      )))
            unitary_ops_and_measurements = protocols.decompose(
                non_display_ops,
                keep=keep,
                on_stuck_raise=on_stuck)

            for op in unitary_ops_and_measurements:
                indices = [qubit_map[qubit] for qubit in op.qubits]
                if protocols.has_unitary(op):
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

    def _simulate_measurement(self, op: ops.Operation, data: _StateAndBuffer,
            indices: List[int], measurements: Dict[str, List[bool]],
            num_qubits: int) -> None:
        """Simulate an op that is a measurement in the computataional basis."""
        meas = ops.op_gate_of_type(op, ops.MeasurementGate)
        # TODO: support measurement outside computational basis.
        if meas:
            invert_mask = meas.invert_mask or num_qubits * (False,)
            # Measure updates inline.
            bits, _ = wave_function.measure_state_vector(data.state,
                                                         indices,
                                                         data.state)
            corrected = [bit ^ mask for bit, mask in
                         zip(bits, invert_mask)]
            key = protocols.measurement_key(meas)
            measurements[key].extend(corrected)

    def _simulate_mixture(self, op: ops.Operation, data: _StateAndBuffer,
            indices: List[int]) -> None:
        """Simulate an op that is a mixtures of unitaries."""
        probs, unitaries = zip(*protocols.mixture(op))
        # We work around numpy barfing on choosing from a list of
        # numpy arrays (which is not `one-dimensional`) by selecting
        # the index of the unitary.
        index = np.random.choice(range(len(unitaries)), p=probs)
        shape = (2,) * (2 * len(indices))
        unitary = unitaries[index].astype(self._dtype).reshape(shape)
        result = linalg.targeted_left_multiply(unitary, data.state, indices,
                                               out=data.buffer)
        data.buffer = data.state
        data.state = result


class SparseSimulatorStep(wave_function.StateVectorMixin,
                          wave_function_simulator.WaveFunctionStepResult):
    """A `StepResult` that includes `StateVectorMixin` methods."""

    def __init__(self, state_vector, measurements, qubit_map, dtype):
        """Results of a step of the simulator.

        Attributes:
            qubit_map: A map from the Qubits in the Circuit to the the index
                of this qubit for a canonical ordering. This canonical ordering
                is used to define the state vector (see the state_vector()
                method).
            measurements: A dictionary from measurement gate key to measurement
                results, ordered by the qubits that the measurement operates on.
        """
        super().__init__(measurements=measurements, qubit_map=qubit_map)
        self._dtype = dtype
        self._state_vector = np.reshape(state_vector, 2 ** len(qubit_map))

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
        return self._simulator_state().state_vector

    def set_state_vector(self, state: Union[int, np.ndarray]):
        update_state = wave_function.to_valid_state_vector(state,
                                                           len(self.qubit_map),
                                                           self._dtype)
        np.copyto(self._state_vector, update_state)

    def sample(self, qubits: List[ops.Qid],
               repetitions: int = 1) -> np.ndarray:
        indices = [self.qubit_map[qubit] for qubit in qubits]
        return wave_function.sample_state_vector(self._state_vector, indices,
                                                 repetitions)
