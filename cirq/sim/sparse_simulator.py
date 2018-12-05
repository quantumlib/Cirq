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

from typing import cast, Dict, Iterator, List, Union

import numpy as np

from cirq import circuits, study, ops, protocols
from cirq.sim import simulator, wave_function


class Simulator(simulator.SimulatesSamples,
                simulator.SimulatesIntermediateWaveFunction):
    """A sparse matrix wave function simulator that uses numpy.

    This simulator can be applied on circuits that are made up of operations
    that have a `_unitary_` method, or `_has_unitary_` and
    `_apply_unitary_` methods, or else a `_decompose_` method that
    returns operations satisfying these same conditions. That is to say,
    the operations should follow the `cirq.SupportsApplyUnitary`
    protocol, the `cirq.SupportsUnitary` protocol, or the
    `cirq.CompositeOperation` protocol. (It is also permitted for the circuit
    to contain measurements.)

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
    `SimulationTrialResult` for details of this ordering). The initial state
    can be either the full wave function, or an integer which represents
    the initial state of being in a computational basis state for the binary
    representation of that integer. Similar to run methods, there are two
    simulate methods that run for single runs or for sweeps across different
    parameters:

        simulate(circuit, param_resolver, qubit_order, initial_state)

        simulate_sweep(circuit, params, qubit_order, initial_state)

    The simulate methods in contrast to the run methods do not perform
    repetitions. The result of these simulations is a `SimulationTrialResult`
    which contains in addition to measurement results and information about
    the parameters that were used in the simulation access to the state
    viat the `final_state` method.

    Finally if one wishes to perform simulations that have access to the
    wave function as one steps through running the circuit there is a generator
    which can be iterated over and each step is an object that gives access
    to the wave function.  This stepping through a `Circuit` is done on a
    `Moment` by `Moment` manner.

        simulate_moment_steps(circuit, param_resolver, qubit_order,
                              initial_state)

    One can iterate over the moments via

        for step_result in simulate_moments(circuit):
           # do something with the wave function via step_result.state


    See `Simulator` for the definitions of the supported methods.
    """

    def __init__(self, dtype=np.complex64):
        """A sparse matrix simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation. One of
            `numpy.complex64` or `numpy.complex128`
        """
        if dtype not in {np.complex64, np.complex128}:
            raise ValueError(
                'dtype must be complex64 or complex128 but was {}'.format(
                    dtype))
        self._dtype = dtype

    def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int) -> Dict[str, List[np.ndarray]]:
        """See definition in `sim.SimulatesSamples`."""
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        if circuit.are_all_measurements_terminal():
            return self._run_sweep_sample(resolved_circuit, repetitions)
        else:
            return self._run_sweep_repeat(resolved_circuit, repetitions)

    def _run_sweep_sample(
        self,
        circuit: circuits.Circuit,
        repetitions: int) -> Dict[str, List[np.ndarray]]:
        step_result = None
        for step_result in self._base_iterator(
                circuit=circuit,
                qubit_order=ops.QubitOrder.DEFAULT,
                initial_state=0,
                perform_measurements=False):
            pass
        if step_result is None:
            return {}
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
            perform_measurements: bool = True,
    ) -> Iterator[simulator.StepResult]:
        """See definition in `sim.SimulatesIntermediateWaveFunction`."""
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        return self._base_iterator(resolved_circuit, qubit_order, initial_state,
                                   perform_measurements)

    def _base_iterator(
            self,
            circuit: circuits.Circuit,
            qubit_order: ops.QubitOrderOrList,
            initial_state: Union[int, np.ndarray],
            perform_measurements: bool=True,
    ) -> Iterator[simulator.StepResult]:
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
                circuit.all_qubits())
        num_qubits = len(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        state = wave_function.to_valid_state_vector(initial_state,
                                                    num_qubits,
                                                    self._dtype)

        def on_stuck(bad_op: ops.Operation):
            return TypeError(
                "Can't simulate unknown operations that don't specify a "
                "_unitary_ method, a _decompose_ method, or "
                "(_has_unitary_ + _apply_unitary_) methods"
                ": {!r}".format(bad_op))

        def keep(potential_op: ops.Operation) -> bool:
            return (protocols.has_unitary(potential_op) or
                    ops.MeasurementGate.is_measurement(potential_op))

        state = np.reshape(state, (2,) * num_qubits)
        buffer = np.empty((2,) * num_qubits, dtype=self._dtype)
        for moment in circuit:
            measurements = collections.defaultdict(
                    list)  # type: Dict[str, List[bool]]

            unitary_ops_and_measurements = protocols.decompose(
                moment.operations,
                keep=keep,
                on_stuck_raise=on_stuck)

            for op in unitary_ops_and_measurements:
                indices = [qubit_map[qubit] for qubit in op.qubits]
                if ops.MeasurementGate.is_measurement(op):
                    gate = cast(ops.MeasurementGate,
                                cast(ops.GateOperation, op).gate)
                    if perform_measurements:
                        invert_mask = gate.invert_mask or num_qubits * (False,)
                        # Measure updates inline.
                        bits, _ = wave_function.measure_state_vector(state,
                                                                     indices,
                                                                     state)
                        corrected = [bit ^ mask for bit, mask in
                                     zip(bits, invert_mask)]
                        measurements[cast(str, gate.key)].extend(corrected)
                else:
                    result = protocols.apply_unitary(
                        op,
                        args=protocols.ApplyUnitaryArgs(
                            state,
                            buffer,
                            indices))
                    if result is buffer:
                        buffer = state
                    state = result
            yield SimulatorStep(state, measurements, qubit_map, self._dtype)


class SimulatorStep(simulator.StepResult):

    def __init__(self, state, measurements, qubit_map, dtype):
        """Results of a step of the simulator.

        Attributes:
            qubit_map: A map from the Qubits in the Circuit to the the index
                of this qubit for a canonical ordering. This canonical ordering
                is used to define the state (see the state() method).
            measurements: A dictionary from measurement gate key to measurement
                results, ordered by the qubits that the measurement operates on.
        """
        super().__init__(qubit_map, measurements)
        self._dtype = dtype
        self._state = np.reshape(state, 2 ** len(qubit_map))

    def state(self) -> np.ndarray:
        return self._state

    def set_state(self, state: Union[int, np.ndarray]):
        update_state = wave_function.to_valid_state_vector(state,
                                                           len(self.qubit_map),
                                                           self._dtype)
        np.copyto(self._state, update_state)

    def sample(self, qubits: List[ops.QubitId],
               repetitions: int = 1) -> np.ndarray:
        indices = [self.qubit_map[qubit] for qubit in qubits]
        return wave_function.sample_state_vector(self._state, indices,
                                                 repetitions)
