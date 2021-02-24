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

"""A simulator that uses numpy's einsum for sparse matrix operations."""

import collections
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Type,
    TYPE_CHECKING,
    DefaultDict,
    Union,
    cast,
)

import numpy as np

from cirq import circuits, ops, protocols, qis, study, value
from cirq.sim import (
    simulator,
    state_vector,
    state_vector_simulator,
    act_on_state_vector_args,
)
from cirq.sim.simulator import check_all_resolved, split_into_matching_protocol_then_general

if TYPE_CHECKING:
    import cirq


class Simulator(
    simulator.SimulatesSamples,
    state_vector_simulator.SimulatesIntermediateStateVector['SparseSimulatorStep'],
    simulator.SimulatesExpectationValues,
):
    """A sparse matrix state vector simulator that uses numpy.

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

    This simulator supports four types of simulation.

    Run simulations which mimic running on actual quantum hardware. These
    simulations do not give access to the state vector (like actual hardware).
    There are two variations of run methods, one which takes in a single
    (optional) way to resolve parameterized circuits, and a second which
    takes in a list or sweep of parameter resolver:

        run(circuit, param_resolver, repetitions)

        run_sweep(circuit, params, repetitions)

    The simulation performs optimizations if the number of repetitions is
    greater than one and all measurements in the circuit are terminal (at the
    end of the circuit). These methods return `Result`s which contain both
    the measurement results, but also the parameters used for the parameterized
    circuit operations. The initial state of a run is always the all 0s state
    in the computational basis.

    By contrast the simulate methods of the simulator give access to the
    state vector of the simulation at the end of the simulation of the circuit.
    These methods take in two parameters that the run methods do not: a
    qubit order and an initial state. The qubit order is necessary because an
    ordering must be chosen for the kronecker product (see
    `SparseSimulationTrialResult` for details of this ordering). The initial
    state can be either the full state vector, or an integer which represents
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
    state vector as one steps through running the circuit there is a generator
    which can be iterated over and each step is an object that gives access
    to the state vector.  This stepping through a `Circuit` is done on a
    `Moment` by `Moment` manner.

        simulate_moment_steps(circuit, param_resolver, qubit_order,
                              initial_state)

    One can iterate over the moments via

        for step_result in simulate_moments(circuit):
           # do something with the state vector via step_result.state_vector

    Note also that simulations can be stochastic, i.e. return different results
    for different runs.  The first version of this occurs for measurements,
    where the results of the measurement are recorded.  This can also
    occur when the circuit has mixtures of unitaries.

    If only the expectation values for some observables on the final state are
    required, there are methods for that as well. These methods take a mapping
    of names to observables, and return a map (or list of maps) of those names
    to the corresponding expectation values.

        simulate_expectation_values(circuit, observables, param_resolver,
                                    qubit_order, initial_state,
                                    permit_terminal_measurements)

        simulate_expectation_values_sweep(circuit, observables, params,
                                          qubit_order, initial_state,
                                          permit_terminal_measurements)

    Expectation values generated by these methods are exact (up to precision of
    the floating-point type used); the closest analogy on hardware requires
    estimating the expectation values from several samples.

    See `Simulator` for the definitions of the supported methods.
    """

    def __init__(
        self,
        *,
        dtype: Type[np.number] = np.complex64,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ):
        """A sparse matrix simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation. One of
                `numpy.complex64` or `numpy.complex128`.
            seed: The random seed to use for this simulator.
        """
        if np.dtype(dtype).kind != 'c':
            raise ValueError('dtype must be a complex type but was {}'.format(dtype))
        self._dtype = dtype
        self._prng = value.parse_random_state(seed)

    def _run(
        self, circuit: circuits.Circuit, param_resolver: study.ParamResolver, repetitions: int
    ) -> Dict[str, np.ndarray]:
        """See definition in `cirq.SimulatesSamples`."""
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        check_all_resolved(resolved_circuit)
        qubit_order = sorted(resolved_circuit.all_qubits())

        # Simulate as many unitary operations as possible before having to
        # repeat work for each sample.
        unitary_prefix, general_suffix = split_into_matching_protocol_then_general(
            resolved_circuit, protocols.has_unitary
        )
        step_result = None
        for step_result in self._base_iterator(
            circuit=unitary_prefix,
            qubit_order=qubit_order,
            initial_state=0,
            perform_measurements=False,
        ):
            pass
        assert step_result is not None

        # When an otherwise unitary circuit ends with non-demolition computation
        # basis measurements, we can sample the results more efficiently.
        general_ops = list(general_suffix.all_operations())
        if all(isinstance(op.gate, ops.MeasurementGate) for op in general_ops):
            return step_result.sample_measurement_ops(
                measurement_ops=cast(List[ops.GateOperation], general_ops),
                repetitions=repetitions,
                seed=self._prng,
            )

        qid_shape = protocols.qid_shape(qubit_order)
        intermediate_state = step_result.state_vector().reshape(qid_shape)
        return self._brute_force_samples(
            initial_state=intermediate_state,
            circuit=general_suffix,
            repetitions=repetitions,
            qubit_order=qubit_order,
        )

    def _brute_force_samples(
        self,
        initial_state: np.ndarray,
        circuit: circuits.Circuit,
        qubit_order: 'cirq.QubitOrderOrList',
        repetitions: int,
    ) -> Dict[str, np.ndarray]:
        """Repeatedly simulate a circuit in order to produce samples."""
        if repetitions == 0:
            return {key: np.empty(shape=[0, 1]) for key in protocols.measurement_keys(circuit)}

        measurements: DefaultDict[str, List[np.ndarray]] = collections.defaultdict(list)
        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                circuit, initial_state=initial_state, qubit_order=qubit_order
            )

            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    measurements[k].append(np.array(v, dtype=np.uint8))
        return {k: np.array(v) for k, v in measurements.items()}

    def _base_iterator(
        self,
        circuit: circuits.Circuit,
        qubit_order: ops.QubitOrderOrList,
        initial_state: 'cirq.STATE_VECTOR_LIKE',
        perform_measurements: bool = True,
    ) -> Iterator['SparseSimulatorStep']:
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
        num_qubits = len(qubits)
        qid_shape = protocols.qid_shape(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        state = qis.to_valid_state_vector(
            initial_state, num_qubits, qid_shape=qid_shape, dtype=self._dtype
        )
        if len(circuit) == 0:
            yield SparseSimulatorStep(state, {}, qubit_map, self._dtype)

        sim_state = act_on_state_vector_args.ActOnStateVectorArgs(
            target_tensor=np.reshape(state, qid_shape),
            available_buffer=np.empty(qid_shape, dtype=self._dtype),
            axes=[],
            prng=self._prng,
            log_of_measurement_results={},
        )

        for moment in circuit:
            for op in moment:
                if perform_measurements or not isinstance(op.gate, ops.MeasurementGate):
                    sim_state.axes = tuple(qubit_map[qubit] for qubit in op.qubits)
                    protocols.act_on(op, sim_state)

            yield SparseSimulatorStep(
                state_vector=sim_state.target_tensor,
                measurements=dict(sim_state.log_of_measurement_results),
                qubit_map=qubit_map,
                dtype=self._dtype,
            )
            sim_state.log_of_measurement_results.clear()

    def simulate_expectation_values_sweep(
        self,
        program: 'cirq.Circuit',
        observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']],
        params: 'study.Sweepable',
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
        permit_terminal_measurements: bool = False,
    ) -> List[List[float]]:
        if not permit_terminal_measurements and program.are_any_measurements_terminal():
            raise ValueError(
                'Provided circuit has terminal measurements, which may '
                'skew expectation values. If this is intentional, set '
                'permit_terminal_measurements=True.'
            )
        swept_evs = []
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        qmap = {q: i for i, q in enumerate(qubit_order.order_for(program.all_qubits()))}
        if not isinstance(observables, List):
            observables = [observables]
        pslist = [ops.PauliSum.wrap(pslike) for pslike in observables]
        for param_resolver in study.to_resolvers(params):
            result = self.simulate(
                program, param_resolver, qubit_order=qubit_order, initial_state=initial_state
            )
            swept_evs.append(
                [
                    obs.expectation_from_state_vector(result.final_state_vector, qmap)
                    for obs in pslist
                ]
            )
        return swept_evs


class SparseSimulatorStep(
    state_vector.StateVectorMixin, state_vector_simulator.StateVectorStepResult
):
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

    def _simulator_state(self) -> state_vector_simulator.StateVectorSimulatorState:
        return state_vector_simulator.StateVectorSimulatorState(
            qubit_map=self.qubit_map, state_vector=self._state_vector
        )

    def state_vector(self, copy: bool = True):
        """Return the state vector at this point in the computation.

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

        Args:
            copy: If True, then the returned state is a copy of the state
                vector. If False, then the state vector is not copied,
                potentially saving memory. If one only needs to read derived
                parameters from the state vector and store then using False
                can speed up simulation by eliminating a memory copy.
        """
        vector = self._simulator_state().state_vector
        return vector.copy() if copy else vector

    def set_state_vector(self, state: 'cirq.STATE_VECTOR_LIKE'):
        update_state = qis.to_valid_state_vector(
            state, len(self.qubit_map), qid_shape=protocols.qid_shape(self, None), dtype=self._dtype
        )
        np.copyto(self._state_vector, update_state)

    def sample(
        self,
        qubits: List[ops.Qid],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        indices = [self.qubit_map[qubit] for qubit in qubits]
        return state_vector.sample_state_vector(
            self._state_vector,
            indices,
            qid_shape=protocols.qid_shape(self, None),
            repetitions=repetitions,
            seed=seed,
        )
