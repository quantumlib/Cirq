# Copyright 2021 The Cirq Developers
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

"""Batteries-included class for Cirq's built-in simulators."""

from __future__ import annotations

import abc
import collections
from collections.abc import Iterator, Sequence
from typing import Any, cast, Generic, TYPE_CHECKING, TypeVar

import numpy as np

from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
    SimulatesIntermediateState,
    SimulatesSamples,
    SimulationTrialResult,
    split_into_matching_protocol_then_general,
    StepResult,
    TSimulationTrialResult,
)

if TYPE_CHECKING:
    import cirq

TStepResultBase = TypeVar('TStepResultBase', bound='StepResultBase')


class SimulatorBase(
    Generic[TStepResultBase, TSimulationTrialResult, TSimulationState],
    SimulatesIntermediateState[
        TStepResultBase, TSimulationTrialResult, SimulationStateBase[TSimulationState]
    ],
    SimulatesSamples,
    metaclass=abc.ABCMeta,
):
    """A base class for the built-in simulators.

    Most implementers of this interface should implement the
    `_create_partial_simulation_state` and `_create_step_result` methods. The
    first one creates the simulator's quantum state representation at the
    beginning of the simulation. The second creates the step result emitted
    after each `Moment` in the simulation.

    Iteration in the subclass is handled by the `_core_iterator` implementation
    here, which handles moment stepping, application of operations, measurement
    collection, and creation of noise. Simulators with more advanced needs can
    override the implementation if necessary.

    Sampling is handled by the implementation of `_run`. This implementation
    iterates the circuit to create a final step result, and samples that
    result when possible. If not possible, due to noise or classical
    probabilities on a state vector, the implementation attempts to fully
    iterate the unitary prefix once, then only repeat the non-unitary
    suffix from copies of the state obtained by the prefix. If more advanced
    functionality is required, then the `_run` method can be overridden.

    Note that state here refers to simulator state, which is not necessarily
    a state vector. The included simulators and corresponding states are state
    vector, density matrix, Clifford, and MPS. Each of these use the default
    `_core_iterator` and `_run` methods.
    """

    def __init__(
        self,
        *,
        dtype: type[np.complexfloating] = np.complex64,
        noise: cirq.NOISE_MODEL_LIKE = None,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        split_untangled_states: bool = False,
    ):
        """Initializes the simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation.
            noise: A noise model to apply while simulating.
            seed: The random seed to use for this simulator.
            split_untangled_states: If True, optimizes simulation by running
                unentangled qubit sets independently and merging those states
                at the end.
        """
        self._dtype = dtype
        self._prng = value.parse_random_state(seed)
        self._noise = devices.NoiseModel.from_noise_model_like(noise)
        self._split_untangled_states = split_untangled_states

    @property
    def noise(self) -> cirq.NoiseModel:
        return self._noise

    @abc.abstractmethod
    def _create_partial_simulation_state(
        self,
        initial_state: Any,
        qubits: Sequence[cirq.Qid],
        classical_data: cirq.ClassicalDataStore,
        prng: np.random.Generator | None = None,
    ) -> TSimulationState:
        """Creates an instance of the TSimulationState class for the simulator.

        It represents the supplied qubits initialized to the provided state.

        Args:
            initial_state: The initial state to represent. An integer state is
                understood to be a pure state. Other state representations are
                simulator-dependent.
            qubits: The sequence of qubits to represent.
            classical_data: The shared classical data container for this
                simulation.
        """

    @abc.abstractmethod
    def _create_step_result(
        self, sim_state: SimulationStateBase[TSimulationState]
    ) -> TStepResultBase:
        """This method should be implemented to create a step result.

        Args:
            sim_state: The SimulationStateBase for this trial.

        Returns:
            The StepResult.
        """

    def _can_be_in_run_prefix(self, val: Any):
        """Determines what should be put in the prefix in `_run`

        The `_run` method has an optimization that reduces repetition by
        splitting the circuit into a prefix that is pure with respect to the
        state representation, and only executing that once per sample set. For
        state vectors, any unitary operation is pure, and we make this the
        default here. For density matrices, any non-measurement operation can
        be represented wholly in the matrix, and thus this method is
        overridden there to enable greater optimization there.

        Custom simulators can override this method appropriately.

        Args:
            val: An operation or noise model to test for purity within the
                state representation.

        Returns:
            A boolean representing whether the value can be added to the
            `_run` prefix."""
        return protocols.has_unitary(val)

    def _base_iterator(
        self,
        circuit: cirq.AbstractCircuit,
        qubits: tuple[cirq.Qid, ...],
        initial_state: Any,
        param_resolver: cirq.ParamResolver | None = None,
        prng: np.random.Generator | None = None,
    ) -> Iterator[TStepResultBase]:
        if prng is None:
            sim_state = self._create_simulation_state(
                initial_state, qubits, param_resolver=param_resolver
            )
        else:
            sim_state = self._create_simulation_state(
                initial_state, qubits, param_resolver=param_resolver, prng=prng
            )
        return self._core_iterator(circuit, sim_state)

    def _core_iterator(
        self,
        circuit: cirq.AbstractCircuit,
        sim_state: SimulationStateBase[TSimulationState],
        all_measurements_are_terminal: bool = False,
    ) -> Iterator[TStepResultBase]:
        """Standard iterator over StepResult from Moments of a Circuit.

        Args:
            circuit: The circuit to simulate.
            sim_state: The initial args for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.
            all_measurements_are_terminal: Whether all measurements in the
                given circuit are terminal.

        Yields:
            StepResults from simulating a Moment of the Circuit.

        Raises:
            TypeError: The simulator encounters an op it does not support.
        """

        if len(circuit) == 0:
            yield self._create_step_result(sim_state)
            return

        system_qubits = sorted(circuit.all_qubits())
        measured: dict[tuple[cirq.Qid, ...], bool] = collections.defaultdict(bool)
        for moment in circuit:
            resolved_moment = protocols.resolve_parameters(moment, sim_state.param_resolver)
            noisy_moment = self.noise.noisy_moment(resolved_moment, system_qubits)
            for op in ops.flatten_to_ops(noisy_moment):
                # Preprocess measurements
                if all_measurements_are_terminal and measured[op.qubits]:
                    continue
                if isinstance(op.gate, ops.MeasurementGate):
                    measured[op.qubits] = True
                    if all_measurements_are_terminal:
                        continue

                # Resolve parameters on the fly
                raw_op = getattr(op, 'untagged', op)
                if protocols.is_parameterized(op) and not isinstance(raw_op, ops.SetVariable):
                    raise ValueError(
                        'Circuit contains ops whose symbols were not specified in the '
                        f'parameter sweep. Ops: [{op!r}]'
                    )

                # Simulate the operation
                try:
                    protocols.act_on(op, sim_state)
                except TypeError:
                    raise TypeError(f"{self.__class__.__name__} doesn't support {op!r}")

            yield self._create_step_result(sim_state)

    def _run(
        self,
        circuit: cirq.AbstractCircuit,
        param_resolver: cirq.ParamResolver,
        repetitions: int,
        prng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """See definition in `cirq.SimulatesSamples`."""
        param_resolver = study.ParamResolver({}) if param_resolver is None else param_resolver
        qubits = tuple(sorted(circuit.all_qubits()))
        if prng is None:
            sim_state = self._create_simulation_state(0, qubits, param_resolver=param_resolver)
        else:
            sim_state = self._create_simulation_state(
                0, qubits, param_resolver=param_resolver, prng=prng
            )

        def can_run_prefix(op: cirq.Operation) -> bool:
            resolved_op = protocols.resolve_parameters(op, param_resolver)
            return self._can_be_in_run_prefix(resolved_op)

        prefix, general_suffix = (
            split_into_matching_protocol_then_general(circuit, can_run_prefix)
            if self._can_be_in_run_prefix(self.noise)
            else (circuit[0:0], circuit)
        )
        step_result: TStepResultBase | None = None
        for step_result in self._core_iterator(circuit=prefix, sim_state=sim_state):
            pass
        assert step_result is not None

        general_ops = [
            protocols.resolve_parameters(op, sim_state.param_resolver)
            for op in general_suffix.all_operations()
        ]
        if all(isinstance(op.gate, ops.MeasurementGate) for op in general_ops):
            for step_result in self._core_iterator(
                circuit=general_suffix, sim_state=sim_state, all_measurements_are_terminal=True
            ):
                pass
            assert step_result is not None
            measurement_ops = [cast(ops.GateOperation, op) for op in general_ops]
            return step_result.sample_measurement_ops(
                measurement_ops,
                repetitions,
                seed=prng if prng is not None else self._prng,
                _allow_repeated=True,
            )

        records: dict[cirq.MeasurementKey, list[Sequence[Sequence[int]]]] = {}
        for i in range(repetitions):
            for step_result in self._core_iterator(
                general_suffix,
                sim_state=(
                    sim_state.copy(deep_copy_buffers=False) if i < repetitions - 1 else sim_state
                ),
            ):
                pass
            for k, r in step_result._classical_data.records.items():
                if k not in records:
                    records[k] = []
                records[k].append(r)
            for k, cr in step_result._classical_data.channel_records.items():
                if k not in records:
                    records[k] = []
                records[k].append([cr])

        def pad_evenly(results: Sequence[Sequence[Sequence[int]]]):
            largest = max(len(result) for result in results)
            xs = np.zeros((len(results), largest, len(results[0][0])), dtype=np.uint8)
            for i, result in enumerate(results):
                xs[i, 0 : len(result), :] = result
            return xs

        return {str(k): pad_evenly(v) for k, v in records.items()}

    def simulate_sweep_iter(
        self,
        program: cirq.AbstractCircuit,
        params: cirq.Sweepable,
        qubit_order: cirq.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
        prng: np.random.Generator | None = None,
    ) -> Iterator[TSimulationTrialResult]:
        """Simulates the supplied Circuit.

        This particular implementation overrides the base implementation such
        that an unparameterized prefix circuit is simulated and fed into the
        parameterized suffix circuit.

        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. This can be
                either a raw state or an `SimulationStateBase`. The form of the
                raw state depends on the simulation implementation. See
                documentation of the implementing class for details.
            prng: An `np.random.Generator` to draw from for this call
                instead of the internal random state.

        Returns:
            List of SimulationTrialResults for this run, one for each
            possible parameter resolver.
        """

        def sweep_prefixable(op: cirq.Operation):
            return self._can_be_in_run_prefix(op) and not protocols.is_parameterized(op)

        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(program.all_qubits())
        initial_state = 0 if initial_state is None else initial_state
        if prng is None:
            sim_state = self._create_simulation_state(initial_state, qubits)
        else:
            sim_state = self._create_simulation_state(initial_state, qubits, prng=prng)
        prefix, suffix = (
            split_into_matching_protocol_then_general(program, sweep_prefixable)
            if self._can_be_in_run_prefix(self.noise)
            else (program[0:0], program)
        )
        step_result: TStepResultBase | None = None
        for step_result in self._core_iterator(circuit=prefix, sim_state=sim_state):
            pass
        assert step_result is not None
        sim_state = step_result._sim_state
        yield from super().simulate_sweep_iter(suffix, params, qubit_order, sim_state, prng)

    def _create_simulation_state(
        self,
        initial_state: Any,
        qubits: Sequence[cirq.Qid],
        param_resolver: cirq.ParamResolver | None = None,
        prng: np.random.Generator | None = None,
    ) -> SimulationStateBase[TSimulationState]:
        if isinstance(initial_state, SimulationStateBase):
            if param_resolver is not None:
                initial_state.param_resolver = param_resolver
            return initial_state

        classical_data = value.ClassicalDataDictionaryStore()
        if self._split_untangled_states:
            args_map: dict[cirq.Qid | None, TSimulationState] = {}
            if isinstance(initial_state, int):
                for q in reversed(qubits):
                    if prng is None:
                        args_map[q] = self._create_partial_simulation_state(
                            initial_state=initial_state % q.dimension,
                            qubits=[q],
                            classical_data=classical_data,
                        )
                    else:
                        args_map[q] = self._create_partial_simulation_state(
                            initial_state=initial_state % q.dimension,
                            qubits=[q],
                            classical_data=classical_data,
                            prng=prng,
                        )
                    initial_state = int(initial_state / q.dimension)
            else:
                if prng is None:
                    args = self._create_partial_simulation_state(
                        initial_state=initial_state, qubits=qubits, classical_data=classical_data
                    )
                else:
                    args = self._create_partial_simulation_state(
                        initial_state=initial_state,
                        qubits=qubits,
                        classical_data=classical_data,
                        prng=prng,
                    )
                for q in qubits:
                    args_map[q] = args
            if prng is None:
                args_map[None] = self._create_partial_simulation_state(0, (), classical_data)
            else:
                args_map[None] = self._create_partial_simulation_state(0, (), classical_data, prng)
            return SimulationProductState(
                args_map,
                qubits,
                self._split_untangled_states,
                classical_data=classical_data,
                param_resolver=param_resolver,
            )
        else:
            if prng is None:
                state = self._create_partial_simulation_state(
                    initial_state=initial_state, qubits=qubits, classical_data=classical_data
                )
            else:
                state = self._create_partial_simulation_state(
                    initial_state=initial_state,
                    qubits=qubits,
                    classical_data=classical_data,
                    prng=prng,
                )
            state.param_resolver = (
                study.ParamResolver({}) if param_resolver is None else param_resolver
            )
            return state


class StepResultBase(
    Generic[TSimulationState], StepResult[SimulationStateBase[TSimulationState]], abc.ABC
):
    """A base class for step results."""

    def __init__(self, sim_state: SimulationStateBase[TSimulationState]):
        """Initializes the step result.

        Args:
            sim_state: The `SimulationStateBase` for this step.
        """
        super().__init__(sim_state)
        self._merged_sim_state_cache: TSimulationState | None = None
        qubits = sim_state.qubits
        self._qubits = qubits
        self._qubit_mapping = {q: i for i, q in enumerate(qubits)}
        self._qubit_shape = tuple(q.dimension for q in qubits)
        self._classical_data = sim_state.classical_data

    def _qid_shape_(self):
        return self._qubit_shape

    @property
    def _merged_sim_state(self) -> TSimulationState:
        if self._merged_sim_state_cache is None:
            self._merged_sim_state_cache = self._sim_state.create_merged_state()
        return self._merged_sim_state_cache

    def sample(
        self,
        qubits: list[cirq.Qid],
        repetitions: int = 1,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> np.ndarray:
        return self._sim_state.sample(qubits, repetitions, seed)


class SimulationTrialResultBase(
    SimulationTrialResult[SimulationStateBase[TSimulationState]], Generic[TSimulationState], abc.ABC
):
    """A base class for trial results."""

    def __init__(
        self,
        params: study.ParamResolver,
        measurements: dict[str, np.ndarray],
        final_simulator_state: cirq.SimulationStateBase[TSimulationState],
    ) -> None:
        """Initializes the `SimulationTrialResultBase` class.

        Args:
            params: A ParamResolver of settings used for this result.
            measurements: A dictionary from measurement gate key to measurement
                results. Measurement results are a numpy ndarray of actual
                boolean measurement results (ordered by the qubits acted on by
                the measurement gate.)
            final_simulator_state: The final simulator state of the system after the
                trial finishes.
        """
        super().__init__(params, measurements, final_simulator_state=final_simulator_state)
        self._merged_sim_state_cache: TSimulationState | None = None

    def get_state_containing_qubit(self, qubit: cirq.Qid) -> TSimulationState:
        """Returns the independent state space containing the qubit.

        Args:
            qubit: The qubit whose state space is required.

        Returns:
            The state space containing the qubit."""
        return self._final_simulator_state[qubit]

    def _get_substates(self) -> Sequence[TSimulationState]:
        state = self._final_simulator_state
        if isinstance(state, SimulationProductState):
            substates: dict[TSimulationState, int] = {}
            for q in state.qubits:
                substates[self.get_state_containing_qubit(q)] = 0
            substates[state[None]] = 0
            return tuple(substates.keys())
        return [state.create_merged_state()]

    def _get_merged_sim_state(self) -> TSimulationState:
        if self._merged_sim_state_cache is None:
            self._merged_sim_state_cache = self._final_simulator_state.create_merged_state()
        return self._merged_sim_state_cache
