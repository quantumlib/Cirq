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

import abc
import collections
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Tuple,
    TYPE_CHECKING,
    cast,
    Generic,
    Type,
    Sequence,
    Optional,
    TypeVar,
)

import numpy as np

from cirq import circuits, ops, protocols, study, value, devices
from cirq.sim import ActOnArgsContainer
from cirq.sim.operation_target import OperationTarget
from cirq.sim.simulator import (
    TSimulationTrialResult,
    TSimulatorState,
    TActOnArgs,
    SimulatesIntermediateState,
    SimulatesSamples,
    StepResult,
    check_all_resolved,
    split_into_matching_protocol_then_general,
)

if TYPE_CHECKING:
    import cirq


TStepResultBase = TypeVar('TStepResultBase', bound='StepResultBase')


class SimulatorBase(
    Generic[TStepResultBase, TSimulationTrialResult, TSimulatorState, TActOnArgs],
    SimulatesIntermediateState[
        TStepResultBase, TSimulationTrialResult, TSimulatorState, TActOnArgs
    ],
    SimulatesSamples,
    metaclass=abc.ABCMeta,
):
    """A base class for the built-in simulators.

    Most implementors of this interface should implement the
    `_create_partial_act_on_args` and `_create_step_result` methods. The first
    one creates the simulator's quantum state representation at the beginning
    of the simulation. The second creates the step result emitted after each
    `Moment` in the simulation.

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
        dtype: Type[np.number] = np.complex64,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        ignore_measurement_results: bool = False,
        split_untangled_states: bool = False,
    ):
        """Initializes the simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation.
            noise: A noise model to apply while simulating.
            seed: The random seed to use for this simulator.
            ignore_measurement_results: If True, then the simulation
                will treat measurement as dephasing instead of collapsing
                process. This is only applicable to simulators that can
                model dephasing.
            split_untangled_states: If True, optimizes simulation by running
                unentangled qubit sets independently and merging those states
                at the end.
        """
        self._dtype = dtype
        self._prng = value.parse_random_state(seed)
        self.noise = devices.NoiseModel.from_noise_model_like(noise)
        self._ignore_measurement_results = ignore_measurement_results
        self._split_untangled_states = split_untangled_states

    @abc.abstractmethod
    def _create_partial_act_on_args(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
        logs: Dict[str, Any],
    ) -> TActOnArgs:
        """Creates an instance of the TActOnArgs class for the simulator.

        It represents the supplied qubits initialized to the provided state.

        Args:
            initial_state: The initial state to represent. An integer state is
                understood to be a pure state. Other state representations are
                simulator-dependent.
            qubits: The sequence of qubits to represent.
            logs: The structure to hold measurement logs. A single instance
                should be shared among all ActOnArgs within the simulation.
        """

    @abc.abstractmethod
    def _create_step_result(
        self,
        sim_state: OperationTarget[TActOnArgs],
    ) -> TStepResultBase:
        """This method should be implemented to create a step result.

        Args:
            sim_state: The OperationTarget for this trial.

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
        be represented wholely in the matrix, and thus this method is
        overridden there to enable greater optimization there.

        Custom simulators can override this method appropriately.

        Args:
            val: An operation or noise model to test for purity within the
                state representation.

        Returns:
            A boolean representing whether the value can be added to the
            `_run` prefix."""
        return protocols.has_unitary(val)

    def _core_iterator(
        self,
        circuit: circuits.Circuit,
        sim_state: OperationTarget[TActOnArgs],
        all_measurements_are_terminal: bool = False,
    ) -> Iterator[TStepResultBase]:
        """Standard iterator over StepResult from Moments of a Circuit.

        Args:
            circuit: The circuit to simulate.
            sim_state: The initial args for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Yields:
            StepResults from simulating a Moment of the Circuit.
        """

        if len(circuit) == 0:
            yield self._create_step_result(sim_state)
            return

        noisy_moments = self.noise.noisy_moments(circuit, sorted(circuit.all_qubits()))
        measured: Dict[Tuple['cirq.Qid', ...], bool] = collections.defaultdict(bool)
        for moment in noisy_moments:
            for op in ops.flatten_to_ops(moment):
                try:
                    # TODO: support more general measurements.
                    # Github issue: https://github.com/quantumlib/Cirq/issues/3566

                    # Preprocess measurements
                    if all_measurements_are_terminal and measured[op.qubits]:
                        continue
                    if isinstance(op.gate, ops.MeasurementGate):
                        measured[op.qubits] = True
                        if all_measurements_are_terminal:
                            continue
                        if self._ignore_measurement_results:
                            op = ops.phase_damp(1).on(*op.qubits)

                    # Simulate the operation
                    sim_state.apply_operation(op)
                except TypeError:
                    raise TypeError(f"{self.__class__.__name__} doesn't support {op!r}")

            step_result = self._create_step_result(sim_state)
            yield step_result
            sim_state = step_result._sim_state
            sim_state.log_of_measurement_results.clear()

    def _run(
        self, circuit: circuits.Circuit, param_resolver: study.ParamResolver, repetitions: int
    ) -> Dict[str, np.ndarray]:
        """See definition in `cirq.SimulatesSamples`."""
        if self._ignore_measurement_results:
            raise ValueError("run() is not supported when ignore_measurement_results = True")

        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        check_all_resolved(resolved_circuit)
        qubits = tuple(sorted(resolved_circuit.all_qubits()))
        act_on_args = self._create_act_on_args(0, qubits)

        prefix, general_suffix = (
            split_into_matching_protocol_then_general(resolved_circuit, self._can_be_in_run_prefix)
            if self._can_be_in_run_prefix(self.noise)
            else (resolved_circuit[0:0], resolved_circuit)
        )
        step_result = None
        for step_result in self._core_iterator(
            circuit=prefix,
            sim_state=act_on_args,
        ):
            pass

        general_ops = list(general_suffix.all_operations())
        if all(isinstance(op.gate, ops.MeasurementGate) for op in general_ops):
            for step_result in self._core_iterator(
                circuit=general_suffix,
                sim_state=act_on_args,
                all_measurements_are_terminal=True,
            ):
                pass
            assert step_result is not None
            measurement_ops = [cast(ops.GateOperation, op) for op in general_ops]
            return step_result.sample_measurement_ops(measurement_ops, repetitions, seed=self._prng)

        measurements: Dict[str, List[np.ndarray]] = {}
        for i in range(repetitions):
            all_step_results = self._core_iterator(
                general_suffix,
                sim_state=act_on_args.copy() if i < repetitions - 1 else act_on_args,
            )
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    if k not in measurements:
                        measurements[k] = []
                    measurements[k].append(np.array(v, dtype=np.uint8))
        return {k: np.array(v) for k, v in measurements.items()}

    def _create_act_on_args(
        self,
        initial_state: Any,
        qubits: Sequence['cirq.Qid'],
    ) -> OperationTarget[TActOnArgs]:
        if isinstance(initial_state, OperationTarget):
            return initial_state

        log: Dict[str, Any] = {}
        if self._split_untangled_states:
            args_map: Dict[Optional['cirq.Qid'], TActOnArgs] = {}
            if isinstance(initial_state, int):
                for q in reversed(qubits):
                    args_map[q] = self._create_partial_act_on_args(
                        initial_state=initial_state % q.dimension,
                        qubits=[q],
                        logs=log,
                    )
                    initial_state = int(initial_state / q.dimension)
            else:
                args = self._create_partial_act_on_args(
                    initial_state=initial_state,
                    qubits=qubits,
                    logs=log,
                )
                for q in qubits:
                    args_map[q] = args
            args_map[None] = self._create_partial_act_on_args(0, (), log)
            return ActOnArgsContainer(args_map, qubits, self._split_untangled_states, log)
        else:
            return self._create_partial_act_on_args(
                initial_state=initial_state,
                qubits=qubits,
                logs=log,
            )


class StepResultBase(Generic[TSimulatorState, TActOnArgs], StepResult[TSimulatorState], abc.ABC):
    """A base class for step results."""

    def __init__(
        self,
        sim_state: OperationTarget[TActOnArgs],
    ):
        """Initializes the step result.

        Args:
            sim_state: The `OperationTarget` for this step.
        """
        self._sim_state = sim_state
        self._merged_sim_state_cache: Optional[TActOnArgs] = None
        super().__init__(sim_state.log_of_measurement_results)
        qubits = sim_state.qubits
        self._qubits = qubits
        self._qubit_mapping = {q: i for i, q in enumerate(qubits)}
        self._qubit_shape = tuple(q.dimension for q in qubits)

    def _qid_shape_(self):
        return self._qubit_shape

    @property
    def _merged_sim_state(self):
        if self._merged_sim_state_cache is None:
            self._merged_sim_state_cache = self._sim_state.create_merged_state()
        return self._merged_sim_state_cache

    def sample(
        self,
        qubits: List[ops.Qid],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        return self._sim_state.sample(qubits, repetitions, seed)
