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
)

import numpy as np

from cirq import circuits, ops, protocols, study, value, devices
from cirq.sim.simulator import (
    TStepResult,
    TSimulationTrialResult,
    TSimulatorState,
    TActOnArgs,
    SimulatesIntermediateState,
    SimulatesSamples,
    check_all_resolved,
    split_into_matching_protocol_then_general,
)

if TYPE_CHECKING:
    import cirq


class SimulatorBase(
    Generic[TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs],
    SimulatesIntermediateState[TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs],
    SimulatesSamples,
    metaclass=abc.ABCMeta,
):
    """A base class for the built-in simulators.

    Most implementors of this interface should implement the
    `_create_act_on_arg` and `_create_step_result` methods. The first one
    creates the simulator's quantum state representation at the beginning of
    the simulation. The second creates the step result emitted after each
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
    def _create_act_on_arg(
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
        sim_state: TActOnArgs,
        qubit_map: Dict['cirq.Qid', int],
    ) -> TStepResult:
        """This method should be implemented to create a step result.

        Args:
            sim_state: The TActOnArgs for this trial.
            qubit_map: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

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
        sim_state: Dict['cirq.Qid', TActOnArgs],
        qubits: Sequence['cirq.Qid'],
        all_measurements_are_terminal: bool = False,
    ) -> Iterator[TStepResult]:
        """Standard iterator over StepResult from Moments of a Circuit.

        Args:
            circuit: The circuit to simulate.
            sim_state: The initial args for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Yields:
            StepResults from simulating a Moment of the Circuit.
        """

        # Create a default target for when there's no qubits or for qubit-free ops like GlobalPhase
        if self._split_untangled_states or len(qubits) == 0:
            first_value = None if not any(sim_state) else next(iter(sim_state.values()))
            logs = {} if first_value is None else first_value.log_of_measurement_results
            default_arg = self._create_act_on_arg(0, (), logs)
        else:
            default_arg = next(iter(sim_state.values()))

        def merge_states(sim_state: Dict['cirq.Qid', TActOnArgs]) -> TActOnArgs:
            if not self._split_untangled_states:
                return default_arg
            final_args = default_arg
            for args in set(sim_state.values()):
                final_args = final_args.join(args)
            return final_args.reorder(qubits)

        if len(circuit) == 0:
            step_state = merge_states(sim_state)
            yield self._create_step_result(step_state, step_state.qubit_map)
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

                    # Go through the op's qubits and join any disparate ActOnArgs states
                    # into a new combined state.
                    op_args: Optional[TActOnArgs] = None
                    for q in op.qubits if len(op.qubits) != 0 else qubits:
                        if op_args is None:
                            op_args = sim_state[q]
                        elif q not in op_args.qubits:
                            op_args = op_args.join(sim_state[q])
                    op_args = op_args or default_arg

                    # (Backfill the args map with the new value)
                    for q in op_args.qubits:
                        sim_state[q] = op_args

                    # Act on the args with the operation
                    op_args.axes = tuple(op_args.qubit_map[q] for q in op.qubits)
                    protocols.act_on(op, op_args)

                    # Decouple any measurements or resets
                    if self._split_untangled_states and isinstance(
                        op.gate, (ops.MeasurementGate, ops.ResetChannel)
                    ):
                        for q in op.qubits:
                            q_args, op_args = op_args.extract((q,))
                            sim_state[q] = q_args

                        # (Backfill the args map with the new value)
                        for q in op_args.qubits:
                            sim_state[q] = op_args

                except TypeError:
                    raise TypeError(f"{self.__class__.__name__} doesn't support {op!r}")

            step_state = merge_states(sim_state)
            yield self._create_step_result(step_state, step_state.qubit_map)
            step_state.log_of_measurement_results.clear()

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
            qubits=qubits,
        ):
            pass

        general_ops = list(general_suffix.all_operations())
        if all(isinstance(op.gate, ops.MeasurementGate) for op in general_ops):
            for step_result in self._core_iterator(
                circuit=general_suffix,
                sim_state=act_on_args,
                qubits=qubits,
                all_measurements_are_terminal=True,
            ):
                pass
            assert step_result is not None
            measurement_ops = [cast(ops.GateOperation, op) for op in general_ops]
            return step_result.sample_measurement_ops(measurement_ops, repetitions, seed=self._prng)

        measurements: Dict[str, List[np.ndarray]] = {}
        for i in range(repetitions):
            if i < repetitions - 1:
                copies = {a: a.copy() for a in set(act_on_args.values())}
                sim_state = {q: copies[a] for q, a in act_on_args.items()}
            else:
                sim_state = act_on_args

            all_step_results = self._core_iterator(
                general_suffix,
                sim_state=sim_state,
                qubits=qubits,
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
    ) -> Dict['cirq.Qid', TActOnArgs]:
        if isinstance(initial_state, dict):
            return initial_state

        args_map: Dict['cirq.Qid', TActOnArgs] = {}
        if isinstance(initial_state, int) and self._split_untangled_states:
            log: Dict[str, Any] = {}
            for q in reversed(qubits):
                args_map[q] = self._create_act_on_arg(
                    initial_state=initial_state % q.dimension,
                    qubits=[q],
                    logs=log,
                )
                initial_state = int(initial_state / q.dimension)
            return args_map

        args = self._create_act_on_arg(
            initial_state=initial_state,
            qubits=qubits,
            logs={},
        )
        for q in qubits:
            args_map[q] = args
        return args_map
