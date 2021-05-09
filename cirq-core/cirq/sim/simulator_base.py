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
    `_create_act_on_args` and `_create_step_result` methods. The first one
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
        """
        self._dtype = dtype
        self._prng = value.parse_random_state(seed)
        self.noise = devices.NoiseModel.from_noise_model_like(noise)
        self._ignore_measurement_results = ignore_measurement_results

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
        sim_state: TActOnArgs,
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
        if len(circuit) == 0:
            yield self._create_step_result(sim_state, sim_state.qubit_map)
            return

        noisy_moments = self.noise.noisy_moments(circuit, sorted(circuit.all_qubits()))
        measured: Dict[Tuple[cirq.Qid, ...], bool] = collections.defaultdict(bool)
        for moment in noisy_moments:
            for op in ops.flatten_to_ops(moment):
                try:
                    # TODO: support more general measurements.
                    # Github issue: https://github.com/quantumlib/Cirq/issues/3566
                    if all_measurements_are_terminal and measured[op.qubits]:
                        continue
                    if isinstance(op.gate, ops.MeasurementGate):
                        measured[op.qubits] = True
                        if all_measurements_are_terminal:
                            continue
                        if self._ignore_measurement_results:
                            op = ops.phase_damp(1).on(*op.qubits)
                    sim_state.axes = tuple(sim_state.qubit_map[qubit] for qubit in op.qubits)
                    protocols.act_on(op, sim_state)
                except TypeError:
                    raise TypeError(f"{self.__class__.__name__} doesn't support {op!r}")

            yield self._create_step_result(sim_state, sim_state.qubit_map)
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
        for _ in range(repetitions):
            all_step_results = self._core_iterator(
                general_suffix,
                sim_state=act_on_args.copy(),
            )
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    if k not in measurements:
                        measurements[k] = []
                    measurements[k].append(np.array(v, dtype=np.uint8))
        return {k: np.array(v) for k, v in measurements.items()}
