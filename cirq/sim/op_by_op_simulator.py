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

"""A simulator that uses numpy's einsum for sparse matrix operations."""
import abc
import collections
from typing import (
    Dict,
    Iterator,
    List,
    TYPE_CHECKING,
    Tuple,
    cast,
    Set,
    TypeVar,
    Generic,
)

import numpy as np

from cirq import circuits, ops, protocols, study, devices
from cirq.sim import (
    simulator,
)
from cirq.sim.abstract_state import AbstractState
from cirq.sim.simulator import (
    check_all_resolved,
    SimulationTrialResult,
    StepResult,
    SimulatesIntermediateState,
)

if TYPE_CHECKING:
    import cirq


TState = TypeVar('TState', bound=AbstractState)


class StateFactory(Generic[TState], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_sim_state(self, initial_state, qubits) -> TState:
        raise NotImplementedError()

    @abc.abstractmethod
    def act_on_state(self, op, sim_state: TState, qubit_map):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def prng(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def retains_noise(self):
        raise NotImplementedError()

    def keep(self, op):
        return True

    def on_stuck(self, op):
        raise NotImplementedError()


TStepResult = TypeVar('TStepResult', bound=StepResult)
TSimulationTrialResult = TypeVar('TSimulationTrialResult', bound=SimulationTrialResult)
TFinalState = TypeVar('TFinalState')


class SimulationResultFactory(
    Generic[TState, TStepResult, TSimulationTrialResult, TFinalState], metaclass=abc.ABCMeta
):
    @abc.abstractmethod
    def step_result(self, sim_state: TState, qubit_map) -> TStepResult:
        raise NotImplementedError()

    @abc.abstractmethod
    def trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: TFinalState,
    ) -> TSimulationTrialResult:
        raise NotImplementedError()


class OpByOpSimulator(
    Generic[TState, TStepResult, TSimulationTrialResult, TFinalState],
    simulator.SimulatesSamples,
    SimulatesIntermediateState,
):
    def __init__(
        self,
        state_algo: StateFactory[TState],
        result_producer: SimulationResultFactory[
            TState, TStepResult, TSimulationTrialResult, TFinalState
        ],
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
    ):
        self.state_algo = state_algo
        self.result_producer = result_producer
        self.noise = devices.NoiseModel.from_noise_model_like(noise)

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
        unitary_prefix, general_suffix = _split_into_unitary_then_general(resolved_circuit)
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
                seed=self.state_algo.prng,
            )

        intermediate_state = step_result.state_vector()
        if self.state_algo.retains_noise and resolved_circuit.are_all_measurements_terminal():
            return self._run_sweep_sample(
                intermediate_state, general_suffix, qubit_order, repetitions
            )
        return self._run_sweep_repeat(intermediate_state, general_suffix, qubit_order, repetitions)

    def _run_sweep_sample(
        self,
        initial_state: 'cirq.STATE_VECTOR_LIKE',
        circuit: circuits.Circuit,
        qubit_order: 'cirq.QubitOrderOrList',
        repetitions: int,
    ) -> Dict[str, np.ndarray]:
        step_result = None
        for step_result in self._base_iterator(
            circuit=circuit,
            qubit_order=qubit_order,
            initial_state=initial_state,
            all_measurements_are_terminal=True,
        ):
            pass
        measurement_ops = [
            op for _, op, _ in circuit.findall_operations_with_gate_type(ops.MeasurementGate)
        ]

        return step_result.sample_measurement_ops(
            measurement_ops, repetitions, seed=self.state_algo.prng
        )

    def _run_sweep_repeat(
        self,
        initial_state: 'cirq.STATE_VECTOR_LIKE',
        circuit: circuits.Circuit,
        qubit_order: 'cirq.QubitOrderOrList',
        repetitions: int,
    ) -> Dict[str, np.ndarray]:
        measurements = {}  # type: Dict[str, List[np.ndarray]]
        if repetitions == 0:
            for _, op, _ in circuit.findall_operations_with_gate_type(ops.MeasurementGate):
                measurements[protocols.measurement_key(op)] = np.empty([0, 1])

        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                circuit, qubit_order=qubit_order, initial_state=initial_state
            )
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    if k not in measurements:
                        measurements[k] = []
                    measurements[k].append(np.array(v, dtype=np.uint8))
        return {k: np.array(v) for k, v in measurements.items()}

    def _base_iterator(
        self,
        circuit: circuits.Circuit,
        qubit_order: ops.QubitOrderOrList,
        initial_state: 'cirq.STATE_VECTOR_LIKE',
        perform_measurements: bool = True,
        all_measurements_are_terminal: bool = False,
    ) -> Iterator[TStepResult]:
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
        sim_state = (
            cast(TState, initial_state)
            if isinstance(initial_state, AbstractState)
            else self.state_algo.create_sim_state(initial_state, qubits)
        )
        qubit_map = {q: i for i, q in enumerate(qubits)}
        if len(circuit) == 0:
            yield self.result_producer.step_result(sim_state, qubit_map)
            return
        measured = collections.defaultdict(bool)  # type: Dict[Tuple[cirq.Qid, ...], bool]

        noisy_moments = self.noise.noisy_moments(circuit, sorted(circuit.all_qubits()))
        for moment in noisy_moments:
            operations = protocols.decompose(
                moment, keep=self.state_algo.keep, on_stuck_raise=self.state_algo.on_stuck
            )
            for op in operations:
                if all_measurements_are_terminal and measured[op.qubits]:
                    continue
                if isinstance(op.gate, ops.MeasurementGate):
                    measured[op.qubits] = True
                    if all_measurements_are_terminal:
                        continue
                if perform_measurements or not isinstance(op.gate, ops.MeasurementGate):
                    self.state_algo.act_on_state(op, sim_state, qubit_map)

            yield self.result_producer.step_result(sim_state, qubit_map)
            sim_state.log_of_measurement_results.clear()

    def _create_simulator_trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: TFinalState,
    ) -> TSimulationTrialResult:
        """This method can be overridden to creation of a trial result.

        Args:
            params: The ParamResolver for this trial.
            measurements: The measurement results for this trial.
            final_simulator_state: The final state of the simulator for the
                StepResult.

        Returns:
            The SimulationTrialResult.
        """
        return self.result_producer.trial_result(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )


def _split_into_unitary_then_general(
    circuit: 'cirq.Circuit',
) -> Tuple['cirq.Circuit', 'cirq.Circuit']:
    """Splits the circuit into a unitary prefix and non-unitary suffix.

    The splitting happens in a per-qubit fashion. A non-unitary operation on
    qubit A will cause later operations on A to be part of the non-unitary
    suffix, but later operations on other qubits will continue to be put into
    the unitary part (as long as those qubits have had no non-unitary operation
    up to that point).
    """
    blocked_qubits: Set[cirq.Qid] = set()
    unitary_prefix = circuits.Circuit()
    general_suffix = circuits.Circuit()
    for moment in circuit:
        unitary_part = []
        general_part = []
        for op in moment:
            qs = set(op.qubits)
            if not protocols.has_unitary(op) or not qs.isdisjoint(blocked_qubits):
                blocked_qubits |= qs

            if qs.isdisjoint(blocked_qubits):
                unitary_part.append(op)
            else:
                general_part.append(op)
        if unitary_part:
            unitary_prefix.append(ops.Moment(unitary_part))
        if general_part:
            general_suffix.append(ops.Moment(general_part))
    return unitary_prefix, general_suffix
