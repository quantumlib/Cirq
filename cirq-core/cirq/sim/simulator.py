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

"""Abstract base classes for different types of simulators.

Simulator types include:

    SimulatesSamples: mimics the interface of quantum hardware.

    SimulatesAmplitudes: computes amplitudes of desired bitstrings in the
        final state of the simulation.

    SimulatesFinalState: allows access to the final state of the simulation.

    SimulatesIntermediateState: allows for access to the state of the simulation
        as the simulation iterates through the moments of a cirq.
"""

import abc
import collections
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import numpy as np

from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase

if TYPE_CHECKING:
    import cirq


TStepResult = TypeVar('TStepResult', bound='StepResult')
TSimulationTrialResult = TypeVar('TSimulationTrialResult', bound='SimulationTrialResult')
TSimulatorState = TypeVar('TSimulatorState', bound=Any)


class SimulatesSamples(work.Sampler, metaclass=abc.ABCMeta):
    """Simulator that mimics running on quantum hardware.

    Implementors of this interface should implement the _run method.
    """

    def run_sweep(
        self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', repetitions: int = 1
    ) -> Sequence['cirq.Result']:
        return list(self.run_sweep_iter(program, params, repetitions))

    def run_sweep_iter(
        self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', repetitions: int = 1
    ) -> Iterator['cirq.Result']:
        """Runs the supplied Circuit, mimicking quantum hardware.

        In contrast to run, this allows for sweeping over different parameter
        values.

        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.

        Returns:
            Result list for this run; one for each possible parameter
            resolver.

        Raises:
            ValueError: If the circuit has no measurements.
        """
        if not program.has_measurements():
            raise ValueError("Circuit has no measurements to sample.")

        for param_resolver in study.to_resolvers(params):
            records = {}
            if repetitions == 0:
                for _, op, _ in program.findall_operations_with_gate_type(ops.MeasurementGate):
                    records[protocols.measurement_key_name(op)] = np.empty([0, 1, 1])
            else:
                records = self._run(
                    circuit=program, param_resolver=param_resolver, repetitions=repetitions
                )
            yield study.ResultDict(params=param_resolver, records=records)

    @abc.abstractmethod
    def _run(
        self,
        circuit: 'cirq.AbstractCircuit',
        param_resolver: 'cirq.ParamResolver',
        repetitions: int,
    ) -> Dict[str, np.ndarray]:
        """Run a simulation, mimicking quantum hardware.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: Number of times to repeat the run. It is expected that
                this is validated greater than zero before calling this method.

        Returns:
            A dictionary from measurement gate key to measurement
            results. Measurement results are stored in a 3-dimensional
            numpy array, the first dimension corresponding to the repetition.
            the second to the instance of that key in the circuit, and the
            third to the actual boolean measurement results (ordered by the
            qubits being measured.)
        """
        raise NotImplementedError()


class SimulatesAmplitudes(metaclass=value.ABCMetaImplementAnyOneOf):
    """Simulator that computes final amplitudes of given bitstrings.

    Given a circuit and a list of bitstrings, computes the amplitudes
    of the given bitstrings in the state obtained by applying the circuit
    to the all zeros state. Implementors of this interface should implement
    the compute_amplitudes_sweep_iter method.
    """

    def compute_amplitudes(
        self,
        program: 'cirq.AbstractCircuit',
        bitstrings: Sequence[int],
        param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ) -> Sequence[complex]:
        """Computes the desired amplitudes.

        The initial state is assumed to be the all zeros state.

        Args:
            program: The circuit to simulate.
            bitstrings: The bitstrings whose amplitudes are desired, input
                as an integer array where each integer is formed from measured
                qubit values according to `qubit_order` from most to least
                significant qubit, i.e. in big-endian ordering. If inputting
                a binary literal add the prefix 0b or 0B.
                For example: 0010 can be input as 0b0010, 0B0010, 2, 0x2, etc.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

        Returns:
            List of amplitudes.
        """
        return self.compute_amplitudes_sweep(
            program, bitstrings, study.ParamResolver(param_resolver), qubit_order
        )[0]

    def compute_amplitudes_sweep(
        self,
        program: 'cirq.AbstractCircuit',
        bitstrings: Sequence[int],
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ) -> Sequence[Sequence[complex]]:
        """Wraps computed amplitudes in a list.

        Prefer overriding `compute_amplitudes_sweep_iter`.
        """
        return list(self.compute_amplitudes_sweep_iter(program, bitstrings, params, qubit_order))

    def _compute_amplitudes_sweep_to_iter(
        self,
        program: 'cirq.AbstractCircuit',
        bitstrings: Sequence[int],
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ) -> Iterator[Sequence[complex]]:
        if type(self).compute_amplitudes_sweep == SimulatesAmplitudes.compute_amplitudes_sweep:
            raise RecursionError(
                "Must define either compute_amplitudes_sweep or compute_amplitudes_sweep_iter."
            )
        yield from self.compute_amplitudes_sweep(program, bitstrings, params, qubit_order)

    @value.alternative(
        requires='compute_amplitudes_sweep', implementation=_compute_amplitudes_sweep_to_iter
    )
    def compute_amplitudes_sweep_iter(
        self,
        program: 'cirq.AbstractCircuit',
        bitstrings: Sequence[int],
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ) -> Iterator[Sequence[complex]]:
        """Computes the desired amplitudes.

        The initial state is assumed to be the all zeros state.

        Args:
            program: The circuit to simulate.
            bitstrings: The bitstrings whose amplitudes are desired, input
                as an integer array where each integer is formed from measured
                qubit values according to `qubit_order` from most to least
                significant qubit, i.e. in big-endian ordering. If inputting
                a binary literal add the prefix 0b or 0B.
                For example: 0010 can be input as 0b0010, 0B0010, 2, 0x2, etc.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

        Returns:
            An Iterator over lists of amplitudes. The outer dimension indexes
            the circuit parameters and the inner dimension indexes bitstrings.
        """
        raise NotImplementedError()

    def sample_from_amplitudes(
        self,
        circuit: 'cirq.AbstractCircuit',
        param_resolver: 'cirq.ParamResolver',
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE',
        repetitions: int = 1,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ) -> Dict[int, int]:
        """Uses amplitude simulation to sample from the given circuit.

        This implements the algorithm outlined by Bravyi, Gosset, and Liu in
        https://arxiv.org/abs/2112.08499 to more efficiently calculate samples
        given an amplitude-based simulator.

        Simulators which also implement SimulatesSamples or SimulatesFullState
        should prefer `run()` or `simulate()`, respectively, as this method
        only accelerates sampling for amplitude-based simulators.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            seed: Random state to use as a seed. This must be provided
                manually - if the simulator has its own seed, it will not be
                used unless it is passed as this argument.
            repetitions: The number of repetitions to simulate.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

        Returns:
            A dict of bitstrings sampled from the final state of `circuit` to
            the number of occurrences of that bitstring.

        Raises:
            ValueError: if 'circuit' has non-unitary elements, as differences
                in behavior between sampling steps break this algorithm.
        """
        prng = value.parse_random_state(seed)
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
        base_circuit = circuits.Circuit(ops.I(q) for q in qubits) + circuit.unfreeze()
        qmap = {q: i for i, q in enumerate(qubits)}
        current_samples = {(0,) * len(qubits): repetitions}
        solved_circuit = protocols.resolve_parameters(base_circuit, param_resolver)
        if not protocols.has_unitary(solved_circuit):
            raise ValueError("sample_from_amplitudes does not support non-unitary behavior.")
        if protocols.is_measurement(solved_circuit):
            raise ValueError("sample_from_amplitudes does not support intermediate measurement.")
        for m_id, moment in enumerate(solved_circuit[1:]):
            circuit_prefix = solved_circuit[: m_id + 1]
            for t, op in enumerate(moment.operations):
                new_samples: Dict[Tuple[int, ...], int] = collections.defaultdict(int)
                qubit_indices = {qmap[q] for q in op.qubits}
                subcircuit = circuit_prefix + circuits.Moment(moment.operations[: t + 1])
                for current_sample, count in current_samples.items():
                    sample_set = [current_sample]
                    for idx in qubit_indices:
                        sample_set = [
                            target[:idx] + (result,) + target[idx + 1 :]
                            for target in sample_set
                            for result in [0, 1]
                        ]
                    bitstrings = [int(''.join(map(str, sample)), base=2) for sample in sample_set]
                    amps = self.compute_amplitudes(subcircuit, bitstrings, qubit_order=qubit_order)
                    weights = np.abs(np.square(np.array(amps))).astype(np.float64)
                    weights /= np.linalg.norm(weights, 1)
                    subsample = prng.choice(len(sample_set), p=weights, size=count)
                    for sample_index in subsample:
                        new_samples[sample_set[sample_index]] += 1
                current_samples = new_samples

        return {int(''.join(map(str, k)), base=2): v for k, v in current_samples.items()}


class SimulatesExpectationValues(metaclass=value.ABCMetaImplementAnyOneOf):
    """Simulator that computes exact expectation values of observables.

    Given a circuit and an observable map, computes exact (to float precision)
    expectation values for each observable at the end of the circuit.

    Implementors of this interface should implement the
    simulate_expectation_values_sweep_iter method.
    """

    def simulate_expectation_values(
        self,
        program: 'cirq.AbstractCircuit',
        observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']],
        param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
        permit_terminal_measurements: bool = False,
    ) -> List[float]:
        """Simulates the supplied circuit and calculates exact expectation
        values for the given observables on its final state.

        This method has no perfect analogy in hardware. Instead compare with
        Sampler.sample_expectation_values, which calculates estimated
        expectation values by sampling multiple times.

        Args:
            program: The circuit to simulate.
            observables: An observable or list of observables.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.
            permit_terminal_measurements: If the provided circuit ends with
                measurement(s), this method will generate an error unless this
                is set to True. This is meant to prevent measurements from
                ruining expectation value calculations.

        Returns:
            A list of expectation values, with the value at index `n`
            corresponding to `observables[n]` from the input.

        Raises:
            ValueError if 'program' has terminal measurement(s) and
            'permit_terminal_measurements' is False.
        """
        return self.simulate_expectation_values_sweep(
            program,
            observables,
            study.ParamResolver(param_resolver),
            qubit_order,
            initial_state,
            permit_terminal_measurements,
        )[0]

    def simulate_expectation_values_sweep(
        self,
        program: 'cirq.AbstractCircuit',
        observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']],
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
        permit_terminal_measurements: bool = False,
    ) -> List[List[float]]:
        """Wraps computed expectation values in a list.

        Prefer overriding `simulate_expectation_values_sweep_iter`.
        """
        return list(
            self.simulate_expectation_values_sweep_iter(
                program,
                observables,
                params,
                qubit_order,
                initial_state,
                permit_terminal_measurements,
            )
        )

    def _simulate_expectation_values_sweep_to_iter(
        self,
        program: 'cirq.AbstractCircuit',
        observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']],
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
        permit_terminal_measurements: bool = False,
    ) -> Iterator[List[float]]:
        if (
            type(self).simulate_expectation_values_sweep
            == SimulatesExpectationValues.simulate_expectation_values_sweep
        ):
            raise RecursionError(
                "Must define either simulate_expectation_values_sweep or "
                "simulate_expectation_values_sweep_iter."
            )
        yield from self.simulate_expectation_values_sweep(
            program, observables, params, qubit_order, initial_state, permit_terminal_measurements
        )

    @value.alternative(
        requires='simulate_expectation_values_sweep',
        implementation=_simulate_expectation_values_sweep_to_iter,
    )
    def simulate_expectation_values_sweep_iter(
        self,
        program: 'cirq.AbstractCircuit',
        observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']],
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
        permit_terminal_measurements: bool = False,
    ) -> Iterator[List[float]]:
        """Simulates the supplied circuit and calculates exact expectation
        values for the given observables on its final state, sweeping over the
        given params.

        This method has no perfect analogy in hardware. Instead compare with
        Sampler.sample_expectation_values, which calculates estimated
        expectation values by sampling multiple times.

        Args:
            program: The circuit to simulate.
            observables: An observable or list of observables.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.
            permit_terminal_measurements: If the provided circuit ends in a
                measurement, this method will generate an error unless this
                is set to True. This is meant to prevent measurements from
                ruining expectation value calculations.

        Returns:
            An Iterator over expectation-value lists. The outer index determines
            the sweep, and the inner index determines the observable. For
            instance, results[1][3] would select the fourth observable measured
            in the second sweep.

        Raises:
            ValueError if 'program' has terminal measurement(s) and
            'permit_terminal_measurements' is False.
        """
        raise NotImplementedError


class SimulatesFinalState(
    Generic[TSimulationTrialResult], metaclass=value.ABCMetaImplementAnyOneOf
):
    """Simulator that allows access to the simulator's final state.

    Implementors of this interface should implement the simulate_sweep_iter
    method. This simulator only returns the state of the quantum system
    for the final step of a simulation. This simulator state may be a state
    vector, the density matrix, or another representation, depending on the
    implementation.  For simulators that also allow stepping through
    a circuit see `SimulatesIntermediateState`.
    """

    def simulate(
        self,
        program: 'cirq.AbstractCircuit',
        param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> TSimulationTrialResult:
        """Simulates the supplied Circuit.

        This method returns a result which allows access to the entire
        simulator's final state.

        Args:
            program: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            SimulationTrialResults for the simulation. Includes the final state.
        """
        return self.simulate_sweep(
            program, study.ParamResolver(param_resolver), qubit_order, initial_state
        )[0]

    def simulate_sweep(
        self,
        program: 'cirq.AbstractCircuit',
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> List[TSimulationTrialResult]:
        """Wraps computed states in a list.

        Prefer overriding `simulate_sweep_iter`.
        """
        return list(self.simulate_sweep_iter(program, params, qubit_order, initial_state))

    def _simulate_sweep_to_iter(
        self,
        program: 'cirq.AbstractCircuit',
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> Iterator[TSimulationTrialResult]:
        if type(self).simulate_sweep == SimulatesFinalState.simulate_sweep:
            raise RecursionError("Must define either simulate_sweep or simulate_sweep_iter.")
        yield from self.simulate_sweep(program, params, qubit_order, initial_state)

    @value.alternative(requires='simulate_sweep', implementation=_simulate_sweep_to_iter)
    def simulate_sweep_iter(
        self,
        program: 'cirq.AbstractCircuit',
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> Iterator[TSimulationTrialResult]:
        """Simulates the supplied Circuit.

        This method returns a result which allows access to the entire final
        simulator state. In contrast to simulate, this allows for sweeping
        over different parameter values.

        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            Iterator over SimulationTrialResults for this run, one for each
            possible parameter resolver.
        """
        raise NotImplementedError()


class SimulatesIntermediateState(
    Generic[TStepResult, TSimulationTrialResult, TSimulatorState],
    SimulatesFinalState[TSimulationTrialResult],
    metaclass=abc.ABCMeta,
):
    """A SimulatesFinalState that simulates a circuit by moments.

    Whereas a general SimulatesFinalState may return the entire simulator
    state at the end of a circuit, a SimulatesIntermediateState can
    simulate stepping through the moments of a circuit.

    Implementors of this interface should implement the _core_iterator
    method.

    Note that state here refers to simulator state, which is not necessarily
    a state vector.
    """

    def simulate_sweep_iter(
        self,
        program: 'cirq.AbstractCircuit',
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> Iterator[TSimulationTrialResult]:
        """Simulates the supplied Circuit.

        This method returns a result which allows access to the entire
        state vector. In contrast to simulate, this allows for sweeping
        over different parameter values.

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

        Returns:
            List of SimulationTrialResults for this run, one for each
            possible parameter resolver.
        """
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        resolvers = list(study.to_resolvers(params))
        for i, param_resolver in enumerate(resolvers):
            state = (
                initial_state.copy()
                if isinstance(initial_state, SimulationStateBase) and i < len(resolvers) - 1
                else initial_state
            )
            all_step_results = self.simulate_moment_steps(
                program, param_resolver, qubit_order, state
            )
            measurements: Dict[str, np.ndarray] = {}
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    measurements[k] = np.array(v, dtype=np.uint8)
            yield self._create_simulator_trial_result(
                params=param_resolver,
                measurements=measurements,
                final_simulator_state=step_result._simulator_state(),
            )

    def simulate_moment_steps(
        self,
        circuit: 'cirq.AbstractCircuit',
        param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> Iterator[TStepResult]:
        """Returns an iterator of StepResults for each moment simulated.

        If the circuit being simulated is empty, a single step result should
        be returned with the state being set to the initial state.

        Args:
            circuit: The Circuit to simulate.
            param_resolver: A ParamResolver for determining values of Symbols.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. This can be
                either a raw state or a `TSimulationState`. The form of the
                raw state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            Iterator that steps through the simulation, simulating each
            moment and returning a StepResult for each moment.
        """
        param_resolver = study.ParamResolver(param_resolver)
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        check_all_resolved(resolved_circuit)
        actual_initial_state = 0 if initial_state is None else initial_state
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
        return self._base_iterator(resolved_circuit, qubits, actual_initial_state)

    @abc.abstractmethod
    def _base_iterator(
        self, circuit: 'cirq.AbstractCircuit', qubits: Tuple['cirq.Qid', ...], initial_state: Any
    ) -> Iterator[TStepResult]:
        """Iterator over StepResult from Moments of a Circuit.

        Args:
            circuit: The circuit to simulate.
            qubits: Specifies the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Yields:
            StepResults from simulating a Moment of the Circuit.
        """

    @abc.abstractmethod
    def _create_simulator_trial_result(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: TSimulatorState,
    ) -> TSimulationTrialResult:
        """This method can be implemented to create a trial result.

        Args:
            params: The ParamResolver for this trial.
            measurements: The measurement results for this trial.
            final_simulator_state: The final state of the simulation.

        Returns:
            The SimulationTrialResult.
        """
        raise NotImplementedError()


class StepResult(Generic[TSimulatorState], metaclass=abc.ABCMeta):
    """Results of a step of a SimulatesIntermediateState.

    Attributes:
        measurements: A dictionary from measurement gate key to measurement
            results, ordered by the qubits that the measurement operates on.
    """

    def __init__(self, sim_state: TSimulatorState) -> None:
        self._sim_state = sim_state
        self._measurements = sim_state.log_of_measurement_results

    @property
    def measurements(self) -> Mapping[str, Sequence[int]]:
        return self._measurements

    def _simulator_state(self) -> TSimulatorState:
        """Returns the simulator state of the simulator after this step.

        This method starts with an underscore to indicate that it is private.
        To access public state, see public methods on StepResult.

        The form of the simulator_state depends on the implementation of the
        simulation,see documentation for the implementing class for the form of
        details.
        """
        return self._sim_state

    @abc.abstractmethod
    def sample(
        self,
        qubits: List['cirq.Qid'],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        """Samples from the system at this point in the computation.

        Note that this does not collapse the state vector.

        Args:
            qubits: The qubits to be sampled in an order that influence the
                returned measurement results.
            repetitions: The number of samples to take.
            seed: A seed for the pseudorandom number generator.

        Returns:
            Measurement results with True corresponding to the ``|1⟩`` state.
            The outer list is for repetitions, and the inner corresponds to
            measurements ordered by the supplied qubits. These lists
            are wrapped as a numpy ndarray.
        """
        raise NotImplementedError()

    def sample_measurement_ops(
        self,
        measurement_ops: List['cirq.GateOperation'],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        *,
        _allow_repeated=False,
    ) -> Dict[str, np.ndarray]:
        """Samples from the system at this point in the computation.

        Note that this does not collapse the state vector.

        In contrast to `sample` which samples qubits, this takes a list of
        `cirq.GateOperation` instances whose gates are `cirq.MeasurementGate`
        instances and then returns a mapping from the key in the measurement
        gate to the resulting bit strings. Different measurement operations must
        not act on the same qubits.

        Args:
            measurement_ops: `GateOperation` instances whose gates are
                `MeasurementGate` instances to be sampled form.
            repetitions: The number of samples to take.
            seed: A seed for the pseudorandom number generator.
            _allow_repeated: If True, adds extra dimension to the result,
                corresponding to the number of times a key is repeated.

        Returns: A dictionary from measurement gate key to measurement
            results. Measurement results are stored in a 2-dimensional
            numpy array, the first dimension corresponding to the repetition
            and the second to the actual boolean measurement results (ordered
            by the qubits being measured.)

        Raises:
            ValueError: If the operation's gates are not `MeasurementGate`
                instances or a qubit is acted upon multiple times by different
                operations from `measurement_ops`.
        """

        # Sanity checks.
        for op in measurement_ops:
            gate = op.gate
            if not isinstance(gate, ops.MeasurementGate):
                raise ValueError(f'{op.gate} was not a MeasurementGate')
        result = collections.Counter(
            key for op in measurement_ops for key in protocols.measurement_key_names(op)
        )
        if result and not _allow_repeated:
            duplicates = [k for k, v in result.most_common() if v > 1]
            if duplicates:
                raise ValueError(f"Measurement key {','.join(duplicates)} repeated")

        # Find measured qubits, ensuring a consistent ordering.
        measured_qubits = []
        seen_qubits: Set[cirq.Qid] = set()
        for op in measurement_ops:
            for q in op.qubits:
                if q not in seen_qubits:
                    seen_qubits.add(q)
                    measured_qubits.append(q)

        # Perform whole-system sampling of the measured qubits.
        indexed_sample = self.sample(measured_qubits, repetitions, seed=seed)

        # Extract results for each measurement.
        results: Dict[str, Any] = {}
        qubits_to_index = {q: i for i, q in enumerate(measured_qubits)}
        for op in measurement_ops:
            gate = cast(ops.MeasurementGate, op.gate)
            key = gate.key
            out = np.zeros(shape=(repetitions, len(op.qubits)), dtype=np.int8)
            inv_mask = gate.full_invert_mask()
            cmap = gate.confusion_map
            for i, q in enumerate(op.qubits):
                out[:, i] = indexed_sample[:, qubits_to_index[q]]
                if inv_mask[i]:
                    out[:, i] ^= out[:, i] < 2
            self._confuse_results(out, op.qubits, cmap, seed)
            if _allow_repeated:
                if key not in results:
                    results[key] = []
                results[key].append(out)
            else:
                results[gate.key] = out
        return (
            results
            if not _allow_repeated
            else {k: np.array(v).swapaxes(0, 1) for k, v in results.items()}
        )

    def _confuse_results(
        self,
        bits: np.ndarray,
        qubits: Sequence['cirq.Qid'],
        confusion_map: Dict[Tuple[int, ...], np.ndarray],
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> None:
        """Mutates `bits` using the confusion_map.

        Compare with _confuse_result in cirq-core/cirq/sim/simulation_state.py.
        """
        prng = value.parse_random_state(seed)
        for rep in bits:
            dims = [q.dimension for q in qubits]
            for indices, confuser in confusion_map.items():
                mat_dims = [dims[k] for k in indices]
                row = value.big_endian_digits_to_int((rep[k] for k in indices), base=mat_dims)
                new_val = prng.choice(len(confuser), p=confuser[row])
                new_bits = value.big_endian_int_to_digits(new_val, base=mat_dims)
                for i, k in enumerate(indices):
                    rep[k] = new_bits[i]


@value.value_equality(unhashable=True)
class SimulationTrialResult(Generic[TSimulatorState]):
    """Results of a simulation by a SimulatesFinalState.

    Unlike `cirq.Result`, a SimulationTrialResult contains the final
    simulator_state of the system. This simulator_state is dependent on the
    simulation implementation and may be, for example, the state vector
    or the density matrix of the system.

    Attributes:
        params: A ParamResolver of settings used for this result.
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are a numpy ndarray of actual boolean
            measurement results (ordered by the qubits acted on by the
            measurement gate.)
    """

    def __init__(
        self,
        params: 'cirq.ParamResolver',
        measurements: Mapping[str, np.ndarray],
        final_simulator_state: TSimulatorState,
    ) -> None:
        """Initializes the `SimulationTrialResult` class.

        Args:
            params: A ParamResolver of settings used for this result.
            measurements: A mapping from measurement gate key to measurement
                results. Measurement results are a numpy ndarray of actual
                boolean measurement results (ordered by the qubits acted on by
                the measurement gate.)
            final_simulator_state: The final simulator state.
        """
        self._params = params
        self._measurements = measurements
        self._final_simulator_state = final_simulator_state

    @property
    def params(self) -> 'cirq.ParamResolver':
        return self._params

    @property
    def measurements(self) -> Mapping[str, np.ndarray]:
        return self._measurements

    def __repr__(self) -> str:
        return (
            f'cirq.SimulationTrialResult(params={self.params!r}, '
            f'measurements={self.measurements!r}, '
            f'final_simulator_state={self._final_simulator_state!r})'
        )

    def __str__(self) -> str:
        def bitstring(vals):
            separator = ' ' if np.max(vals) >= 10 else ''
            return separator.join(str(v.item()) for v in vals)

        results = sorted([(key, bitstring(val)) for key, val in self.measurements.items()])
        if not results:
            return '(no measurements)'
        return ' '.join([f'{key}={val}' for key, val in results])

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('SimulationTrialResult(...)')
        else:
            p.text(str(self))

    def _value_equality_values_(self) -> Any:
        measurements = {k: v.tolist() for k, v in sorted(self.measurements.items())}
        return self.params, measurements, self._final_simulator_state

    @property
    def qubit_map(self) -> Mapping['cirq.Qid', int]:
        """A map from Qid to index used to define the ordering of the basis in
        the result.
        """
        return self._final_simulator_state.qubit_map

    def _qid_shape_(self) -> Tuple[int, ...]:
        return _qubit_map_to_shape(self.qubit_map)


def _qubit_map_to_shape(qubit_map: Mapping['cirq.Qid', int]) -> Tuple[int, ...]:
    qid_shape: List[int] = [-1] * len(qubit_map)
    try:
        for q, i in qubit_map.items():
            qid_shape[i] = q.dimension
    except IndexError:
        raise ValueError(f'Invalid qubit_map. Qubit index out of bounds. Map is <{qubit_map!r}>.')
    if -1 in qid_shape:
        raise ValueError(f'Invalid qubit_map. Duplicate qubit index. Map is <{qubit_map!r}>.')
    return tuple(qid_shape)


def check_all_resolved(circuit):
    """Raises if the circuit contains unresolved symbols."""
    if protocols.is_parameterized(circuit):
        unresolved = [op for moment in circuit for op in moment if protocols.is_parameterized(op)]
        raise ValueError(
            'Circuit contains ops whose symbols were not specified in '
            f'parameter sweep. Ops: {unresolved}'
        )


def split_into_matching_protocol_then_general(
    circuit: 'cirq.AbstractCircuit', predicate: Callable[['cirq.Operation'], bool]
) -> Tuple['cirq.AbstractCircuit', 'cirq.AbstractCircuit']:
    """Splits the circuit into a matching prefix and non-matching suffix.

    The splitting happens in a per-qubit fashion. A non-matching operation on
    qubit A will cause later operations on A to be part of the non-matching
    suffix, but later operations on other qubits will continue to be put into
    the matching part (as long as those qubits have had no non-matching operation
    up to that point).
    """
    blocked_qubits: Set[cirq.Qid] = set()
    matching_prefix = circuits.Circuit()
    general_suffix = circuits.Circuit()
    for moment in circuit:
        matching_part = []
        general_part = []
        for op in moment:
            qs = set(op.qubits)
            if not predicate(op) or not qs.isdisjoint(blocked_qubits):
                blocked_qubits |= qs

            if qs.isdisjoint(blocked_qubits):
                matching_part.append(op)
            else:
                general_part.append(op)
        if matching_part:
            matching_prefix.append(circuits.Moment(matching_part))
        if general_part:
            general_suffix.append(circuits.Moment(general_part))
    return matching_prefix, general_suffix
