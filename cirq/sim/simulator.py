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

from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Sequence,
    Tuple,
    Optional,
    TYPE_CHECKING,
)

import abc
import collections

import numpy as np

from cirq import circuits, ops, protocols, study, value, work

if TYPE_CHECKING:
    import cirq


class SimulatesSamples(work.Sampler, metaclass=abc.ABCMeta):
    """Simulator that mimics running on quantum hardware.

    Implementors of this interface should implement the _run method.
    """

    def run_sweep(
            self,
            program: 'cirq.Circuit',
            params: study.Sweepable,
            repetitions: int = 1,
    ) -> List[study.TrialResult]:
        """Runs the supplied Circuit, mimicking quantum hardware.

        In contrast to run, this allows for sweeping over different parameter
        values.

        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.

        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """
        if not program.has_measurements():
            raise ValueError("Circuit has no measurements to sample.")

        _verify_unique_measurement_keys(program)

        trial_results = []  # type: List[study.TrialResult]
        for param_resolver in study.to_resolvers(params):
            measurements = self._run(circuit=program,
                                     param_resolver=param_resolver,
                                     repetitions=repetitions)
            trial_results.append(
                study.TrialResult.from_single_parameter_set(
                    params=param_resolver, measurements=measurements))
        return trial_results

    @abc.abstractmethod
    def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int
    ) -> Dict[str, np.ndarray]:
        """Run a simulation, mimicking quantum hardware.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: Number of times to repeat the run.

        Returns:
            A dictionary from measurement gate key to measurement
            results. Measurement results are stored in a 2-dimensional
            numpy array, the first dimension corresponding to the repetition
            and the second to the actual boolean measurement results (ordered
            by the qubits being measured.)
        """
        raise NotImplementedError()


class SimulatesAmplitudes(metaclass=abc.ABCMeta):
    """Simulator that computes final amplitudes of given bitstrings.

    Given a circuit and a list of bitstrings, computes the amplitudes
    of the given bitstrings in the state obtained by applying the circuit
    to the all zeros state. Implementors of this interface should implement
    the compute_amplitudes_sweep method.
    """

    def compute_amplitudes(
            self,
            program: 'cirq.Circuit',
            bitstrings: Sequence[int],
            param_resolver: 'study.ParamResolverOrSimilarType' = None,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
    ) -> Sequence[complex]:
        """Computes the desired amplitudes.

        The initial state is assumed to be the all zeros state.

        Args:
            program: The circuit to simulate.
            bitstrings: The bitstrings whose amplitudes are desired, input
                as an integer array where each integer is formed from measured
                qubit values according to `qubit_order` from most to least
                significant qubit, i.e. in big-endian ordering.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

        Returns:
            List of amplitudes.
        """
        return self.compute_amplitudes_sweep(
            program, bitstrings, study.ParamResolver(param_resolver),
            qubit_order)[0]

    @abc.abstractmethod
    def compute_amplitudes_sweep(
            self,
            program: 'cirq.Circuit',
            bitstrings: Sequence[int],
            params: study.Sweepable,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
    ) -> Sequence[Sequence[complex]]:
        """Computes the desired amplitudes.

        The initial state is assumed to be the all zeros state.

        Args:
            program: The circuit to simulate.
            bitstrings: The bitstrings whose amplitudes are desired, input
                as an integer array where each integer is formed from measured
                qubit values according to `qubit_order` from most to least
                significant qubit, i.e. in big-endian ordering.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

        Returns:
            List of lists of amplitudes. The outer dimension indexes the
            circuit parameters and the inner dimension indexes the bitstrings.
        """
        raise NotImplementedError()


class SimulatesFinalState(metaclass=abc.ABCMeta):
    """Simulator that allows access to the simulator's final state.

    Implementors of this interface should implement the simulate_sweep
    method. This simulator only returns the state of the quantum system
    for the final step of a simulation. This simulator state may be a state
    vector, the density matrix, or another representation, depending on the
    implementation.  For simulators that also allow stepping through
    a circuit see `SimulatesIntermediateState`.
    """

    def simulate(
            self,
            program: 'cirq.Circuit',
            param_resolver: 'study.ParamResolverOrSimilarType' = None,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            initial_state: Any = None,
    ) -> 'SimulationTrialResult':
        """Simulates the supplied Circuit.

        This method returns a result which allows access to the entire
        simulator's final state.

        Args:
            program: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The  form of
                this state depends on the simulation implementation.  See
                documentation of the implementing class for details.

        Returns:
            SimulationTrialResults for the simulation. Includes the final state.
        """
        return self.simulate_sweep(
            program,
            study.ParamResolver(param_resolver),
            qubit_order,
            initial_state)[0]

    @abc.abstractmethod
    def simulate_sweep(
            self,
            program: 'cirq.Circuit',
            params: study.Sweepable,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            initial_state: Any = None,
    ) -> List['SimulationTrialResult']:
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
                this state depends on the simulation implementation.  See
                documentation of the implementing class for details.

        Returns:
            List of SimulationTrialResults for this run, one for each
            possible parameter resolver.
        """
        raise NotImplementedError()


class SimulatesIntermediateState(SimulatesFinalState, metaclass=abc.ABCMeta):
    """A SimulatesFinalState that simulates a circuit by moments.

    Whereas a general SimulatesFinalState may return the entire simulator
    state at the end of a circuit, a SimulatesIntermediateState can
    simulate stepping through the moments of a circuit.

    Implementors of this interface should implement the _simulator_iterator
    method.

    Note that state here refers to simulator state, which is not necessarily
    a state vector.
    """

    def simulate_sweep(
            self,
            program: 'cirq.Circuit',
            params: study.Sweepable,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            initial_state: Any = None,
    ) -> List['SimulationTrialResult']:
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
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            List of SimulationTrialResults for this run, one for each
            possible parameter resolver.
        """
        trial_results = []
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        for param_resolver in study.to_resolvers(params):
            all_step_results = self.simulate_moment_steps(
                program, param_resolver, qubit_order, initial_state)
            measurements = {}  # type: Dict[str, np.ndarray]
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    measurements[k] = np.array(v, dtype=np.uint8)
            trial_results.append(
                self._create_simulator_trial_result(
                    params=param_resolver,
                    measurements=measurements,
                    final_simulator_state=step_result._simulator_state()))
        return trial_results

    def simulate_moment_steps(
        self,
        circuit: circuits.Circuit,
        param_resolver: 'study.ParamResolverOrSimilarType' = None,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None
    ) -> Iterator:
        """Returns an iterator of StepResults for each moment simulated.

        If the circuit being simulated is empty, a single step result should
        be returned with the state being set to the initial state.

        Args:
            circuit: The Circuit to simulate.
            param_resolver: A ParamResolver for determining values of Symbols.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            Iterator that steps through the simulation, simulating each
            moment and returning a StepResult for each moment.
        """
        return self._simulator_iterator(
            circuit,
            study.ParamResolver(param_resolver),
            qubit_order,
            initial_state)

    @abc.abstractmethod
    def _simulator_iterator(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        qubit_order: ops.QubitOrderOrList,
        initial_state: Any,
    ) -> Iterator:
        """Iterator over StepResult from Moments of a Circuit.

        Args:
            circuit: The circuit to simulate.
            param_resolver: A ParamResolver for determining values of
                Symbols.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Yields:
            StepResults from simulating a Moment of the Circuit.
        """
        raise NotImplementedError()

    def _create_simulator_trial_result(self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: Any) \
        -> 'SimulationTrialResult':
        """This method can be overridden to creation of a trial result.

        Args:
            params: The ParamResolver for this trial.
            measurements: The measurement results for this trial.
            final_simulator_state: The final state of the simulator for the
                StepResult.

        Returns:
            The SimulationTrialResult.
        """
        return SimulationTrialResult(
            params=params,
            measurements=measurements,
            final_simulator_state=final_simulator_state)


class StepResult(metaclass=abc.ABCMeta):
    """Results of a step of a SimulatesIntermediateState.

    Attributes:
        measurements: A dictionary from measurement gate key to measurement
            results, ordered by the qubits that the measurement operates on.
    """

    def __init__(self,
                 measurements: Optional[Dict[str, List[int]]] = None) -> None:
        self.measurements = measurements or collections.defaultdict(list)

    @abc.abstractmethod
    def _simulator_state(self) -> Any:
        """Returns the simulator state of the simulator after this step.

        This method starts with an underscore to indicate that it is private.
        To access public state, see public methods on StepResult.

        The form of the simulator_state depends on the implementation of the
        simulation,see documentation for the implementing class for the form of
        details.
        """

    @abc.abstractmethod
    def sample(self,
               qubits: List[ops.Qid],
               repetitions: int = 1,
               seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None) -> np.ndarray:
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
            are wrapped as an numpy ndarray.
        """
        raise NotImplementedError()

    def sample_measurement_ops(self,
                               measurement_ops: List[ops.GateOperation],
                               repetitions: int = 1,
                               seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
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
        bounds = {}  # type: Dict[str, Tuple]
        all_qubits = []  # type: List[ops.Qid]
        meas_ops = {}
        current_index = 0
        for op in measurement_ops:
            gate = op.gate
            if not isinstance(gate, ops.MeasurementGate):
                raise ValueError('{} was not a MeasurementGate'.format(gate))
            key = protocols.measurement_key(gate)
            meas_ops[key] = gate
            if key in bounds:
                raise ValueError(
                    'Duplicate MeasurementGate with key {}'.format(key))
            bounds[key] = (current_index, current_index + len(op.qubits))
            all_qubits.extend(op.qubits)
            current_index += len(op.qubits)
        indexed_sample = self.sample(all_qubits, repetitions, seed=seed)

        results = {}
        for k, (s, e) in bounds.items():
            before_invert_mask = indexed_sample[:, s:e]
            results[k] = before_invert_mask ^ (np.logical_and(
                before_invert_mask < 2, meas_ops[k].full_invert_mask()))
        return results


@value.value_equality(unhashable=True)
class SimulationTrialResult:
    """Results of a simulation by a SimulatesFinalState.

    Unlike TrialResult these results contain the final simulator_state of the
    system. This simulator_state is dependent on the simulation implementation
    and may be, for example, the state vector or the density matrix of the
    system.

    Attributes:
        params: A ParamResolver of settings used for this result.
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are a numpy ndarray of actual boolean
            measurement results (ordered by the qubits acted on by the
            measurement gate.)
    """

    def __init__(self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: Any) -> None:
        self.params = params
        self.measurements = measurements
        self._final_simulator_state = final_simulator_state

    def __repr__(self) -> str:
        return (f'cirq.SimulationTrialResult(params={self.params!r}, '
                f'measurements={self.measurements!r}, '
                f'final_simulator_state={self._final_simulator_state!r})')

    def __str__(self) -> str:

        def bitstring(vals):
            separator = ' ' if np.max(vals) >= 10 else ''
            return separator.join(str(int(v)) for v in vals)

        results = sorted(
            [(key, bitstring(val)) for key, val in self.measurements.items()])
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
        measurements = {k: v.tolist() for k, v in
                        sorted(self.measurements.items())}
        return (self.params, measurements, self._final_simulator_state)

    @property
    def qubit_map(self) -> Dict[ops.Qid, int]:
        """A map from Qid to index used to define the ordering of the basis in
        the result.
        """
        return self._final_simulator_state.qubit_map

    def _qid_shape_(self) -> Tuple[int, ...]:
        return _qubit_map_to_shape(self.qubit_map)


def _qubit_map_to_shape(qubit_map: Dict[ops.Qid, int]) -> Tuple[int, ...]:
    qid_shape: List[int] = [-1] * len(qubit_map)
    try:
        for q, i in qubit_map.items():
            qid_shape[i] = q.dimension
    except IndexError:
        raise ValueError(
            'Invalid qubit_map. Qubit index out of bounds. Map is <{!r}>.'.
            format(qubit_map))
    if -1 in qid_shape:
        raise ValueError(
            'Invalid qubit_map. Duplicate qubit index. Map is <{!r}>.'.format(
                qubit_map))
    return tuple(qid_shape)


def _verify_unique_measurement_keys(circuit: circuits.Circuit):
    result = collections.Counter(
        key for op in ops.flatten_op_tree(iter(circuit))
        for key in protocols.measurement_keys(op))
    if result:
        duplicates = [k for k, v in result.most_common() if v > 1]
        if duplicates:
            raise ValueError('Measurement key {} repeated'.format(
                ",".join(duplicates)))
