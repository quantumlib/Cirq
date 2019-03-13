# Copyright 2019 The Cirq Developers
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

    SimulatesFinalState: allows access to the final state of the simulation.

    SimulatesIntermediateState: allows for access to the state of the simulation
        as the simulation iterates through the moments of a cirq.
"""

from typing import (
    Any, Dict, Hashable, Iterator, List, Tuple, Union, Optional)

import abc
import collections

import numpy as np

from cirq import circuits, ops, schedules, study, value


class SimulatesSamples(metaclass=abc.ABCMeta):
    """Simulator that mimics running on quantum hardware.

    Implementors of this interface should implement the _run method.
    """

    def run(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        param_resolver: Optional[study.ParamResolver] = None,
        repetitions: int = 1,
    ) -> study.TrialResult:
        """Runs the supplied Circuit or Schedule, mimicking quantum hardware.

        Args:
            program: The circuit or schedule to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.

        Returns:
            TrialResult for a run.
        """
        return self.run_sweep(program,
                              [param_resolver or study.ParamResolver({})],
                              repetitions)[0]

    def run_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: study.Sweepable,
        repetitions: int = 1,
    ) -> List[study.TrialResult]:
        """Runs the supplied Circuit or Schedule, mimicking quantum hardware.

        In contrast to run, this allows for sweeping over different parameter
        values.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.

        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """
        circuit = (program if isinstance(program, circuits.Circuit)
                   else program.to_circuit())
        param_resolvers = study.to_resolvers(params)

        trial_results = []  # type: List[study.TrialResult]
        for param_resolver in param_resolvers:
            measurements = self._run(circuit=circuit,
                                     param_resolver=param_resolver,
                                     repetitions=repetitions)
            trial_results.append(study.TrialResult(params=param_resolver,
                                                   repetitions=repetitions,
                                                   measurements=measurements))
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

    def compute_samples_displays(
            self,
            program: Union[circuits.Circuit, schedules.Schedule],
            param_resolver: Optional[study.ParamResolver] = None,
    ) -> study.ComputeDisplaysResult:
        """Computes SamplesDisplays in the supplied Circuit or Schedule.

        Args:
            program: The circuit or schedule to simulate.
            param_resolver: Parameters to run with the program.

        Returns:
            ComputeDisplaysResult for the simulation.
        """
        return self.compute_samples_displays_sweep(
            program, [param_resolver or study.ParamResolver({})])[0]

    def compute_samples_displays_sweep(
            self,
            program: Union[circuits.Circuit, schedules.Schedule],
            params: Optional[study.Sweepable] = None
    ) -> List[study.ComputeDisplaysResult]:
        """Computes SamplesDisplays in the supplied Circuit or Schedule.

        In contrast to `compute_displays`, this allows for sweeping
        over different parameter values.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.

        Returns:
            List of ComputeDisplaysResults for this run, one for each
            possible parameter resolver.
        """
        circuit = (program if isinstance(program, circuits.Circuit)
                   else program.to_circuit())
        param_resolvers = study.to_resolvers(params or study.ParamResolver({}))

        compute_displays_results = []  # type: List[study.ComputeDisplaysResult]
        for param_resolver in param_resolvers:
            display_values = {}  # type: Dict[Hashable, Any]
            preceding_circuit = circuits.Circuit()
            for i, moment in enumerate(circuit):
                displays = (op for op in moment
                            if isinstance(op, ops.SamplesDisplay))
                for display in displays:
                    measurement_key = str(display.key)
                    measurement_circuit = circuits.Circuit.from_ops(
                        display.measurement_basis_change(),
                        ops.measure(*display.qubits,
                                    key=measurement_key)
                    )
                    measurements = self._run(
                        preceding_circuit + measurement_circuit,
                        param_resolver,
                        display.num_samples)
                    display_values[display.key] = (
                        display.value_derived_from_samples(
                            measurements[measurement_key]))
                preceding_circuit.append(circuit[i])
            compute_displays_results.append(study.ComputeDisplaysResult(
                params=param_resolver,
                display_values=display_values))

        return compute_displays_results


class SimulatesFinalState(metaclass=abc.ABCMeta):
    """Simulator that allows access to a quantum computer's final state.

    Implementors of this interface should implement the simulate_sweep
    method. This simulator only returns the state of the quantum system
    for the final step of a simulation. This simulator state may be a wave
    function, the density matrix, or another representation, depending on the
    implementation.  For simulators that also allow stepping through
    a circuit see `SimulatesIntermediateState`.
    """

    def simulate(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        param_resolver: Optional[study.ParamResolver] = None,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> 'SimulationTrialResult':
        """Simulates the supplied Circuit or Schedule.

        This method returns a result which allows access to the entire
        wave function.

        Args:
            program: The circuit or schedule to simulate.
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
        return self.simulate_sweep(program,
                                   [param_resolver or study.ParamResolver({})],
                                   qubit_order,
                                   initial_state)[0]

    @abc.abstractmethod
    def simulate_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: study.Sweepable,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> List['SimulationTrialResult']:
        """Simulates the supplied Circuit or Schedule.

        This method returns a result which allows access to the entire
        wave function. In contrast to simulate, this allows for sweeping
        over different parameter values.

        Args:
            program: The circuit or schedule to simulate.
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

    Whereas a general SimulatesFinalState may return the entire wave
    function at the end of a circuit, a SimulatesIntermediateState can
    simulate stepping through the moments of a circuit.

    Implementors of this interface should implement the _simulator_iterator
    method.
    """

    def simulate_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: study.Sweepable,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> List['SimulationTrialResult']:
        """Simulates the supplied Circuit or Schedule.

        This method returns a result which allows access to the entire
        wave function. In contrast to simulate, this allows for sweeping
        over different parameter values.

        Args:
            program: The circuit or schedule to simulate.
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
        circuit = (program if isinstance(program, circuits.Circuit)
                   else program.to_circuit())
        param_resolvers = study.to_resolvers(params)

        trial_results = []
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        for param_resolver in param_resolvers:
            all_step_results = self.simulate_moment_steps(circuit,
                                                          param_resolver,
                                                          qubit_order,
                                                          initial_state)
            measurements = {}  # type: Dict[str, np.ndarray]
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    measurements[k] = np.array(v, dtype=bool)
            trial_results.append(
                self._create_simulator_trial_result(
                    params=param_resolver,
                    measurements=measurements,
                    final_simulator_state=step_result.simulator_state()))
        return trial_results

    def simulate_moment_steps(
        self,
        circuit: circuits.Circuit,
        param_resolver: Optional[study.ParamResolver] = None,
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
        param_resolver = param_resolver or study.ParamResolver({})
        return self._simulator_iterator(circuit, param_resolver, qubit_order,
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
                 measurements: Optional[Dict[str, List[bool]]] = None) -> None:
        self.measurements = measurements or collections.defaultdict(list)

    @abc.abstractmethod
    def simulator_state(self) -> Any:
        """Returns the simulator_state of the simulator after this step.

        The form of the simulator_state depends on the implementation of the
        simulation,see documentation for the implementing class for the form of
        details.
        """

    @abc.abstractmethod
    def sample(self,
               qubits: List[ops.QubitId],
               repetitions: int = 1) -> np.ndarray:
        """Samples from the system at this point in the computation.

        Note that this does not collapse the wave function.

        Args:
            qubits: The qubits to be sampled in an order that influence the
                returned measurement results.
            repetitions: The number of samples to take.

        Returns:
            Measurement results with True corresponding to the ``|1⟩`` state.
            The outer list is for repetitions, and the inner corresponds to
            measurements ordered by the supplied qubits. These lists
            are wrapped as an numpy ndarray.
        """
        raise NotImplementedError()

    def sample_measurement_ops(
            self,
            measurement_ops: List[ops.GateOperation],
            repetitions: int = 1) -> Dict[str, np.ndarray]:
        """Samples from the system at this point in the computation.

        Note that this does not collapse the wave function.

        In contrast to `sample` which samples qubits, this takes a list of
        `cirq.GateOperation` instances whose gates are `cirq.MeasurementGate`
        instances and then returns a mapping from the key in the measurement
        gate to the resulting bit strings. Different measurement operations must
        not act on the same qubits.

        Args:
            measurement_ops: `GateOperation` instances whose gates are
                `MeasurementGate` instances to be sampled form.
            repetitions: The number of samples to take.

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
        all_qubits = []  # type: List[ops.QubitId]
        current_index = 0
        for op in measurement_ops:
            gate = op.gate
            if not isinstance(gate, ops.MeasurementGate):
                raise ValueError('{} was not a MeasurementGate'.format(gate))
            if gate.key in bounds:
                raise ValueError(
                    'Duplicate MeasurementGate with key {}'.format(gate.key))
            bounds[gate.key] = (current_index, current_index + len(op.qubits))
            all_qubits.extend(op.qubits)
            current_index += len(op.qubits)
        indexed_sample = self.sample(all_qubits, repetitions)
        return {k: np.array([x[s:e] for x in indexed_sample]) for k, (s, e) in
                bounds.items()}


@value.value_equality(unhashable=True)
class SimulationTrialResult:
    """Results of a simulation by a SimulatesFinalState.

    Unlike TrialResult these results contain the final simulator_state of the
    system. This simulator_state is dependent on the simulation implementation
    and may be, for example, the wave function of the system or the density
    matrix of the system.

    Attributes:
        params: A ParamResolver of settings used for this result.
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are a numpy ndarray of actual boolean
            measurement results (ordered by the qubits acted on by the
            measurement gate.)
        final_simulator_state: The final simulator state of the system after the
            trial finishes.
    """

    def __init__(self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: Any) -> None:
        self.params = params
        self.measurements = measurements
        self.final_simulator_state = final_simulator_state

    def __repr__(self):
        return (
            'cirq.SimulationTrialResult(params={!r}, '
            'measurements={!r}, '
            'final_simulator_state={!r})').format(
                self.params, self.measurements, self.final_simulator_state)

    def __str__(self):
        def bitstring(vals):
            return ''.join('1' if v else '0' for v in vals)

        results = sorted(
            [(key, bitstring(val)) for key, val in self.measurements.items()])
        return ' '.join(
            ['{}={}'.format(key, val) for key, val in results])

    def _value_equality_values_(self):
        measurements = {k: v.tolist() for k, v in
                        sorted(self.measurements.items())}
        return (self.params, measurements, self.final_simulator_state)