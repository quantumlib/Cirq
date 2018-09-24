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

"""Abstract base classes for different types of simulators."""

import abc
import collections

from typing import Dict, Iterator, List, Union

import numpy as np

from cirq import circuits, extension, ops, schedules, study
from cirq.sim import state


class RunSimulator:
    """Simulator that mimics running on quantum hardware.

    Implementors of this class should implement the _run method.
    """

    def run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver = study.ParamResolver({}),
        repetitions: int = 1,
        extensions: extension.Extensions = None,
    ) -> study.TrialResult:
        """Runs the entire supplied Circuit, mimicking the quantum hardware.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            extensions: Extensions that will be applied during the run. See
                documentation of class for details.
        Returns:
            TrialResult for a run.
        """
        return self.run_sweep(circuit,
                              [param_resolver],
                              repetitions,
                              extensions)[0]

    def run_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: study.Sweepable = study.ParamResolver({}),
        repetitions: int = 1,
        extensions: extension.Extensions = None
    ) -> List[study.TrialResult]:
        """Runs the entire supplied Circuit, mimicking the quantum hardware.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            extensions: Extensions that will be applied during the run. See
                documentation of class for details.

        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """
        circuit = (program if isinstance(program, circuits.Circuit)
                   else program.to_circuit())
        param_resolvers = study.to_resolvers(params or study.ParamResolver({}))

        trial_results = []  # type: List[study.TrialResult]
        for param_resolver in param_resolvers:
            measurements = self._run(circuit=circuit,
                                     param_resolver=param_resolver,
                                     repetitions=repetitions,
                                     extensions=extensions)
            trial_results.append(study.TrialResult(params=param_resolver,
                                                   repetitions=repetitions,
                                                   measurements=measurements))
        return trial_results

    @abc.abstractmethod
    def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int,
        extensions: extension.Extensions) -> Dict[str, List[np.ndarray]]:
        """Run a simulation, mimicking quantum hardware.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: Number of times to repeat the run.
            extensions: Extensions that will be applied during the run. See
                documentation of class for details.

        Returns:
            A dictionary from measurement key to a list of lists representing
            the results. Measurement results are a list of lists (a numpy
            ndarray), the first list corresponding to the repetition, and the
            second is the actual boolean measurement results (ordered by
            the qubits acted upon by the measurement gate.)
        """
        raise NotImplementedError()


class WaveFunctionSimulator:

    def simulate(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver = study.ParamResolver({}),
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0,
        extensions: extension.Extensions = None,
    ) -> 'SimulateTrialResult':
        """Simulates the entire supplied Circuit.

        This method returns the final wave function.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits used
                to define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state.
                Otherwise  if this is a np.ndarray it is the full initial
                state. In this case it must be the correct size, be normalized
                (an L2 norm of 1), and be safely castable to a np.complex64.
            extensions: Extensions that will be applied during the run. See
                documentation of class for details.

        Returns:
            SimulateTrialResults for the simulation. Includes the final wave
            function.
        """
        return self.simulate_sweep(circuit, [param_resolver], qubit_order,
                                   initial_state, extensions)[0]

    def simulate_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: study.Sweepable = study.ParamResolver({}),
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0,
        extensions: extension.Extensions = None
    ) -> List['SimulateTrialResult']:
        """Simulates the entire supplied Circuit.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state.
                Otherwise if this is a np.ndarray it is the full initial state.
                In this case it must be the correct size, be normalized (an L2
                norm of 1), and be safely castable to a np.complex64.
            extensions: Extensions that will be applied during the run. See
                documentation of class for details.

        Returns:
            List of SimulatorTrialResults for this run, one for each
            possible parameter resolver.
        """
        circuit = (program if isinstance(program, circuits.Circuit)
                   else program.to_circuit())
        param_resolvers = study.to_resolvers(params or study.ParamResolver({}))

        trial_results = []  # type: List[SimulateTrialResult]
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        for param_resolver in param_resolvers:
            step_result = None
            all_step_results = self.simulate_moment_steps(circuit,
                                                          param_resolver,
                                                          qubit_order,
                                                          initial_state,
                                                          extensions)
            measurements = {}  # type: Dict[str, np.ndarray]
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    measurements[k] = np.array(v, dtype=bool)
            if step_result:
                final_state = step_result.state()
            else:
                # Empty circuit, so final state should be initial state.
                num_qubits = len(qubit_order.order_for(circuit.all_qubits()))
                final_state = state.decode_initial_state(initial_state,
                                                       num_qubits)
            trial_results.append(SimulateTrialResult(
                params=param_resolver,
                measurements=measurements,
                final_state=final_state))

        return trial_results

    def simulate_moment_steps(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver = None,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0,
        extensions: extension.Extensions = None) -> Iterator['StepResult']:
        """Returns an iterator of XmonStepResults for each moment simulated.

        Args:
            circuit: The Circuit to simulate.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state.
                Otherwise if this is a np.ndarray it is the full initial state.
                In this case it must be the correct size, be normalized (an L2
                norm of 1), and be safely castable to a np.complex64.
            param_resolver: A ParamResolver for determining values of
                Symbols.
            extensions: Extensions that will be applied while trying to
                decompose the circuit's gates into XmonGates. If None, this
                uses the default of xmon_gate_ext.

        Returns:
            SimulatorIterator that steps through the simulation, simulating
            each moment and returning a XmonStepResult for each moment.
        """
        param_resolver = param_resolver or study.ParamResolver({})
        return self._simulator_iterator(circuit, param_resolver, qubit_order,
                                        initial_state, extensions)

    @abc.abstractmethod
    def _simulator_iterator(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        qubit_order: ops.QubitOrderOrList,
        initial_state: Union[int, np.ndarray],
        extensions: extension.Extensions
    ) -> Iterator['StepResult']:
        """Iterator over StepResult from Moments of a Circuit.

        Args:
            circuit: The circuit to simulate.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: The full initial state. This must be the correct
                size, be normalized (an L2 norm of 1), and be safely
                castable to a complex type handled by the simulator.

        Yields:
            StepResults from simulating a Moment of the Circuit.
        """
        raise NotImplementedError();


class StepResult:
    """Results of a step of a WaveFunctionSimulator.

    Attributes:
        qubit_map: A map from the Qubits in the Circuit to the the index
            of this qubit for a canonical ordering. This canonical ordering is
            used to define the state (see the state() method).
        measurements: A dictionary from measurement gate key to measurement
            results, ordered by the qubits that the measurement operates on.
    """

    def __init__(
        self,
        qubit_map: Dict,
        measurements: Dict[str, List[bool]]) -> None:
        self.qubit_map = qubit_map or {}
        self.measurements = measurements or collections.defaultdict(list)

    @abc.abstractmethod
    def state(self) -> np.ndarray:
        """Return the state (wave function) at this point in the computation.

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
               |   | QubitA | QubitB | QubitC |
               +---+--------+--------+--------+
               | 0 |   0    |   0    |   0    |
               | 1 |   0    |   0    |   1    |
               | 2 |   0    |   1    |   0    |
               | 3 |   0    |   1    |   1    |
               | 4 |   1    |   0    |   0    |
               | 5 |   1    |   0    |   1    |
               | 6 |   1    |   1    |   0    |
               | 7 |   1    |   1    |   1    |
               +---+--------+--------+--------+
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_state(self, state: Union[int, np.ndarray]):
        """Updates the state of the simulator to the given new state.

        Args:
            state: If this is an int, then this is the state to reset
            the stepper to, expressed as an integer of the computational basis.
            Integer to bitwise indices is little endian. Otherwise if this is
            a np.ndarray this must be the correct size and have dtype of
            np.complex64.

        Raises:
            ValueError if the state is incorrectly sized or not of the correct
            dtype.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self, qubits: List[ops.QubitId], repetitions: int = 1):
        """Samples from the wave function at this point in the computation.

        Note that this does not collapse the wave function.

        Returns:
            Measurement results with True corresponding to the |1> state.
            The outer list is for repetitions, and the inner corresponds to
            measurements ordered by the supplied qubits.
        """
        raise NotImplementedError()

    def pretty_state(self, decimals=2):
        return state.pretty_state(self.state, decimals)


class SimulateTrialResult:
    """Results of a simulation by a WaveFunctionSimulator.

    Unlike TrialResult these results contain the final state (wave function)
    of the system.

    Attributes:
        params: A ParamResolver of settings used for this result.
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are a numpy ndarray of actual boolean
            measurement results (ordered by the qubits acted on by the
            measurement gate.)
        final_state: The final state (wave function) of the system after the
            trial finishes.
    """

    def __init__(self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_state: np.ndarray) -> None:
        self.params = params
        self.measurements = measurements
        self.final_state = final_state

    def __repr__(self):
        return ('SimulateTrialResult(params={!r}, '
                'measurements={!r}, '
                'final_state={!r})').format(self.params,
                                            self.measurements,
                                            self.final_state)

    def __str__(self):
        def bitstring(vals):
            return ''.join('1' if v else '0' for v in vals)

        results = sorted(
            [(key, bitstring(val)) for key, val in self.measurements.items()])
        return ' '.join(
            ['{}={}'.format(key, val) for key, val in results])

    def pretty_state(self, decimals=2):
        return state.pretty_state(self.final_state, decimals)