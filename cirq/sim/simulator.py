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

Simulator types include
    SimulatesSamples: mimics the interface of quantum hardware.
    SimulatesFinalWaveFunction: allows access to the wave function.
"""

import abc
import collections

from typing import Dict, Iterable, Iterator, List, Tuple, Union, Optional

import numpy as np

from cirq import circuits, ops, schedules, study, value
from cirq.sim import wave_function


class SimulatesSamples:
    """Simulator that mimics running on quantum hardware.

    Implementors of this interface should implement the _run method.
    """

    def run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver = study.ParamResolver({}),
        repetitions: int = 1,
    ) -> study.TrialResult:
        """Runs the entire supplied Circuit, mimicking the quantum hardware.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.

        Returns:
            TrialResult for a run.
        """
        return self.run_sweep(circuit, [param_resolver], repetitions)[0]

    def run_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: study.Sweepable = study.ParamResolver({}),
        repetitions: int = 1,
    ) -> List[study.TrialResult]:
        """Runs the entire supplied Circuit, mimicking the quantum hardware.

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
        param_resolvers = study.to_resolvers(params or study.ParamResolver({}))

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


class SimulatesFinalWaveFunction:
    """Simulator that allows access to a quantum computer's wavefunction.

    Implementors of this interface should implement the simulate_sweep
    method. This simulator only returns the wave function for the final
    step of a simulation. For simulators that also allow stepping through
    a circuit see `SimulatesIntermediateWaveFunction`.
    """

    def simulate(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver = study.ParamResolver({}),
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0,
    ) -> 'SimulationTrialResult':
        """Simulates the entire supplied Circuit.

        This method returns a result which allows access to the entire
        wave function.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits used
                to define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state. Otherwise  if this
                is a np.ndarray it is the full initial state. In this case it
                must be the correct size, be normalized (an L2 norm of 1), and
                be safely castable to an appropriate dtype for the simulator.

        Returns:
            SimulateTrialResults for the simulation. Includes the final wave
            function.
        """
        return self.simulate_sweep(circuit, [param_resolver], qubit_order,
                                   initial_state)[0]

    @abc.abstractmethod
    def simulate_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: study.Sweepable = study.ParamResolver({}),
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0,
    ) -> List['SimulationTrialResult']:
        """Simulates the entire supplied Circuit.

        This method returns a result which allows access to the entire
        wave function. In contrast to simulate, this allows for sweeping
        over different parameter values.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state.
                Otherwise if this is a np.ndarray it is the full initial state.
                In this case it must be the correct size, be normalized (an L2
                norm of 1), and  be safely castable to an appropriate
                dtype for the simulator.

        Returns:
            List of SimulatorTrialResults for this run, one for each
            possible parameter resolver.
        """
        raise NotImplementedError()


@value.value_equality(unhashable=True)
class SimulationTrialResult:
    """Results of a simulation by a SimulatesFinalWaveFunction.

    Unlike TrialResult these results contain the final state (wave function)
    of the system.

    Attributes:
        params: A ParamResolver of settings used for this result.
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are a numpy ndarray of actual boolean
            measurement results (ordered by the qubits acted on by the
            measurement gate.)
        final_state: The final state (wave function) of the system after the
            trial finishes. The state is returned in the computational basis
            with these basis states defined by the qubit ordering of the
            simulation. In particular the qubit ordering can be used to produce
            a list of qubits, and these qubits can the be associated with their
            index in the list.  This mapping of qubit to index is then
            translated into binary vectors where the last qubit is the
            1s bit of the index, the second-to-last is the 2s bit of the index,
            and so forth (i.e. big endian ordering). Example:
                 qubit ordering: [QubitA, QubitB, QubitC]
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

    def __init__(self,
                 params: study.ParamResolver,
                 measurements: Dict[str, np.ndarray],
                 final_state: np.ndarray) -> None:
        self.params = params
        self.measurements = measurements
        self.final_state = final_state

    def __repr__(self):
        return ('SimulationTrialResult(params={!r}, '
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

    def dirac_notation(self, decimals: int = 2) -> str:
        """Returns the wavefunction as a string in Dirac notation.

        Args:
            decimals: How many decimals to include in the pretty print.

        Returns:
            A pretty string consisting of a sum of computational basis kets
            and non-zero floats of the specified accuracy."""
        return wave_function.dirac_notation(self.final_state, decimals)

    def density_matrix(self, indices: Iterable[int] = None) -> np.ndarray:
        """Returns the density matrix of the wavefunction.

        Calculate the density matrix for the system on the given qubit
        indices, with the qubits not in indices that are present in
        self.final_state traced out. If indices is None the full density
        matrix for self.final_state is returned, given self.final_state
        follows the standard Kronecker convention of numpy.kron.

        For example:
            self.final_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)],
                dtype=np.complex64)
            indices = None
            gives us \rho = \begin{bmatrix}
                                0.5 & 0.5
                                0.5 & 0.5
                            \end{bmatrix}

        Args:
            indices: list containing indices for qubits that you would like
                to include in the density matrix (i.e.) qubits that WON'T
                be traced out.

        Returns:
            A numpy array representing the density matrix.

        Raises:
            ValueError: if the size of the state represents more than 25 qubits.
            IndexError: if the indices are out of range for the number of qubits
                corresponding to the state.
        """
        return wave_function.density_matrix_from_state_vector(
            self.final_state, indices)

    def bloch_vector(self, index: int) -> np.ndarray:
        """Returns the bloch vector of a qubit.

        Calculates the bloch vector of the qubit at index
        in the wavefunction given by self.state. Given that self.state
        follows the standard Kronecker convention of numpy.kron.

        Args:
            index: index of qubit who's bloch vector we want to find.

        Returns:
            A length 3 numpy array representing the qubit's bloch vector.

        Raises:
            ValueError: if the size of the state represents more than 25 qubits.
            IndexError: if index is out of range for the number of qubits
                corresponding to the state.
        """
        return wave_function.bloch_vector_from_state_vector(
            self.final_state, index)

    def _value_equality_values_(self):
        measurements = {k: v.tolist() for k, v in
                        sorted(self.measurements.items())}
        return (SimulationTrialResult, self.params, measurements,
                self.final_state.tolist())


class SimulatesIntermediateWaveFunction(SimulatesFinalWaveFunction):
    """A SimulatesFinalWaveFunction that simulates a circuit by moments.

    Whereas a general SimulatesFinalWaveFunction may return the entire wave
    function at the end of a circuit, a SimulatesIntermediateWaveFunction can
    simulate stepping through the moments of a circuit.

    Implementors of this interface should implement the _simulator_iterator
    method.
    """

    def simulate_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: study.Sweepable = study.ParamResolver({}),
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0,
    ) -> List['SimulationTrialResult']:
        """Simulates the entire supplied Circuit.

        This method returns a result which allows access to the entire
        wave function. In contrast to simulate, this allows for sweeping
        over different parameter values.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state.
                Otherwise if this is a np.ndarray it is the full initial state.
                In this case it must be the correct size, be normalized (an L2
                norm of 1), and  be safely castable to an appropriate
                dtype for the simulator.

        Returns:
            List of SimulatorTrialResults for this run, one for each
            possible parameter resolver.
        """
        circuit = (program if isinstance(program, circuits.Circuit)
                   else program.to_circuit())
        param_resolvers = study.to_resolvers(params or study.ParamResolver({}))

        trial_results = []  # type: List[SimulationTrialResult]
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        for param_resolver in param_resolvers:
            step_result = None
            all_step_results = self.simulate_moment_steps(circuit,
                                                          param_resolver,
                                                          qubit_order,
                                                          initial_state)
            measurements = {}  # type: Dict[str, np.ndarray]
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    measurements[k] = np.array(v, dtype=bool)
            if step_result:
                final_state = step_result.state()
            else:
                # Empty circuit, so final state should be initial state.
                num_qubits = len(qubit_order.order_for(circuit.all_qubits()))
                final_state = wave_function.to_valid_state_vector(initial_state,
                                                                  num_qubits)
            trial_results.append(SimulationTrialResult(
                params=param_resolver,
                measurements=measurements,
                final_state=final_state))

        return trial_results

    def simulate_moment_steps(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver = None,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0
    ) -> Iterator['StepResult']:
        """Returns an iterator of StepResults for each moment simulated.

        Args:
            circuit: The Circuit to simulate.
            param_resolver: A ParamResolver for determining values of Symbols.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state. Otherwise if this is
                a np.ndarray it is the full initial state. In this case it must
                be the correct size, be normalized (an L2 norm of 1), and
                be safely castable to an appropriate dtype for the simulator.

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
        initial_state: Union[int, np.ndarray],
    ) -> Iterator['StepResult']:
        """Iterator over StepResult from Moments of a Circuit.

        Args:
            circuit: The circuit to simulate.
            param_resolver: A ParamResolver for determining values of
                Symbols.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: The full initial state. This must be the correct
                size, be normalized (an L2 norm of 1), and be safely
                castable to a complex type handled by the simulator.

        Yields:
            StepResults from simulating a Moment of the Circuit.
        """
        raise NotImplementedError()


class StepResult:
    """Results of a step of a SimulatesFinalWaveFunction.

    Attributes:
        qubit_map: A map from the Qubits in the Circuit to the the index
            of this qubit for a canonical ordering. This canonical ordering is
            used to define the state (see the state() method).
        measurements: A dictionary from measurement gate key to measurement
            results, ordered by the qubits that the measurement operates on.
    """

    def __init__(self,
                 qubit_map: Optional[Dict],
                 measurements: Optional[Dict[str, List[bool]]]) -> None:
        self.qubit_map = qubit_map or {}
        self.measurements = measurements or collections.defaultdict(list)

    @abc.abstractmethod
    def state(self) -> np.ndarray:
        """Return the state (wave function) at this point in the computation.

        The state is returned in the computational basis with these basis
        states defined by the `qubit_map`. In particular the value in the
        `qubit_map` is the index of the qubit, and these are translated into
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
    def set_state(self, state: Union[int, np.ndarray]) -> None:
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
    def sample(self,
               qubits: List[ops.QubitId],
               repetitions: int = 1) -> np.ndarray:
        """Samples from the wave function at this point in the computation.

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
        """Samples from the wave function at this point in the computation.

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

    def dirac_notation(self, decimals: int = 2) -> str:
        """Returns the wavefunction as a string in Dirac notation.

        Args:
            decimals: How many decimals to include in the pretty print.

        Returns:
            A pretty string consisting of a sum of computational basis kets
            and non-zero floats of the specified accuracy."""
        return wave_function.dirac_notation(self.state(), decimals)

    def density_matrix(self, indices: Iterable[int] = None) -> np.ndarray:
        """Returns the density matrix of the wavefunction.

        Calculate the density matrix for the system on the given qubit
        indices, with the qubits not in indices that are present in self.state
        traced out. If indices is None the full density matrix for self.state
        is returned, given self.state follows standard Kronecker convention
        of numpy.kron.

        For example:
            self.state = np.array([1/np.sqrt(2), 1/np.sqrt(2)],
                dtype=np.complex64)
            indices = None
            gives us \rho = \begin{bmatrix}
                                0.5 & 0.5
                                0.5 & 0.5
                            \end{bmatrix}

        Args:
            indices: list containing indices for qubits that you would like
                to include in the density matrix (i.e.) qubits that WON'T
                be traced out.

        Returns:
            A numpy array representing the density matrix.

        Raises:
            ValueError: if the size of the state represents more than 25 qubits.
            IndexError: if the indices are out of range for the number of qubits
                corresponding to the state.
        """
        return wave_function.density_matrix_from_state_vector(
            self.state(), indices)

    def bloch_vector(self, index: int) -> np.ndarray:
        """Returns the bloch vector of a qubit.

        Calculates the bloch vector of the qubit at index
        in the wavefunction given by self.state. Given that self.state
        follows the standard Kronecker convention of numpy.kron.

        Args:
            index: index of qubit who's bloch vector we want to find.

        Returns:
            A length 3 numpy array representing the qubit's bloch vector.

        Raises:
            ValueError: if the size of the state represents more than 25 qubits.
            IndexError: if index is out of range for the number of qubits
                corresponding to the state.
        """
        return wave_function.bloch_vector_from_state_vector(
            self.state(), index)
