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

"""Abstract classes for simulations which keep track of wave functions."""

import abc

from typing import Any, Dict, Iterator, Hashable, List, Optional, Union

import numpy as np

from cirq import circuits, ops, schedules, study, value
from cirq.sim import simulator, wave_function


class SimulatesIntermediateWaveFunction(simulator.SimulatesIntermediateState,
                                        metaclass=abc.ABCMeta):
    """A simulator that accesses its wave function as it does its simulation.

    Implementors of this interface should implement the _simulator_iterator
    method."""

    @abc.abstractmethod
    def _simulator_iterator(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        qubit_order: ops.QubitOrderOrList,
        initial_state: np.ndarray,
    ) -> Iterator:
        """Iterator over WaveFunctionStepResult from Moments of a Circuit.

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
            WaveFunctionStepResult from simulating a Moment of the Circuit.
        """
        raise NotImplementedError()

    def _create_simulator_trial_result(self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'WaveFunctionSimulatorState') \
        -> 'WaveFunctionTrialResult':
        return WaveFunctionTrialResult(
            params=params,
            measurements=measurements,
            final_simulator_state=final_simulator_state)

    def compute_displays(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        param_resolver: study.ParamResolver = study.ParamResolver({}),
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0,
    ) -> study.ComputeDisplaysResult:
        """Computes displays in the supplied Circuit or Schedule.

        Args:
            program: The circuit or schedule to simulate.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits used
                to define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state. Otherwise  if this
                is a np.ndarray it is the full initial state. In this case it
                must be the correct size, be normalized (an L2 norm of 1), and
                be safely castable to an appropriate dtype for the simulator.

        Returns:
            ComputeDisplaysResult for the simulation.
        """
        return self.compute_displays_sweep(
            program, [param_resolver], qubit_order, initial_state)[0]

    def compute_displays_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: Optional[study.Sweepable] = None,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0,
    ) -> List[study.ComputeDisplaysResult]:
        """Computes displays in the supplied Circuit or Schedule.

        In contrast to `compute_displays`, this allows for sweeping
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
            List of ComputeDisplaysResults for this run, one for each
            possible parameter resolver.
        """
        circuit = (program if isinstance(program, circuits.Circuit)
                   else program.to_circuit())
        param_resolvers = study.to_resolvers(params or study.ParamResolver({}))
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        qubits = qubit_order.order_for(circuit.all_qubits())

        compute_displays_results = []  # type: List[study.ComputeDisplaysResult]
        for param_resolver in param_resolvers:
            display_values = {}  # type: Dict[Hashable, Any]

            # Compute the displays in the first Moment
            moment = circuit[0]
            state = wave_function.to_valid_state_vector(
                initial_state, num_qubits=len(qubits))
            qubit_map = {q: i for i, q in enumerate(qubits)}
            _enter_moment_display_values_into_dictionary(
                display_values, moment, state, qubit_order, qubit_map)

            # Compute the displays in the rest of the Moments
            all_step_results = self.simulate_moment_steps(
                circuit,
                param_resolver,
                qubit_order,
                initial_state)
            for step_result, moment in zip(all_step_results, circuit[1:]):
                _enter_moment_display_values_into_dictionary(
                    display_values,
                    moment,
                    step_result.state_vector(),
                    qubit_order,
                    step_result.qubit_map)

            compute_displays_results.append(study.ComputeDisplaysResult(
                params=param_resolver,
                display_values=display_values))

        return compute_displays_results


def _enter_moment_display_values_into_dictionary(
    display_values: Dict,
    moment: ops.Moment,
    state: np.ndarray,
    qubit_order: ops.QubitOrder,
    qubit_map: Dict[ops.Qid, int]):
    for op in moment:
        if isinstance(op, ops.WaveFunctionDisplay):
            display_values[op.key] = (
                op.value_derived_from_wavefunction(state, qubit_map))
        elif isinstance(op, ops.SamplesDisplay):
            display_values[op.key] = _compute_samples_display_value(
                op, state, qubit_order, qubit_map)


def _compute_samples_display_value(display: ops.SamplesDisplay,
    state: np.ndarray,
    qubit_order: ops.QubitOrder,
    qubit_map: Dict[ops.Qid, int]):
    basis_change_circuit = circuits.Circuit.from_ops(
        display.measurement_basis_change())
    modified_state = basis_change_circuit.final_wavefunction(
        state,
        qubit_order=qubit_order,
        qubits_that_should_be_present=qubit_map.keys())
    indices = [qubit_map[qubit] for qubit in display.qubits]
    samples = wave_function.sample_state_vector(
        modified_state, indices, display.num_samples)
    return display.value_derived_from_samples(samples)


class WaveFunctionStepResult(simulator.StepResult, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _simulator_state(self) -> 'WaveFunctionSimulatorState':
        """Returns the simulator_state of the simulator after this step.

        The form of the simulator_state depends on the implementation of the
        simulation,see documentation for the implementing class for the form of
        details.
        """
        raise NotImplementedError()


@value.value_equality(unhashable=True)
class WaveFunctionSimulatorState:

    def __init__(self,
        state_vector: np.ndarray,
        qubit_map: Dict[ops.Qid, int]):
        self.state_vector = state_vector
        self.qubit_map = qubit_map

    def __repr__(self):
        return (
            'cirq.WaveFunctionSimulatorState(state_vector={!r}, qubit_map={!r})'
                .format(self.state_vector, self.qubit_map))

    def _value_equality_values_(self):
        return (self.state_vector.tolist(), self.qubit_map)


@value.value_equality(unhashable=True)
class WaveFunctionTrialResult(wave_function.StateVectorMixin,
                              simulator.SimulationTrialResult):
    """A `SimulationTrialResult` that includes the `StateVectorMixin` methods.

    Attributes:
        final_state: The final wave function of the system.
    """

    def __init__(self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: WaveFunctionSimulatorState) -> None:
        super().__init__(params=params,
                         measurements=measurements,
                         final_simulator_state=final_simulator_state,
                         qubit_map=final_simulator_state.qubit_map)
        self.final_state = final_simulator_state.state_vector

    def state_vector(self):
        """Return the wave function at the end of the computation.

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

                    | QubitA | QubitB | QubitC
                :-: | :----: | :----: | :----:
                 0  |   0    |   0    |   0
                 1  |   0    |   0    |   1
                 2  |   0    |   1    |   0
                 3  |   0    |   1    |   1
                 4  |   1    |   0    |   0
                 5  |   1    |   0    |   1
                 6  |   1    |   1    |   0
                 7  |   1    |   1    |   1
        """
        return self._final_simulator_state.state_vector

    def _value_equality_values_(self):
        measurements = {k: v.tolist() for k, v in
                        sorted(self.measurements.items())}
        return (self.params, measurements, self._final_simulator_state)

    def __str__(self):
        samples = super().__str__()
        final = self.state_vector()
        if len([1 for e in final if abs(e) > 0.001]) < 16:
            wave = self.dirac_notation(3)
        else:
            wave = str(final)

        return 'measurements: {}\noutput vector: {}'.format(samples, wave)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('WaveFunctionTrialResult(...)')
        else:
            p.text(str(self))

    def __repr__(self):
        return ('cirq.WaveFunctionTrialResult(params={!r}, '
                'measurements={!r}, '
                'final_simulator_state={!r})').format(
                    self.params, self.measurements, self._final_simulator_state)
