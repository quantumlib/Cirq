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

from typing import Any, cast, Dict, Iterator, Sequence, TYPE_CHECKING

import numpy as np

from cirq import circuits, ops, study, value
from cirq.sim import simulator, wave_function

if TYPE_CHECKING:
    import cirq


class SimulatesIntermediateWaveFunction(simulator.SimulatesAmplitudes,
                                        simulator.SimulatesIntermediateState,
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

    def compute_amplitudes_sweep(
            self,
            program: 'cirq.Circuit',
            bitstrings: Sequence[int],
            params: study.Sweepable,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
    ) -> Sequence[Sequence[complex]]:
        if isinstance(bitstrings, np.ndarray) and len(bitstrings.shape) > 1:
            raise ValueError('The list of bitstrings must be input as a '
                             '1-dimensional array of ints. Got an array with '
                             f'shape {bitstrings.shape}.')

        trial_results = self.simulate_sweep(program, params, qubit_order)

        # 1-dimensional tuples don't trigger advanced Numpy array indexing
        # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        if isinstance(bitstrings, tuple):
            bitstrings = list(bitstrings)

        all_amplitudes = []
        for trial_result in trial_results:
            trial_result = cast(WaveFunctionTrialResult, trial_result)
            amplitudes = trial_result.final_state[bitstrings]
            all_amplitudes.append(amplitudes)

        return all_amplitudes


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
        self._qid_shape = simulator._qubit_map_to_shape(qubit_map)

    def _qid_shape_(self):
        return self._qid_shape

    def __repr__(self):
        return (
            "cirq.WaveFunctionSimulatorState(state_vector=np.{!r}, qubit_map="
            "{!r})".format(self.state_vector, self.qubit_map))

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

                |     | QubitA | QubitB | QubitC |
                | :-: | :----: | :----: | :----: |
                |  0  |   0    |   0    |   0    |
                |  1  |   0    |   0    |   1    |
                |  2  |   0    |   1    |   0    |
                |  3  |   0    |   1    |   1    |
                |  4  |   1    |   0    |   0    |
                |  5  |   1    |   0    |   1    |
                |  6  |   1    |   1    |   0    |
                |  7  |   1    |   1    |   1    |
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
