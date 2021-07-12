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
"""Abstract classes for simulations which keep track of state vector."""

import abc

from typing import Any, Dict, Iterator, Sequence, TYPE_CHECKING, Tuple, Generic, TypeVar, Type

import numpy as np

from cirq import ops, study, value
from cirq.sim import simulator, state_vector, simulator_base
from cirq.sim.act_on_state_vector_args import ActOnStateVectorArgs

if TYPE_CHECKING:
    import cirq


TStateVectorStepResult = TypeVar('TStateVectorStepResult', bound='StateVectorStepResult')


class SimulatesIntermediateStateVector(
    Generic[TStateVectorStepResult],
    simulator_base.SimulatorBase[
        TStateVectorStepResult,
        'StateVectorTrialResult',
        'StateVectorSimulatorState',
        ActOnStateVectorArgs,
    ],
    simulator.SimulatesAmplitudes,
    metaclass=abc.ABCMeta,
):
    """A simulator that accesses its state vector as it does its simulation.

    Implementors of this interface should implement the _core_iterator
    method."""

    def __init__(
        self,
        *,
        dtype: Type[np.number] = np.complex64,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        split_untangled_states: bool = False,
    ):
        super().__init__(
            dtype=dtype,
            noise=noise,
            seed=seed,
            split_untangled_states=split_untangled_states,
        )

    def _create_simulator_trial_result(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'StateVectorSimulatorState',
    ) -> 'StateVectorTrialResult':
        return StateVectorTrialResult(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

    def compute_amplitudes_sweep_iter(
        self,
        program: 'cirq.Circuit',
        bitstrings: Sequence[int],
        params: study.Sweepable,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
    ) -> Iterator[Sequence[complex]]:
        if isinstance(bitstrings, np.ndarray) and len(bitstrings.shape) > 1:
            raise ValueError(
                'The list of bitstrings must be input as a '
                '1-dimensional array of ints. Got an array with '
                f'shape {bitstrings.shape}.'
            )

        # 1-dimensional tuples don't trigger advanced Numpy array indexing
        # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        if isinstance(bitstrings, tuple):
            bitstrings = list(bitstrings)

        trial_result_iter = self.simulate_sweep_iter(program, params, qubit_order)

        yield from (
            trial_result.final_state_vector[bitstrings] for trial_result in trial_result_iter
        )


class StateVectorStepResult(
    simulator_base.StepResultBase['StateVectorSimulatorState', 'cirq.ActOnStateVectorArgs'],
    metaclass=abc.ABCMeta,
):
    @abc.abstractmethod
    def _simulator_state(self) -> 'StateVectorSimulatorState':
        """Returns the simulator_state of the simulator after this step.

        The form of the simulator_state depends on the implementation of the
        simulation,see documentation for the implementing class for the form of
        details.
        """
        raise NotImplementedError()


@value.value_equality(unhashable=True)
class StateVectorSimulatorState:
    def __init__(self, state_vector: np.ndarray, qubit_map: Dict[ops.Qid, int]) -> None:
        self.state_vector = state_vector
        self.qubit_map = qubit_map
        self._qid_shape = simulator._qubit_map_to_shape(qubit_map)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def __repr__(self) -> str:
        return (
            'cirq.StateVectorSimulatorState('
            f'state_vector=np.{self.state_vector!r}, '
            f'qubit_map={self.qubit_map!r})'
        )

    def _value_equality_values_(self) -> Any:
        return (self.state_vector.tolist(), self.qubit_map)


@value.value_equality(unhashable=True)
class StateVectorTrialResult(state_vector.StateVectorMixin, simulator.SimulationTrialResult):
    """A `SimulationTrialResult` that includes the `StateVectorMixin` methods.

    Attributes:
        final_state_vector: The final state vector for the system.
    """

    def __init__(
        self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state: StateVectorSimulatorState,
    ) -> None:
        super().__init__(
            params=params,
            measurements=measurements,
            final_simulator_state=final_simulator_state,
            qubit_map=final_simulator_state.qubit_map,
        )
        self.final_state_vector = final_simulator_state.state_vector

    def state_vector(self):
        """Return the state vector at the end of the computation.

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
        return self._final_simulator_state.state_vector.copy()

    def _value_equality_values_(self):
        measurements = {k: v.tolist() for k, v in sorted(self.measurements.items())}
        return (self.params, measurements, self._final_simulator_state)

    def __str__(self) -> str:
        samples = super().__str__()
        final = self.state_vector()
        if len([1 for e in final if abs(e) > 0.001]) < 16:
            state_vector = self.dirac_notation(3)
        else:
            state_vector = str(final)
        return f'measurements: {samples}\noutput vector: {state_vector}'

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('StateVectorTrialResult(...)')
        else:
            p.text(str(self))

    def __repr__(self) -> str:
        return (
            f'cirq.StateVectorTrialResult(params={self.params!r}, '
            f'measurements={self.measurements!r}, '
            f'final_simulator_state={self._final_simulator_state!r})'
        )
