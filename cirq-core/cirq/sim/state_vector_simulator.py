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
from typing import Any, Dict, Iterator, Sequence, Type, TYPE_CHECKING, Generic, TypeVar

import numpy as np

from cirq import _compat, ops, value, qis
from cirq.sim import simulator, state_vector, simulator_base
from cirq.protocols import qid_shape

if TYPE_CHECKING:
    import cirq


TStateVectorStepResult = TypeVar('TStateVectorStepResult', bound='StateVectorStepResult')


class SimulatesIntermediateStateVector(
    Generic[TStateVectorStepResult],
    simulator_base.SimulatorBase[
        TStateVectorStepResult, 'cirq.StateVectorTrialResult', 'cirq.StateVectorSimulationState'
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
        dtype: Type[np.complexfloating] = np.complex64,
        noise: 'cirq.NOISE_MODEL_LIKE' = None,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        split_untangled_states: bool = False,
    ):
        super().__init__(
            dtype=dtype, noise=noise, seed=seed, split_untangled_states=split_untangled_states
        )

    def _create_simulator_trial_result(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[cirq.StateVectorSimulationState]',
    ) -> 'cirq.StateVectorTrialResult':
        return StateVectorTrialResult(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

    def compute_amplitudes_sweep_iter(
        self,
        program: 'cirq.AbstractCircuit',
        bitstrings: Sequence[int],
        params: 'cirq.Sweepable',
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
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
            trial_result.final_state_vector[bitstrings].tolist()
            for trial_result in trial_result_iter
        )


class StateVectorStepResult(
    simulator_base.StepResultBase['cirq.StateVectorSimulationState'], metaclass=abc.ABCMeta
):
    pass


@value.value_equality(unhashable=True)
class StateVectorTrialResult(
    state_vector.StateVectorMixin,
    simulator_base.SimulationTrialResultBase['cirq.StateVectorSimulationState'],
):
    """A `SimulationTrialResult` that includes the `StateVectorMixin` methods.

    Attributes:
        final_state_vector: The final state vector for the system.
    """

    def __init__(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[cirq.StateVectorSimulationState]',
    ) -> None:
        super().__init__(
            params=params,
            measurements=measurements,
            final_simulator_state=final_simulator_state,
            qubit_map=final_simulator_state.qubit_map,
        )

    @_compat.cached_property
    def final_state_vector(self) -> np.ndarray:
        return self._get_merged_sim_state().target_tensor.reshape(-1)

    def state_vector(self, copy: bool = False) -> np.ndarray:
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

        Args:
            copy: If True, the returned state vector will be a copy of that
            stored by the object. This is potentially expensive for large
            state vectors, but prevents mutation of the object state, e.g. for
            operating on intermediate states of a circuit.
            Defaults to False.
        """
        return self.final_state_vector.copy() if copy else self.final_state_vector

    def _value_equality_values_(self):
        measurements = {k: v.tolist() for k, v in sorted(self.measurements.items())}
        return self.params, measurements, self.qubit_map, self.final_state_vector.tolist()

    def __str__(self) -> str:
        samples = super().__str__()
        ret = f'measurements: {samples}'
        for substate in self._get_substates():
            final = substate.target_tensor
            shape = final.shape
            size = np.prod(shape, dtype=np.int64)
            final = final.reshape(size)
            if len([1 for e in final if abs(e) > 0.001]) < 16:
                state_vector = qis.dirac_notation(final, 3, qid_shape(substate.qubits))
            else:
                state_vector = str(final)
            label = f'qubits: {substate.qubits}' if substate.qubits else 'phase:'
            ret += f'\n\n{label}\noutput vector: {state_vector}'
        return ret

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('StateVectorTrialResult(...)')
        else:
            p.text(str(self))

    def __repr__(self) -> str:
        return (
            'cirq.StateVectorTrialResult('
            f'params={self.params!r}, measurements={_compat.proper_repr(self.measurements)}, '
            f'final_simulator_state={self._final_simulator_state!r})'
        )
