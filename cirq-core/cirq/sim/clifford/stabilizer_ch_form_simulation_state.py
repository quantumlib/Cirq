# Copyright 2020 The Cirq Developers
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

from typing import Optional, Sequence, TYPE_CHECKING, Union

import numpy as np

from cirq._compat import proper_repr
from cirq.sim.clifford import stabilizer_state_ch_form
from cirq.sim.clifford.stabilizer_simulation_state import StabilizerSimulationState

if TYPE_CHECKING:
    import cirq


class StabilizerChFormSimulationState(
    StabilizerSimulationState[stabilizer_state_ch_form.StabilizerStateChForm]
):
    """Wrapper around a stabilizer state in CH form for the act_on protocol."""

    def __init__(
        self,
        *,
        prng: Optional[np.random.RandomState] = None,
        qubits: Optional[Sequence['cirq.Qid']] = None,
        initial_state: Union[int, 'cirq.StabilizerStateChForm'] = 0,
        classical_data: Optional['cirq.ClassicalDataStore'] = None,
    ):
        """Initializes with the given state and the axes for the operation.

        Args:
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            initial_state: The initial state for the simulation. This can be a
                full CH form passed by reference which will be modified inplace,
                or a big-endian int in the computational basis. If the state is
                an integer, qubits must be provided in order to determine
                array sizes.
            classical_data: The shared classical data container for this
                simulation.

        Raises:
            ValueError: If initial state is an integer but qubits are not
                provided.
        """
        if isinstance(initial_state, int):
            if qubits is None:
                raise ValueError('Must specify qubits if initial state is integer')
            initial_state = stabilizer_state_ch_form.StabilizerStateChForm(
                len(qubits), initial_state
            )
        super().__init__(
            state=initial_state, prng=prng, qubits=qubits, classical_data=classical_data
        )

    def __repr__(self) -> str:
        return (
            'cirq.StabilizerChFormSimulationState('
            f'initial_state={proper_repr(self.state)},'
            f' qubits={self.qubits!r},'
            f' classical_data={self.classical_data!r})'
        )
