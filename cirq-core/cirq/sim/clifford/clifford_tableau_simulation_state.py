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

"""A protocol for implementing high performance clifford tableau evolutions
for Clifford Simulator."""

from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING

import numpy as np

from cirq.qis import clifford_tableau
from cirq.sim.clifford.stabilizer_simulation_state import StabilizerSimulationState

if TYPE_CHECKING:
    import cirq


class CliffordTableauSimulationState(StabilizerSimulationState[clifford_tableau.CliffordTableau]):
    """State and context for an operation acting on a clifford tableau."""

    def __init__(
        self,
        tableau: cirq.CliffordTableau,
        prng: Optional[np.random.RandomState] = None,
        qubits: Optional[Sequence[cirq.Qid]] = None,
        classical_data: Optional[cirq.ClassicalDataStore] = None,
    ):
        """Inits CliffordTableauSimulationState.

        Args:
            tableau: The CliffordTableau to act on. Operations are expected to
                perform inplace edits of this object.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            classical_data: The shared classical data container for this
                simulation.
        """
        super().__init__(state=tableau, prng=prng, qubits=qubits, classical_data=classical_data)

    @property
    def tableau(self) -> cirq.CliffordTableau:
        return self.state
