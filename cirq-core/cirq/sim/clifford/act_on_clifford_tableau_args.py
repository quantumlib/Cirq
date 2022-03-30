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

from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from cirq._compat import deprecated_parameter
from cirq.qis import clifford_tableau
from cirq.sim.clifford.act_on_stabilizer_args import ActOnStabilizerArgs

if TYPE_CHECKING:
    import cirq


class ActOnCliffordTableauArgs(ActOnStabilizerArgs[clifford_tableau.CliffordTableau]):
    """State and context for an operation acting on a clifford tableau."""

    @deprecated_parameter(
        deadline='v0.15',
        fix='Use classical_data.',
        parameter_desc='log_of_measurement_results and positional arguments',
        match=lambda args, kwargs: 'log_of_measurement_results' in kwargs or len(args) > 3,
    )
    def __init__(
        self,
        tableau: 'cirq.CliffordTableau',
        prng: Optional[np.random.RandomState] = None,
        log_of_measurement_results: Optional[Dict[str, List[int]]] = None,
        qubits: Optional[Sequence['cirq.Qid']] = None,
        classical_data: Optional['cirq.ClassicalDataStore'] = None,
    ):
        """Inits ActOnCliffordTableauArgs.

        Args:
            tableau: The CliffordTableau to act on. Operations are expected to
                perform inplace edits of this object.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into.
            classical_data: The shared classical data container for this
                simulation.
        """
        super().__init__(
            state=tableau,
            prng=prng,
            qubits=qubits,
            log_of_measurement_results=log_of_measurement_results,
            classical_data=classical_data,
        )

    @property
    def tableau(self) -> 'cirq.CliffordTableau':
        return self.state
