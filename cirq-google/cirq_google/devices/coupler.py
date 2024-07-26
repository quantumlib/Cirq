# Copyright 2024 The Cirq Developers
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

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from cirq import ops, protocols

if TYPE_CHECKING:
    import cirq


class Coupler(ops.Qid):
    """A Qid representing the coupler between two qubits."""

    _qubit1: ops.Qid
    _qubit2: ops.Qid
    _hash: Optional[int] = None
    _comp_key: Optional[Tuple[Any, Any]] = None

    def __init__(self, qubit1: ops.Qid, qubit2: ops.Qid) -> None:
        """Creates a grid coupler between two qubits.

        Note that the qubits will be implicitly sorted.
        ie. `cirq_google.GridCoupler(q1, q2)` will be the same as
        `cirq_google.GridCoupler(q2, q1)`.

        Note that, if using custom Qid objects, the Qid must
        have an ordering that allows for comparison.

        Args:
            qubit1: The first qubit/Qid of the pair
            qubit2: The second qubit/Qid of the pair
        """
        if qubit1 < qubit2:
            self._qubit1 = qubit1
            self._qubit2 = qubit2
        else:
            self._qubit1 = qubit2
            self._qubit2 = qubit1

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._qubit1, self._qubit2))
        return self._hash

    def _comparison_key(self):
        if self._comp_key is None:
            self._comp_key = (self._qubit1._comparison_key(), self.qubit2._comparison_key())
        return self._comp_key

    @property
    def qubit1(self) -> ops.Qid:
        return self._qubit1

    @property
    def qubit2(self) -> ops.Qid:
        return self._qubit2

    @property
    def dimension(self) -> int:
        return 2

    def __getnewargs__(self):
        """Returns a tuple of args to pass to __new__ when unpickling."""
        return (self._qubit1, self._qubit2)

    def __repr__(self) -> str:
        return f"cirq_google.Coupler({self._qubit1!r}, {self._qubit2!r})"

    def __str__(self) -> str:
        return f"c({self._qubit1},{self._qubit2})"

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=(str(self),))

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['qubit1', 'qubit2'])
