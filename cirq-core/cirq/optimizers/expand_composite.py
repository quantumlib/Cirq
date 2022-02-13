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

"""An optimizer that expands composite operations via `cirq.decompose`."""

from typing import Callable, Optional, TYPE_CHECKING

from cirq import ops, protocols
from cirq.circuits.optimization_pass import (
    PointOptimizer,
    PointOptimizationSummary,
)
from cirq._compat import deprecated_class

if TYPE_CHECKING:
    import cirq


@deprecated_class(deadline='v1.0', fix='Use cirq.expand_composite instead.')
class ExpandComposite(PointOptimizer):
    """An optimizer that expands composite operations via `cirq.decompose`.

    For each operation in the circuit, this pass examines if the operation can
    be decomposed. If it can be, the operation is cleared out and and replaced
    with its decomposition using a fixed insertion strategy.
    """

    def __init__(self, no_decomp: Callable[[ops.Operation], bool] = (lambda _: False)) -> None:
        """Construct the optimization pass.

        Args:
            no_decomp: A predicate that determines whether an operation should
                be decomposed or not. Defaults to decomposing everything.
        """
        super().__init__()
        self.no_decomp = no_decomp

    def optimization_at(
        self, circuit: 'cirq.Circuit', index: int, op: 'cirq.Operation'
    ) -> Optional['cirq.PointOptimizationSummary']:
        decomposition = protocols.decompose(op, keep=self.no_decomp, on_stuck_raise=None)
        if decomposition == [op]:
            return None

        return PointOptimizationSummary(
            clear_span=1, clear_qubits=op.qubits, new_operations=decomposition
        )
