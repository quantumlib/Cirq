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

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Tuple,
    List,
    Sequence,
)


if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq


def _try_decompose_into_operations_and_qubits(
        val: Any
) -> Tuple[Optional[List['cirq.Operation']], Sequence['cirq.Qid']]:
    """Returns the value's decomposition (if any) and the qubits it applies to.
    """
    from cirq.protocols.decompose import (decompose_once,
                                          decompose_once_with_qubits)
    from cirq import LineQubit, Gate, Operation

    if isinstance(val, Gate):
        # Gates don't specify qubits, and so must be handled specially.
        qubits = LineQubit.range(val.num_qubits())
        return decompose_once_with_qubits(val, qubits, None), qubits

    if isinstance(val, Operation):
        return decompose_once(val, None), val.qubits

    result = decompose_once(val, None)
    if result is not None:
        return result, sorted({q for op in result for q in op.qubits})

    return None, ()
