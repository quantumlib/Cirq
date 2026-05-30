# Copyright 2026 The Cirq Developers
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

"""Transformer to remove gates that are outside of the backwards lightcone of measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cirq import circuits, protocols
from cirq.transformers import transformer_api

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer()
def lightcone_filter(
    circuit: cirq.AbstractCircuit, *, context: cirq.TransformerContext | None = None
) -> cirq.Circuit:
    """Apply a lightcone filter to the input circuit.

    Returns:
        A copy of the original circuit, with gates outside of the backwards lightcone of
        measurements removed.
    """

    support: set[cirq.Qid] = set()
    new_moments = []
    for moment in circuit[::-1]:
        new_ops = []
        for op in moment:
            if protocols.is_measurement(op):
                new_ops.append(op)
                support.update(op.qubits)
            else:
                if max(q in support for q in op.qubits):
                    new_ops.append(op)
                    support.update(op.qubits)
        new_moments.append(circuits.Moment(*new_ops))
    return circuits.Circuit.from_moments(*new_moments[::-1])
