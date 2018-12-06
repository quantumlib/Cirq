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

from typing import cast

from cirq import ops, protocols

from cirq.contrib.qcircuit.qcircuit_diagrammable import (
    known_qcircuit_operation_symbols,
    _TextToQCircuitDiagrammable,
    _FallbackQCircuitGate,
)


def get_qcircuit_diagram_info(op: ops.Operation,
                              args: protocols.CircuitDiagramInfoArgs
                              ) -> protocols.CircuitDiagramInfo:
    diagrammable = known_qcircuit_operation_symbols(op)
    if diagrammable is None:
        info = protocols.circuit_diagram_info(op, default=None)
        if info is not None:
            diagrammable = _TextToQCircuitDiagrammable(
                cast(protocols.SupportsCircuitDiagramInfo, op))
        elif isinstance(op, ops.GateOperation):
            diagrammable = _FallbackQCircuitGate(op.gate)
        else:
            diagrammable = _FallbackQCircuitGate(op)
    return diagrammable.qcircuit_diagram_info(args)

