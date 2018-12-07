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

from typing import cast, Optional, Tuple

from cirq import ops, protocols

from cirq.contrib.qcircuit.qcircuit_diagrammable import (
    QCircuitDiagrammable,
    _TextToQCircuitDiagrammable,
    _FallbackQCircuitGate,
)


def hardcoded_qcircuit_diagram_info(
        op: ops.Operation) -> Optional[protocols.CircuitDiagramInfo]:
    if not isinstance(op, ops.GateOperation):
        return None
    symbols = (
        (r'\targ',) if op.gate == ops.X else
        (r'\control', r'\control') if op.gate == ops.CZ else
        (r'\control', r'\targ') if op.gate == ops.CNOT else
        (r'\meter',) if ops.MeasurementGate.is_measurement(op.gate) else
        ())
    return (protocols.CircuitDiagramInfo(cast(Tuple[str, ...], symbols))
            if symbols else None)


def get_qcircuit_diagram_info(op: ops.Operation,
                              args: protocols.CircuitDiagramInfoArgs
                              ) -> protocols.CircuitDiagramInfo:
    info = hardcoded_qcircuit_diagram_info(op)
    if info is None:
        info = protocols.circuit_diagram_info(op, default=None)
        if info is not None:
            diagrammable = _TextToQCircuitDiagrammable(
                cast(protocols.SupportsCircuitDiagramInfo, op)
                ) # type: QCircuitDiagrammable
        elif isinstance(op, ops.GateOperation):
            diagrammable = _FallbackQCircuitGate(op.gate)
        else:
            diagrammable = _FallbackQCircuitGate(op)
        info = diagrammable.qcircuit_diagram_info(args)
    return info

