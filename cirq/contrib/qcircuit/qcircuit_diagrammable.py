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

from typing import Tuple

import cirq
from cirq import Extensions, ops
from cirq import abc


class QCircuitDiagrammable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def qcircuit_diagram_info(self, args: ops.TextDiagramInfoArgs
                              ) -> Tuple[str, ...]:
        pass


def _escape_text_for_latex(text):
    escaped = (text
               .replace('\\', '\\textbackslash{}')
               .replace('^', '\\textasciicircum{}')
               .replace('~', '\\textasciitilde{}')
               .replace('_', '\\_')
               .replace('{', '\\{')
               .replace('}', '\\}')
               .replace('$', '\\$')
               .replace('%', '\\%')
               .replace('&', '\\&')
               .replace('#', '\\#'))
    return '\\text{' + escaped + '}'


class _HardcodedQCircuitSymbolsGate(QCircuitDiagrammable):
    def __init__(self, *symbols: str) -> None:
        self.symbols = symbols

    def qcircuit_diagram_info(self, args: ops.TextDiagramInfoArgs
                              ) -> Tuple[str, ...]:
        return ops.TextDiagramInfo(self.symbols).wire_symbols


class _TextToQCircuitDiagrammable(QCircuitDiagrammable):
    def __init__(self, sub: ops.TextDiagrammable) -> None:
        self.sub = sub

    def qcircuit_diagram_info(self, args: ops.TextDiagramInfoArgs
                              ) -> Tuple[str, ...]:
        info = self.sub.text_diagram_info(args)

        s = [_escape_text_for_latex(e) for e in info.wire_symbols]
        if info.exponent != 1:
            s[0] += '^{' + str(info.exponent) + '}'
        return tuple('\\gate{' + e + '}' for e in s)


class _FallbackQCircuitGate(QCircuitDiagrammable):
    def __init__(self, sub: ops.Gate) -> None:
        self.sub = sub

    def qcircuit_diagram_info(self, args: ops.TextDiagramInfoArgs
                              ) -> Tuple[str, ...]:
        name = str(self.sub)
        qubit_count = (1 if args.known_qubit_count is None
                       else args.known_qubit_count)
        symbols = tuple(_escape_text_for_latex('{}:{}'.format(name, i))
                        for i in range(qubit_count))
        return symbols


fallback_qcircuit_extensions = Extensions()
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammable,
    ops.TextDiagrammable,
    _TextToQCircuitDiagrammable)
fallback_qcircuit_extensions.add_recursive_cast(
    QCircuitDiagrammable,
    ops.GateOperation,
    lambda ext, op: ext.try_cast(QCircuitDiagrammable, op.gate))
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammable,
    ops.RotXGate,
    lambda gate:
        _HardcodedQCircuitSymbolsGate('\\targ')
        if gate.half_turns == 1
        else None)
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammable,
    ops.MeasurementGate,
    lambda gate: _HardcodedQCircuitSymbolsGate('\\meter'))
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammable,
    cirq.google.ExpWGate,
    lambda gate:
        _HardcodedQCircuitSymbolsGate('\\targ')
        if gate.half_turns == 1 and gate.axis_half_turns == 0
        else None)
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammable,
    ops.Rot11Gate,
    lambda gate:
        _HardcodedQCircuitSymbolsGate('\\control', '\\control')
        if gate.half_turns == 1
        else None)
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammable,
    ops.CNotGate,
    lambda gate: _HardcodedQCircuitSymbolsGate('\\control', '\\targ'))
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammable,
    ops.Gate,
    _FallbackQCircuitGate)
