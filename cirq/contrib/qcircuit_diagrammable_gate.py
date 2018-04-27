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

from typing import Optional

import cirq
from cirq import Extensions, ops
from cirq import abc


class QCircuitDiagrammableGate(ops.Gate, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def qcircuit_wire_symbols(self, qubit_count: Optional[int] = None):
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


class _HardcodedQCircuitSymbolsGate(QCircuitDiagrammableGate):
    def __init__(self, *symbols) -> None:
        self.symbols = symbols

    def qcircuit_wire_symbols(self, qubit_count=None):
        return self.symbols


class _WrappedSymbolsQCircuitGate(QCircuitDiagrammableGate):
    def __init__(self, sub: ops.TextDiagrammableGate) -> None:
        self.sub = sub

    def qcircuit_wire_symbols(self, qubit_count=None):
        s = self.sub.text_diagram_wire_symbols()
        s = [_escape_text_for_latex(e) for e in s]
        e = self.sub.text_diagram_exponent()
        if e != 1:
            s[0] += '^{' + str(e) + '}'
        return tuple('\\gate{' + e + '}' for e in s)


class _FallbackQCircuitSymbolsGate(QCircuitDiagrammableGate):
    def __init__(self, sub: ops.Gate) -> None:
        self.sub = sub

    def qcircuit_wire_symbols(self, qubit_count=None):
        name = str(self.sub)
        if qubit_count is None:
            qubit_count = 1
        return tuple(_escape_text_for_latex('{}:{}'.format(name, i))
                     for i in range(qubit_count))


fallback_qcircuit_extensions = Extensions()
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammableGate,
    ops.TextDiagrammableGate,
    _WrappedSymbolsQCircuitGate)
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammableGate,
    ops.RotXGate,
    lambda gate:
        _HardcodedQCircuitSymbolsGate('\\targ')
        if gate.half_turns == 1
        else None)
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammableGate,
    ops.MeasurementGate,
    lambda gate: _HardcodedQCircuitSymbolsGate('\\meter'))
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammableGate,
    cirq.google.ExpWGate,
    lambda gate:
        _HardcodedQCircuitSymbolsGate('\\targ')
        if gate.half_turns == 1 and gate.axis_half_turns == 0
        else None)
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammableGate,
    ops.Rot11Gate,
    lambda gate:
        _HardcodedQCircuitSymbolsGate('\\control', '\\control')
        if gate.half_turns == 1
        else None)
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammableGate,
    ops.CNotGate,
    lambda gate: _HardcodedQCircuitSymbolsGate('\\control', '\\targ'))
fallback_qcircuit_extensions.add_cast(
    QCircuitDiagrammableGate,
    ops.Gate,
    _FallbackQCircuitSymbolsGate)
