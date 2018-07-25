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

import cirq
from cirq import Extensions, ops
from cirq import abc
from cirq.ops import gate_features


class QCircuitDiagrammable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def qcircuit_diagram_info(self, args: ops.TextDiagramInfoArgs
                              ) -> ops.TextDiagramInfo:
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
                              ) -> ops.TextDiagramInfo:
        return ops.TextDiagramInfo(self.symbols)


class _TextToQCircuitDiagrammable(QCircuitDiagrammable):
    def __init__(self, sub: ops.TextDiagrammable) -> None:
        self.sub = sub

    def qcircuit_diagram_info(self, args: ops.TextDiagramInfoArgs
                              ) -> ops.TextDiagramInfo:
        info = self.sub.text_diagram_info(args)
        while True:
            if (args.qubit_map is None) or (args.known_qubits is None):
                break
            if not isinstance(self.sub,
                    gate_features.InterchangeableQubitsGate):
                break
            indices = [args.qubit_map[q] for q in args.known_qubits]
            min_index = min(indices)
            if sorted(indices) != list(range(min_index, max(indices) + 1)):
                break
            name = _escape_text_for_latex(str(self.sub).rsplit('**', 1)[0])
            if info.exponent != 1:
                name += '^{' + str(info.exponent) + '}'
            box = '\multigate{' + str(len(indices) - 1) + '}{' + name + '}'
            ghost = '\ghost{' + name + '}'
            symbols = tuple(box if (args.qubit_map[q] == min_index) else
                            ghost for q in args.known_qubits)
            return ops.TextDiagramInfo(symbols, exponent=info.exponent,
                    connected=False)
        s = [_escape_text_for_latex(e) for e in info.wire_symbols]
        if info.exponent != 1:
            s[0] += '^{' + str(info.exponent) + '}'
        return ops.TextDiagramInfo(tuple('\\gate{' + e + '}' for e in s))


class _FallbackQCircuitGate(QCircuitDiagrammable):
    def __init__(self, sub: ops.Gate) -> None:
        self.sub = sub

    def qcircuit_diagram_info(self, args: ops.TextDiagramInfoArgs
                              ) -> ops.TextDiagramInfo:
        name = str(self.sub)
        qubit_count = ((len(args.known_qubits) if
                       (args.known_qubits is not None) else 1)
                       if args.known_qubit_count is None
                       else args.known_qubit_count)
        symbols = tuple(_escape_text_for_latex('{}:{}'.format(name, i))
                        for i in range(qubit_count))
        return ops.TextDiagramInfo(symbols)


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
