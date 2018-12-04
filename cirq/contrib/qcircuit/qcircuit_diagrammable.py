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

from typing import Optional, Tuple, Any

import abc

from cirq import ops, protocols


class QCircuitDiagrammable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def qcircuit_diagram_info(self, args: protocols.CircuitDiagramInfoArgs
                              ) -> protocols.CircuitDiagramInfo:
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

    def qcircuit_diagram_info(self, args: protocols.CircuitDiagramInfoArgs
                              ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(self.symbols)


def _get_multigate_parameters(gate: Any,
                              args: protocols.CircuitDiagramInfoArgs
                              ) -> Optional[Tuple[int, int]]:
    if not isinstance(gate, ops.InterchangeableQubitsGate):
        return None
    if args.qubit_map is None or args.known_qubits is None:
        return None

    indices = [args.qubit_map[q] for q in args.known_qubits]
    min_index = min(indices)
    n_qubits = len(args.known_qubits)
    if sorted(indices) != list(range(min_index, min_index + n_qubits)):
        return None
    return min_index, n_qubits


class _TextToQCircuitDiagrammable(QCircuitDiagrammable):
    def __init__(self, sub: protocols.SupportsCircuitDiagramInfo) -> None:
        self.sub = sub

    def qcircuit_diagram_info(self, args: protocols.CircuitDiagramInfoArgs
                              ) -> protocols.CircuitDiagramInfo:
        info = protocols.circuit_diagram_info(self.sub, args)
        multigate_parameters = _get_multigate_parameters(self.sub, args)
        if multigate_parameters is not None:
            min_index, n_qubits = multigate_parameters
            name = _escape_text_for_latex(str(self.sub).rsplit('**', 1)[0])
            if info.exponent != 1:
                name += '^{' + str(info.exponent) + '}'
            box = '\multigate{' + str(n_qubits - 1) + '}{' + name + '}'
            ghost = '\ghost{' + name + '}'
            assert args.qubit_map is not None
            assert args.known_qubits is not None
            symbols = tuple(box if (args.qubit_map[q] == min_index) else
                            ghost for q in args.known_qubits)
            return protocols.CircuitDiagramInfo(symbols,
                                                exponent=info.exponent,
                                                connected=False)
        s = [_escape_text_for_latex(e) for e in info.wire_symbols]
        if info.exponent != 1:
            s[0] += '^{' + str(info.exponent) + '}'
        return protocols.CircuitDiagramInfo(tuple('\\gate{' + e + '}'
                                                  for e in s))


class _FallbackQCircuitGate(QCircuitDiagrammable):
    def __init__(self, sub: Any) -> None:
        self.sub = sub

    def qcircuit_diagram_info(self, args: protocols.CircuitDiagramInfoArgs
                              ) -> protocols.CircuitDiagramInfo:
        name = str(self.sub)
        qubit_count = ((len(args.known_qubits) if
                       (args.known_qubits is not None) else 1)
                       if args.known_qubit_count is None
                       else args.known_qubit_count)
        symbols = [name] + ['#{}'.format(i + 1) for i in range(1, qubit_count)]
        escaped_symbols = tuple(_escape_text_for_latex(s) for s in symbols)
        return protocols.CircuitDiagramInfo(escaped_symbols)


def known_qcircuit_operation_symbols(op: ops.Operation
                                     ) -> Optional[QCircuitDiagrammable]:
    if isinstance(op, ops.GateOperation):
        return _known_gate_symbols(op.gate)
    return None


def _known_gate_symbols(
        gate: ops.Gate) -> Optional[QCircuitDiagrammable]:
    if gate == ops.X:
        return _HardcodedQCircuitSymbolsGate(r'\targ')
    if gate == ops.CZ:
        return _HardcodedQCircuitSymbolsGate(r'\control', r'\control')
    if gate == ops.CNOT:
        return _HardcodedQCircuitSymbolsGate(r'\control', r'\targ')
    if ops.MeasurementGate.is_measurement(gate):
        return _HardcodedQCircuitSymbolsGate(r'\meter')
    return None
