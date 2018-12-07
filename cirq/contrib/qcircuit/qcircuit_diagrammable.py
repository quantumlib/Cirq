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

class QCircuitDiagrammable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def qcircuit_diagram_info(self, args: protocols.CircuitDiagramInfoArgs
                              ) -> protocols.CircuitDiagramInfo:
        pass


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
