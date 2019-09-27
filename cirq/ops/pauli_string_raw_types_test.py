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

import pytest

import cirq


def _make_qubits(n):
    return [cirq.NamedQubit('q{}'.format(i)) for i in range(n)]


def test_op_calls_validate():
    q0, q1, q2 = _make_qubits(3)
    bad_qubit = cirq.NamedQubit('bad')

    class ValidError(Exception):
        pass

    class ValiGate(cirq.PauliStringGateOperation):

        def validate_args(self, qubits):
            super().validate_args(qubits)
            if bad_qubit in qubits:
                raise ValidError()

        def map_qubits(self, qubit_map):
            ps = self.pauli_string.map_qubits(qubit_map)
            return ValiGate(ps)

    g = ValiGate(cirq.PauliString({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}))

    _ = g.with_qubits(q1, q0, q2)
    with pytest.raises(ValidError):
        _ = g.with_qubits(q0, q1, bad_qubit)


def test_on_wrong_number_qubits():
    q0, q1, q2 = _make_qubits(3)

    class DummyGate(cirq.PauliStringGateOperation):

        def map_qubits(self, qubit_map):
            ps = self.pauli_string.map_qubits(qubit_map)
            return DummyGate(ps)

    g = DummyGate(cirq.PauliString({q0: cirq.X, q1: cirq.Y}))

    _ = g.with_qubits(q1, q2)
    with pytest.raises(ValueError):
        _ = g.with_qubits()
    with pytest.raises(ValueError):
        _ = g.with_qubits(q2)
    with pytest.raises(ValueError):
        _ = g.with_qubits(q0, q1, q2)


def test_default_text_diagram():

    class DiagramGate(cirq.PauliStringGateOperation):

        def map_qubits(self, qubit_map):
            pass

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                                  ) -> cirq.CircuitDiagramInfo:
            return self._pauli_string_diagram_info(args)

    q0, q1, q2 = _make_qubits(3)
    ps = cirq.PauliString({q0: cirq.X, q1: cirq.Y, q2: cirq.Z})

    circuit = cirq.Circuit(
        DiagramGate(ps),
        DiagramGate(-ps),
    )
    cirq.testing.assert_has_diagram(
        circuit, """
q0: ───[X]───[X]───
       │     │
q1: ───[Y]───[Y]───
       │     │
q2: ───[Z]───[Z]───
""")
