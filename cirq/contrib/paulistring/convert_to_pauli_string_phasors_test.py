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
from cirq.contrib.paulistring import ConvertToPauliStringPhasors


def test_convert():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.Y(q1) ** 0.25,
        cirq.Z(q0) ** 0.125,
        cirq.H(q1),
    )
    c_orig = cirq.Circuit(circuit)
    ConvertToPauliStringPhasors().optimize_circuit(circuit)

    cirq.testing.assert_allclose_up_to_global_phase(circuit.unitary(), c_orig.unitary(), atol=1e-7)
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───[X]────────[Z]^(1/8)─────────

1: ───[Y]^0.25───[Y]^-0.5────[Z]───
""",
    )


def test_convert_keep_clifford():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.Y(q1) ** 0.25,
        cirq.Z(q0) ** 0.125,
        cirq.SingleQubitCliffordGate.H(q1),
    )
    c_orig = cirq.Circuit(circuit)
    ConvertToPauliStringPhasors(keep_clifford=True).optimize_circuit(circuit)

    cirq.testing.assert_allclose_up_to_global_phase(circuit.unitary(), c_orig.unitary(), atol=1e-7)
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───X──────────[Z]^(1/8)───

1: ───[Y]^0.25───H───────────
""",
    )


def test_already_converted():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.PauliStringPhasor(cirq.X.on(q0)))
    c_orig = cirq.Circuit(circuit)
    ConvertToPauliStringPhasors().optimize_circuit(circuit)

    assert circuit == c_orig


def test_ignore_unsupported_gate():
    class UnsupportedDummy(cirq.TwoQubitGate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        UnsupportedDummy()(q0, q1),
    )
    c_orig = cirq.Circuit(circuit)
    ConvertToPauliStringPhasors(ignore_failures=True).optimize_circuit(circuit)

    assert circuit == c_orig


def test_fail_unsupported_gate():
    class UnsupportedDummy(cirq.TwoQubitGate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        UnsupportedDummy()(q0, q1),
    )
    with pytest.raises(TypeError):
        ConvertToPauliStringPhasors().optimize_circuit(circuit)
