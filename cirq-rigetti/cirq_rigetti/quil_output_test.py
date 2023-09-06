# Copyright 2020 The Cirq Developers
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

import os
import numpy as np
import pytest

import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate

import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput


def _make_qubits(n):
    return [cirq.NamedQubit(f'q{i}') for i in range(n)]


def test_single_gate_no_parameter():
    (q0,) = _make_qubits(1)
    output = cirq_rigetti.quil_output.QuilOutput((cirq.X(q0),), (q0,))
    assert (
        str(output)
        == """# Created using Cirq.

X 0\n"""
    )


def test_single_gate_with_parameter():
    (q0,) = _make_qubits(1)
    output = cirq_rigetti.quil_output.QuilOutput((cirq.X(q0) ** 0.5,), (q0,))
    assert (
        str(output)
        == f"""# Created using Cirq.

RX({np.pi / 2}) 0\n"""
    )


def test_single_gate_named_qubit():
    q = cirq.NamedQubit('qTest')
    output = cirq_rigetti.quil_output.QuilOutput((cirq.X(q),), (q,))

    assert (
        str(output)
        == """# Created using Cirq.

X 0\n"""
    )


def test_h_gate_with_parameter():
    (q0,) = _make_qubits(1)
    output = cirq_rigetti.quil_output.QuilOutput((cirq.H(q0) ** 0.25,), (q0,))
    assert (
        str(output)
        == f"""# Created using Cirq.

RY({np.pi / 4}) 0
RX({np.pi / 4}) 0
RY({-np.pi / 4}) 0\n"""
    )


def test_save_to_file(tmpdir):
    file_path = os.path.join(tmpdir, 'test.quil')
    (q0,) = _make_qubits(1)
    output = cirq_rigetti.quil_output.QuilOutput((cirq.X(q0)), (q0,))
    output.save_to_file(file_path)
    with open(file_path, 'r') as f:
        file_content = f.read()
    assert (
        file_content
        == """# Created using Cirq.

X 0\n"""
    )


def test_quil_one_qubit_gate_repr():
    gate = cirq_rigetti.quil_output.QuilOneQubitGate(np.array([[1, 0], [0, 1]]))
    assert repr(gate) == (
        """cirq.circuits.quil_output.QuilOneQubitGate(matrix=
[[1 0]
 [0 1]]
)"""
    )


def test_quil_two_qubit_gate_repr():
    gate = cirq_rigetti.quil_output.QuilTwoQubitGate(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    assert repr(gate) == (
        """cirq.circuits.quil_output.QuilTwoQubitGate(matrix=
[[1 0 0 0]
 [0 1 0 0]
 [0 0 1 0]
 [0 0 0 1]]
)"""
    )


def test_quil_one_qubit_gate_eq():
    gate = cirq_rigetti.quil_output.QuilOneQubitGate(np.array([[1, 0], [0, 1]]))
    gate2 = cirq_rigetti.quil_output.QuilOneQubitGate(np.array([[1, 0], [0, 1]]))
    assert cirq.approx_eq(gate, gate2, atol=1e-16)
    gate3 = cirq_rigetti.quil_output.QuilOneQubitGate(np.array([[1, 0], [0, 1]]))
    gate4 = cirq_rigetti.quil_output.QuilOneQubitGate(np.array([[1, 0], [0, 2]]))
    assert not cirq.approx_eq(gate4, gate3, atol=1e-16)


def test_quil_two_qubit_gate_eq():
    gate = cirq_rigetti.quil_output.QuilTwoQubitGate(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    gate2 = cirq_rigetti.quil_output.QuilTwoQubitGate(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    assert cirq.approx_eq(gate, gate2, atol=1e-8)
    gate3 = cirq_rigetti.quil_output.QuilTwoQubitGate(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    gate4 = cirq_rigetti.quil_output.QuilTwoQubitGate(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]])
    )
    assert not cirq.approx_eq(gate4, gate3, atol=1e-8)


def test_quil_one_qubit_gate_output():
    (q0,) = _make_qubits(1)
    gate = cirq_rigetti.quil_output.QuilOneQubitGate(np.array([[1, 0], [0, 1]]))
    output = cirq_rigetti.quil_output.QuilOutput((gate.on(q0),), (q0,))
    assert (
        str(output)
        == """# Created using Cirq.

DEFGATE USERGATE1:
    1.0+0.0i, 0.0+0.0i
    0.0+0.0i, 1.0+0.0i
USERGATE1 0
"""
    )


def test_two_quil_one_qubit_gate_output():
    (q0,) = _make_qubits(1)
    gate = cirq_rigetti.quil_output.QuilOneQubitGate(np.array([[1, 0], [0, 1]]))
    gate1 = cirq_rigetti.quil_output.QuilOneQubitGate(np.array([[2, 0], [0, 3]]))
    output = cirq_rigetti.quil_output.QuilOutput((gate.on(q0), gate1.on(q0)), (q0,))
    assert (
        str(output)
        == """# Created using Cirq.

DEFGATE USERGATE1:
    1.0+0.0i, 0.0+0.0i
    0.0+0.0i, 1.0+0.0i
USERGATE1 0
DEFGATE USERGATE2:
    2.0+0.0i, 0.0+0.0i
    0.0+0.0i, 3.0+0.0i
USERGATE2 0
"""
    )


def test_quil_two_qubit_gate_output():
    (q0, q1) = _make_qubits(2)
    gate = cirq_rigetti.quil_output.QuilTwoQubitGate(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    output = cirq_rigetti.quil_output.QuilOutput((gate.on(q0, q1),), (q0, q1))
    assert (
        str(output)
        == """# Created using Cirq.

DEFGATE USERGATE1:
    1.0+0.0i, 0.0+0.0i, 0.0+0.0i, 0.0+0.0i
    0.0+0.0i, 1.0+0.0i, 0.0+0.0i, 0.0+0.0i
    0.0+0.0i, 0.0+0.0i, 1.0+0.0i, 0.0+0.0i
    0.0+0.0i, 0.0+0.0i, 0.0+0.0i, 1.0+0.0i
USERGATE1 0 1
"""
    )


def test_unsupported_operation():
    (q0,) = _make_qubits(1)

    class UnsupportedOperation(cirq.Operation):
        qubits = (q0,)
        with_qubits = NotImplemented

    output = cirq_rigetti.quil_output.QuilOutput((UnsupportedOperation(),), (q0,))
    with pytest.raises(ValueError):
        _ = str(output)


def test_i_swap_with_power():
    q0, q1 = _make_qubits(2)

    output = QuilOutput((cirq.ISWAP(q0, q1) ** 0.25,), (q0, q1))
    assert (
        str(output)
        == f"""# Created using Cirq.

XY({np.pi / 4}) 0 1
"""
    )


def test_all_operations():
    qubits = tuple(_make_qubits(5))
    operations = _all_operations(*qubits, include_measurements=False)
    output = cirq_rigetti.quil_output.QuilOutput(operations, qubits)

    assert (
        str(output)
        == f"""# Created using Cirq.

DECLARE m0 BIT[1]
DECLARE m1 BIT[1]
DECLARE m2 BIT[1]
DECLARE m3 BIT[3]

Z 0
RZ({5 * np.pi / 8}) 0
Y 0
RY({3 * np.pi / 8}) 0
X 0
RX({7 * np.pi / 8}) 0
H 1
CZ 0 1
CPHASE({np.pi / 4}) 0 1
CNOT 0 1
RY({-np.pi / 2}) 1
CPHASE({np.pi / 2}) 0 1
RY({np.pi / 2}) 1
SWAP 0 1
SWAP 1 0
PSWAP({3 * np.pi / 4}) 0 1
H 2
CCNOT 0 1 2
H 2
CCNOT 0 1 2
RZ({np.pi / 8}) 0
RZ({np.pi / 8}) 1
RZ({np.pi / 8}) 2
CNOT 0 1
CNOT 1 2
RZ({-np.pi / 8}) 1
RZ({np.pi / 8}) 2
CNOT 0 1
CNOT 1 2
RZ({-np.pi / 8}) 2
CNOT 0 1
CNOT 1 2
RZ({-np.pi / 8}) 2
CNOT 0 1
CNOT 1 2
H 2
RZ({np.pi / 8}) 0
RZ({np.pi / 8}) 1
RZ({np.pi / 8}) 2
CNOT 0 1
CNOT 1 2
RZ({-np.pi / 8}) 1
RZ({np.pi / 8}) 2
CNOT 0 1
CNOT 1 2
RZ({-np.pi / 8}) 2
CNOT 0 1
CNOT 1 2
RZ({-np.pi / 8}) 2
CNOT 0 1
CNOT 1 2
H 2
CSWAP 0 1 2
X 0
X 1
RX({3 * np.pi / 4}) 0
RX({3 * np.pi / 4}) 1
Y 0
Y 1
RY({3 * np.pi / 4}) 0
RY({3 * np.pi / 4}) 1
Z 0
Z 1
RZ({3 * np.pi / 4}) 0
RZ({3 * np.pi / 4}) 1
I 0
I 0
I 1
I 2
ISWAP 2 0
RZ({-0.111 * np.pi}) 1
RX({np.pi / 4}) 1
RZ({0.111 * np.pi}) 1
RZ({-0.333 * np.pi}) 1
RX({np.pi / 2}) 1
RZ({0.333 * np.pi}) 1
RZ({-0.777 * np.pi}) 1
RX({-np.pi / 2}) 1
RZ({0.777 * np.pi}) 1
WAIT
MEASURE 0 m0[0]
MEASURE 2 m1[0]
MEASURE 3 m2[0]
MEASURE 2 m1[0]
MEASURE 1 m3[0]
X 2 # Inverting for following measurement
MEASURE 2 m3[1]
MEASURE 3 m3[2]
"""
    )


def _all_operations(q0, q1, q2, q3, q4, include_measurements=True):
    return (
        cirq.Z(q0),
        cirq.Z(q0) ** 0.625,
        cirq.Y(q0),
        cirq.Y(q0) ** 0.375,
        cirq.X(q0),
        cirq.X(q0) ** 0.875,
        cirq.H(q1),
        cirq.CZ(q0, q1),
        cirq.CZ(q0, q1) ** 0.25,  # Requires 2-qubit decomposition
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q1) ** 0.5,  # Requires 2-qubit decomposition
        cirq.SWAP(q0, q1),
        cirq.SWAP(q1, q0) ** -1,
        cirq.SWAP(q0, q1) ** 0.75,  # Requires 2-qubit decomposition
        cirq.CCZ(q0, q1, q2),
        cirq.CCX(q0, q1, q2),
        cirq.CCZ(q0, q1, q2) ** 0.5,
        cirq.CCX(q0, q1, q2) ** 0.5,
        cirq.CSWAP(q0, q1, q2),
        cirq.XX(q0, q1),
        cirq.XX(q0, q1) ** 0.75,
        cirq.YY(q0, q1),
        cirq.YY(q0, q1) ** 0.75,
        cirq.ZZ(q0, q1),
        cirq.ZZ(q0, q1) ** 0.75,
        cirq.IdentityGate(1).on(q0),
        cirq.IdentityGate(3).on(q0, q1, q2),
        cirq.ISWAP(q2, q0),  # Requires 2-qubit decomposition
        cirq.PhasedXPowGate(phase_exponent=0.111, exponent=0.25).on(q1),
        cirq.PhasedXPowGate(phase_exponent=0.333, exponent=0.5).on(q1),
        cirq.PhasedXPowGate(phase_exponent=0.777, exponent=-0.5).on(q1),
        cirq.wait(q0, nanos=0),
        cirq.measure(q0, key='xX'),
        cirq.measure(q2, key='x_a'),
        cirq.measure(q3, key='X'),
        cirq.measure(q2, key='x_a'),
        cirq.measure(q1, q2, q3, key='multi', invert_mask=(False, True)),
    )


def test_fails_on_big_unknowns():
    class UnrecognizedGate(cirq.testing.ThreeQubitGate):
        pass

    q0, q1, q2 = _make_qubits(3)
    res = cirq_rigetti.quil_output.QuilOutput(UnrecognizedGate().on(q0, q1, q2), (q0, q1, q2))
    with pytest.raises(ValueError, match='Cannot output operation as QUIL'):
        _ = str(res)


def test_pauli_interaction_gate():
    (q0, q1) = _make_qubits(2)
    output = cirq_rigetti.quil_output.QuilOutput(PauliInteractionGate.CZ.on(q0, q1), (q0, q1))
    assert (
        str(output)
        == """# Created using Cirq.

CZ 0 1
"""
    )


def test_equivalent_unitaries():
    """This test covers the factor of pi change. However, it will be skipped
    if pyquil is unavailable for import.

    References:
        https://docs.pytest.org/en/latest/skipping.html#skipping-on-a-missing-import-dependency
    """
    pyquil = pytest.importorskip("pyquil")
    pyquil_simulation_tools = pytest.importorskip("pyquil.simulation.tools")
    q0, q1 = _make_qubits(2)
    operations = [
        cirq.XPowGate(exponent=0.5, global_shift=-0.5)(q0),
        cirq.YPowGate(exponent=0.5, global_shift=-0.5)(q0),
        cirq.ZPowGate(exponent=0.5, global_shift=-0.5)(q0),
        cirq.CZPowGate(exponent=0.5)(q0, q1),
        cirq.ISwapPowGate(exponent=0.5)(q0, q1),
    ]
    output = cirq_rigetti.quil_output.QuilOutput(operations, (q0, q1))
    program = pyquil.Program(str(output))
    pyquil_unitary = pyquil_simulation_tools.program_unitary(program, n_qubits=2)
    # Qubit ordering differs between pyQuil and Cirq.
    cirq_unitary = cirq.Circuit(cirq.SWAP(q0, q1), operations, cirq.SWAP(q0, q1)).unitary()
    assert np.allclose(pyquil_unitary, cirq_unitary)


QUIL_CPHASES_PROGRAM = f"""
CPHASE00({np.pi/2}) 0 1
CPHASE01({np.pi/2}) 0 1
CPHASE10({np.pi/2}) 0 1
CPHASE({np.pi/2}) 0 1
"""

QUIL_DIAGONAL_DECOMPOSE_PROGRAM = """
RZ(0) 0
RZ(0) 1
CPHASE(0) 0 1
X 0
X 1
CPHASE(0) 0 1
X 0
X 1
"""


def test_two_qubit_diagonal_gate_quil_output():
    pyquil = pytest.importorskip("pyquil")
    pyquil_simulation_tools = pytest.importorskip("pyquil.simulation.tools")
    q0, q1 = _make_qubits(2)
    operations = [
        cirq.TwoQubitDiagonalGate([np.pi / 2, 0, 0, 0])(q0, q1),
        cirq.TwoQubitDiagonalGate([0, np.pi / 2, 0, 0])(q0, q1),
        cirq.TwoQubitDiagonalGate([0, 0, np.pi / 2, 0])(q0, q1),
        cirq.TwoQubitDiagonalGate([0, 0, 0, np.pi / 2])(q0, q1),
    ]
    output = cirq_rigetti.quil_output.QuilOutput(operations, (q0, q1))
    program = pyquil.Program(str(output))
    assert f"\n{program.out()}" == QUIL_CPHASES_PROGRAM

    pyquil_unitary = pyquil_simulation_tools.program_unitary(program, n_qubits=2)
    # Qubit ordering differs between pyQuil and Cirq.
    cirq_unitary = cirq.Circuit(cirq.SWAP(q0, q1), operations, cirq.SWAP(q0, q1)).unitary()
    assert np.allclose(pyquil_unitary, cirq_unitary)
    # Also test non-CPHASE case, which decomposes into X/RZ/CPhase
    operations = [cirq.TwoQubitDiagonalGate([0, 0, 0, 0])(q0, q1)]
    output = cirq_rigetti.quil_output.QuilOutput(operations, (q0, q1))
    program = pyquil.Program(str(output))
    assert f"\n{program.out()}" == QUIL_DIAGONAL_DECOMPOSE_PROGRAM


def test_parseable_defgate_output():
    pyquil = pytest.importorskip("pyquil")
    q0, q1 = _make_qubits(2)
    operations = [
        cirq_rigetti.quil_output.QuilOneQubitGate(np.array([[1, 0], [0, 1]])).on(q0),
        cirq_rigetti.quil_output.QuilTwoQubitGate(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ).on(q0, q1),
    ]
    output = cirq_rigetti.quil_output.QuilOutput(operations, (q0, q1))
    # Just checks that we can create a pyQuil Program without crashing.
    pyquil.Program(str(output))


def test_unconveritble_op():
    (q0,) = _make_qubits(1)

    class MyGate(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

    op = MyGate()(q0)

    # Documenting that this
    # operation would crash if you call _op_to_quil_directly
    with pytest.raises(ValueError, match="Can't convert"):
        _ = cirq_rigetti.quil_output.QuilOutput(op, (q0,))._op_to_quil(op)
