import cirq
import numpy as np
import pytest
from cirq.contrib.qasm_import import circuit_from_qasm


def assert_gate_isinstance_and_exponent(op, expected_type, expected_exp):
    assert isinstance(op.gate, expected_type)
    if expected_exp is not None:
        assert hasattr(op.gate, "exponent")
        np.testing.assert_allclose(op.gate.exponent, expected_exp)


@pytest.mark.parametrize("name, qasm, expected_type, expected_exp", [
    ("rzz",  "rzz(pi/2) q[0],q[1];", cirq.ZZPowGate, 0.5),
    ("rxx",  "rxx(pi) q[0],q[1];", cirq.XXPowGate, 1.0),
    ("ryy",  "ryy(pi/4) q[0],q[1];", cirq.YYPowGate, 0.25),
    ("rx",   "rx(pi/2) q[0];", cirq.rx(np.pi/2), None),
    ("ry",   "ry(pi/4) q[0];", cirq.ry(np.pi/4), None),
    ("rz",   "rz(pi/3) q[0];", cirq.rz(np.pi/3), None),
    ("x",    "x q[0];", cirq.X, None),
    ("y",    "y q[0];", cirq.Y, None),
    ("z",    "z q[0];", cirq.Z, None),
    ("h",    "h q[0];", cirq.H, None),
    ("s",    "s q[0];", cirq.S, None),
    ("sdg",  "sdg q[0];", cirq.ZPowGate, -0.5),
    ("t",    "t q[0];", cirq.T, None),
    ("tdg",  "tdg q[0];", cirq.ZPowGate, -0.25),
    ("sx",   "sx q[0];", cirq.XPowGate, 0.5),
    ("sxdg", "sxdg q[0];", cirq.XPowGate, -0.5),
    ("id",   "id q[0];", cirq.IdentityGate(1), None),
    ("swap", "swap q[0],q[1];", cirq.SWAP, None),
    ("iswap", "iswap q[0],q[1];", cirq.ISwapPowGate(), None),
    ("cz",   "cz q[0],q[1];", cirq.CZ, None),
    ("cy",   "cy q[0],q[1];", cirq.ControlledGate(cirq.Y), None),
    ("cx",   "cx q[0],q[1];", cirq.CX, None),
    ("ccx",  "ccx q[0],q[1],q[2];", cirq.CCX, None),
    ("cswap", "cswap q[0],q[1],q[2];", cirq.CSWAP, None),
    ("ch",   "ch q[0],q[1];", cirq.ControlledGate(cirq.H), None),
])
def test_basic_gates(name, qasm, expected_type, expected_exp):
    full_qasm = f"""
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    {qasm}
    """
    circuit = circuit_from_qasm(full_qasm)
    op = circuit[0].operations[0]
    if expected_exp is not None:
        assert_gate_isinstance_and_exponent(op, expected_type, expected_exp)
    else:
        assert isinstance(op.gate, type(expected_type))


def test_crx():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    crx(pi/2) q[0],q[1];
    """
    circuit = circuit_from_qasm(qasm)
    op = circuit[0].operations[0]
    assert isinstance(op.gate, cirq.ControlledGate)
    assert isinstance(op.gate.sub_gate, cirq.rx(np.pi/2).__class__)

def test_cry():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    cry(pi/3) q[0],q[1];
    """
    circuit = circuit_from_qasm(qasm)
    op = circuit[0].operations[0]
    assert isinstance(op.gate.sub_gate, cirq.ry(np.pi/3).__class__)

def test_crz():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    crz(pi/5) q[0],q[1];
    """
    circuit = circuit_from_qasm(qasm)
    op = circuit[0].operations[0]
    assert isinstance(op.gate.sub_gate, cirq.rz(np.pi/5).__class__)

def test_cu1_cu3_u1_u2_u3():
    # These gates are mapped via QasmUGate (non-native Cirq gates)
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    u1(pi/2) q[0];
    u2(pi/2, pi/4) q[0];
    u3(pi, pi/2, pi/4) q[0];
    """
    circuit = circuit_from_qasm(qasm)
    assert len(circuit) == 3  # Three gates
