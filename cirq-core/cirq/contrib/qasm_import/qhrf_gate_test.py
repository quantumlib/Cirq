import cirq
import numpy as np
import pytest
from cirq.contrib.qasm_import import circuit_from_qasm

def test_rzz_gate():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    rzz(pi/2) q[0],q[1];
    """
    circuit = circuit_from_qasm(qasm)
    op = circuit[0].operations[0]
    assert isinstance(op.gate, cirq.ZZPowGate)
    np.testing.assert_allclose(op.gate.exponent, 0.5)

def test_rxx_gate():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    rxx(pi) q[0],q[1];
    """
    circuit = circuit_from_qasm(qasm)
    op = circuit[0].operations[0]
    assert isinstance(op.gate, cirq.XXPowGate)
    np.testing.assert_allclose(op.gate.exponent, 1.0)

def test_ryy_gate():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    ryy(pi/4) q[0],q[1];
    """
    circuit = circuit_from_qasm(qasm)
    op = circuit[0].operations[0]
    assert isinstance(op.gate, cirq.YYPowGate)
    np.testing.assert_allclose(op.gate.exponent, 0.25)

