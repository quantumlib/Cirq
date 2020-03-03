import numpy as np
import pytest
import sympy

import cirq
from cirq.sim.clifford.clifford_gate_decomposer import CLIFFORD_GATE_DECOMPOSER

def test_identity():
    seq, phase =  CLIFFORD_GATE_DECOMPOSER.decompose(cirq.I)

    assert seq == ''
    assert np.allclose(phase, 1)

def test_x_gate():
    seq, phase =  CLIFFORD_GATE_DECOMPOSER.decompose(cirq.X)\

    assert seq == 'HSSH'
    assert np.allclose(phase, 1)

def test_y_gate():
    seq, phase =  CLIFFORD_GATE_DECOMPOSER.decompose(cirq.Y)

    assert seq == 'HSSHSS'
    assert np.allclose(phase, 1j)

def test_z_gate():
    seq, phase =  CLIFFORD_GATE_DECOMPOSER.decompose(cirq.Z)

    assert seq == 'SS'
    assert np.allclose(phase, 1)

def test_s_gate():
    seq, phase =  CLIFFORD_GATE_DECOMPOSER.decompose(cirq.S)

    assert seq == 'S'
    assert np.allclose(phase, 1)

def test_h_gate():
    seq, phase =  CLIFFORD_GATE_DECOMPOSER.decompose(cirq.H)

    assert seq == 'H'
    assert np.allclose(phase, 1)

