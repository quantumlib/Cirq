# imports
import cirq
import numpy as np
from expectation_value import expectation_value


def test_sampling_ZZ():
    op = {((0, 'Z'), (1, 'Z')): 1}

    qubit = [cirq.google.XmonQubit(0, 0), cirq.google.XmonQubit(0, 1)]
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(qubit[0]))
    circuit.append(cirq.CNOT.on(qubit[0], qubit[1]))
    circuit.append(cirq.H.on(qubit[0]))

    expect = expectation_value(circuit=circuit, operator=op, measurement=True, method='wavefunction', repetitions=1000)

    assert np.absolute(np.round(expect, 5)) < 0.2


def test_sampling_XX():

    op = {((0, 'X'), (1, 'Z')): 1}
    qubit = [cirq.google.XmonQubit(0, 0), cirq.google.XmonQubit(0, 1)]
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(qubit[0]))
    circuit.append(cirq.CNOT.on(qubit[0], qubit[1]))
    circuit.append(cirq.H.on(qubit[0]))

    expect = expectation_value(circuit=circuit, operator=op, measurement=True, method='wavefunction', repetitions=1000)

    assert np.round(expect, 5) == 1.0


def test_expectation_Y():
    op = {((0, 'Y'),): 1}
    qubit = [cirq.google.XmonQubit(0, 0)]
    circuit = cirq.Circuit()
    circuit.append(cirq.RotXGate(half_turns=-1 / 2).on(qubit[0]))

    expect = expectation_value(circuit=circuit, operator=op, measurement=False, method='wavefunction')

    assert np.round(expect, 5) == 1.0

    circuit.append(cirq.RotXGate(half_turns=+1).on(qubit[0]))
    expect = expectation_value(circuit=circuit, operator=op, measurement=False, method='wavefunction')

    assert np.round(expect, 5) == -1.0


def test_expectation_X():
    op = {((0, 'X'),): 1}
    qubit = [cirq.google.XmonQubit(0, 0)]
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(qubit[0]))

    expect = expectation_value(circuit=circuit, operator=op, measurement=False, method='wavefunction')

    assert np.round(expect, 5) == 1.0

    circuit.append(cirq.H.on(qubit[0]))
    expect = expectation_value(circuit=circuit, operator=op, measurement=False, method='wavefunction')

    assert np.round(expect, 4) == 0


def test_expectation_ZZ():
    op = {((0, 'Z'), (1, 'Z')): 1}
    qubit = [cirq.google.XmonQubit(0, 0), cirq.google.XmonQubit(0, 1)]
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(qubit[0]))
    circuit.append(cirq.CNOT.on(qubit[0], qubit[1]))

    expect = expectation_value(circuit=circuit, operator=op, measurement=False, method='wavefunction')

    assert np.round(expect, 5) == 1.0
