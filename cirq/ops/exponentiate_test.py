# imports
import cirq
import numpy as np
from exponentiate import exponentiate_qubit_operator


def test_exponentiate_X():

    qubit = [cirq.google.XmonQubit(0, 0)]
    circuit1 = cirq.Circuit()
    circuit1.append([cirq.RotXGate(half_turns=-1 / 2).on(qubit[0])])
    sim = cirq.google.XmonSimulator()

    op = {((0, 'X'),): np.pi / 4}
    circuit2 = exponentiate_qubit_operator(operator=op, qubits=qubit, time=1, trotter_steps=1)

    results1 = sim.simulate(circuit1)
    results2 = sim.simulate(circuit2)

    assert np.array_equal(np.round(results1.final_state, 4), np.round(results2.final_state, 4))


def test_exponentiate_Y():

    qubit = [cirq.google.XmonQubit(0, 0)]
    circuit1 = cirq.Circuit()
    circuit1.append([cirq.RotYGate(half_turns=-1 / 2).on(qubit[0])])
    sim = cirq.google.XmonSimulator()

    op = {((0, 'Y'),): np.pi / 4}
    circuit2 = exponentiate_qubit_operator(operator=op, qubits=qubit, time=1, trotter_steps=1)

    results1 = sim.simulate(circuit1)
    results2 = sim.simulate(circuit2)

    assert np.array_equal(np.round(results1.final_state, 4), np.round(results2.final_state, 4))


def test_exponentiate_Z():

    qubit = [cirq.google.XmonQubit(0, 0)]
    circuit1 = cirq.Circuit()
    circuit1.append([cirq.RotZGate(half_turns=-1 / 2).on(qubit[0])])
    sim = cirq.google.XmonSimulator()

    op = {((0, 'Z'),): -1 * np.pi / 4}
    circuit2 = exponentiate_qubit_operator(operator=op, qubits=qubit, time=1, trotter_steps=1)

    results1 = sim.simulate(circuit1)
    results2 = sim.simulate(circuit2)

    assert np.array_equal(np.round(results1.final_state, 4), np.round(results2.final_state, 4))
