""" Example program to demonstrate BB84 QKD Protocol

 === Example output ===

 0: ────X───H───H───M───

1: ────X───H───M───────

2: ────H───M───────────

3: ────H───M───────────

4: ────X───H───M───────

5: ────H───M───────────

6: ────X───H───M───────

7: ────X───H───M───────

8: ────X───H───M───────

9: ────H───M───────────

10: ───X───H───H───M───

11: ───H───H───M───────

12: ───X───H───H───M───

13: ───X───H───H───M───

14: ───H───M───────────

15: ───X───H───H───M───

Simulating...
Alice's basis:  HHHCHHCHCCHHHHHH
Bob's basis:    HCCHCCHCHHHHHHCH
Alice's bits:   1100101110101101
Bases match::   X_________XXXX_X
Expected key:   110111
Actual key:     110111

"""

import cirq
import random


def main(num_qubits=16):

    alice_basis = [random.randint(0, 1) for _ in range(num_qubits)]
    alice_state = [random.randint(0, 1) for _ in range(num_qubits)]
    bob_basis = [random.randint(0, 1) for _ in range(num_qubits)]

    expected_key = bitstring([
        alice_state[i]
        for i in range(num_qubits)
        if alice_basis[i] == bob_basis[i]
    ])

    circuit = make_bb84_circ(num_qubits, alice_basis, bob_basis, alice_state)
    print(circuit)

    # Run simulations.
    repetitions = 1
    print('Simulating...')
    result = cirq.Simulator().run(program=circuit, repetitions=repetitions)
    result_bitstring = bitstring(
        [int(result.measurements[str(i)]) for i in range(num_qubits)])

    # Take only qubits where bases match
    obtained_key = ''.join([
        result_bitstring[i]
        for i in range(num_qubits)
        if alice_basis[i] == bob_basis[i]
    ])

    assert expected_key == obtained_key, "Keys don't match"

    print_results(alice_basis, bob_basis, alice_state, expected_key,
                  obtained_key)


def make_bb84_circ(num_qubits, alice_basis, bob_basis, alice_state):

    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]

    circuit = cirq.Circuit()

    # Alice prepares her qubits
    alice_enc = []
    for index, _ in enumerate(alice_basis):
        if alice_state[index] == 1:
            alice_enc.append(cirq.X(qubits[index]))
        if alice_basis[index] == 1:
            alice_enc.append(cirq.H(qubits[index]))

    circuit.append(alice_enc)

    # Bob measures the received qubits
    bob_basis_choice = []
    for index, _ in enumerate(bob_basis):
        if bob_basis[index] == 1:
            bob_basis_choice.append(cirq.H(qubits[index]))

    circuit.append(bob_basis_choice)
    circuit.append(cirq.measure_each(*qubits))

    return circuit


def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)


def print_results(alice_basis, bob_basis, alice_state, expected_key,
                  obtained_key):
    num_qubits = len(alice_basis)
    basis_match = ''.join([
        'X' if alice_basis[i] == bob_basis[i] else '_'
        for i in range(num_qubits)
    ])
    alice_basis_str = "".join(
        ['C' if alice_basis[i] == 0 else "H" for i in range(num_qubits)])
    bob_basis_str = "".join(
        ['C' if bob_basis[i] == 0 else "H" for i in range(num_qubits)])

    print('Alice\'s basis:\t{}'.format(alice_basis_str))
    print('Bob\'s basis:\t{}'.format(bob_basis_str))
    print('Alice\'s bits:\t{}'.format(bitstring(alice_state)))
    print('Bases match::\t{}'.format(basis_match))
    print('Expected key:\t{}'.format(expected_key))
    print('Actual key:\t{}'.format(obtained_key))


if __name__ == "__main__":
    main()
