""" Example program to demonstrate BB84 QKD Protocol

BB84 [1] is a quantum key distribution (QKD) protocol developed by
Charles Bennett and Gilles Brassard in 1984. It was the first quantum
cryptographic protocol, using the laws of quantum mechanics (specifically,
no-cloning) to provide provably secure key generation.

BB84 relies on the fact that it is impossible to gain information
distinguishing two non-orthogonal states without disturbing the signal.

The scheme involves two parties Alice and Bob connected by a classical
communication channel. In addition to this, Alice can also prepare
qubits in a particular state and send them to Bob using a unidirectional
quantum channel.

Alice generates two random binary strings a and b of the same length n.
The string a encodes the state and the string b encodes the basis.
She then prepares n qubits according to the following prescription:

|q[i]⟩ = |0⟩ if a[i] == 0 and b[i] == 0
|q[i]⟩ = |1⟩ if a[i] == 1 and b[i] == 0
|q[i]⟩ = |+⟩ if a[i] == 0 and b[i] == 1
|q[i]⟩ = |-⟩ if a[i] == 1 and b[i] == 1

where |+/-⟩ = 1/sqrt(2)*(|0⟩+/-|1⟩).

Alice sends her qubits to Bob. Bob then generates a random binary string
c of length n. He measures the qubit |q[i]⟩ in the {|0⟩, |1⟩} basis
(computational basis) if c[i] == 0 and in the {|+⟩,|-⟩} basis
(Hadamard basis) if c[i] == 1 and stores the result in a string m.
Alice and Bob then announce the strings b and c, which encode
the random basis choices of Alice and Bob respectively.

The strings a and m match in the places where b and c are the same.
This happens because the state was measured in the same basis in
which it was prepared. For the remaining bits, the results are
uncorrelated. The bits from strings a and m where the bases match
can be used as a key for cryptography.

[1]: https://doi.org/10.1016/j.tcs.2014.05.025

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
import random
import cirq


def main(num_qubits=8):

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
