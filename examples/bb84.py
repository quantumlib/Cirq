# pylint: disable=wrong-or-nonexistent-copyright-notice
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

BB84 is secure against intercept-and-resend attacks. The no-cloning
theorem [2] guarantees that a qubit that is in an unknown state to
begin with cannot be copied or cloned. Thus, any measurement will
destroy the initial state of the qubit. Suppose an eavesdropper Eve
intercepts all of Alice's qubits, measures them in a randomly chosen
basis, prepares another qubit in the state that she measured and resends
it to Bob. The state Eve measures is not necessarily the state Alice
prepared,  and hence, Alice and Bob will not measure the same outcome
for that qubit even if their basis choices match. Thus, Alice and Bob
can detect eavesdropping by comparing a few bits from their
obtained keys.

[1]: https://en.wikipedia.org/wiki/BB84
[2]: https://en.wikipedia.org/wiki/No-cloning_theorem

 === Example output ===

Simulating non-eavesdropped protocol

0: ───X───M───────────

1: ───H───H───M───────

2: ───X───H───M───────

3: ───X───H───M───────

4: ───X───H───M───────

5: ───X───H───H───M───

6: ───H───M───────────

7: ───H───H───M───────

Alice's basis:  CHCCCHCH
Bob's basis:    CHHHHHHH
Alice's bits:   10111100
Bases match::   XX___X_X
Expected key:   1010
Actual key:     1010

Simulating eavesdropped protocol

0: ───H───M───────────H───M───────────

1: ───H───M───────────H───H───M───────

2: ───X───H───H───M───X───H───H───M───

3: ───H───M───────────H───M───────────

4: ───M───────────────M───────────────

5: ───X───H───M───────X───H───M───────

6: ───H───M───────────X───H───M───────

7: ───X───H───H───M───X───H───M───────

Alice's basis:  HCHCCHCH
Bob's basis:    HHHCCHCC
Alice's bits:   00100101
Bases match::   X_XXXXX_
Expected key:   010010
Actual key:     111011

"""
import numpy as np
import cirq


def main(num_qubits=8):
    # Setup non-eavesdropped protocol
    print('Simulating non-eavesdropped protocol')
    qubits = cirq.LineQubit.range(num_qubits)
    alice_basis = [np.random.randint(0, 2) for _ in range(num_qubits)]
    alice_state = [np.random.randint(0, 2) for _ in range(num_qubits)]
    bob_basis = [np.random.randint(0, 2) for _ in range(num_qubits)]

    expected_key = bitstring(
        [alice_state[i] for i in range(num_qubits) if alice_basis[i] == bob_basis[i]]
    )

    circuit = make_bb84_circ(num_qubits, alice_basis, bob_basis, alice_state)

    # Run simulations.
    repetitions = 1

    result = cirq.Simulator().run(program=circuit, repetitions=repetitions)
    result_bitstring = bitstring([result.measurements[str(q)].item() for q in qubits])

    # Take only qubits where bases match
    obtained_key = ''.join(
        [result_bitstring[i] for i in range(num_qubits) if alice_basis[i] == bob_basis[i]]
    )

    assert expected_key == obtained_key, "Keys don't match"
    print(circuit)
    print_results(alice_basis, bob_basis, alice_state, expected_key, obtained_key)

    # Setup eavesdropped protocol
    print('Simulating eavesdropped protocol')
    np.random.seed(200)  # Seed random generator for consistent results
    alice_basis = [np.random.randint(0, 2) for _ in range(num_qubits)]
    alice_state = [np.random.randint(0, 2) for _ in range(num_qubits)]
    bob_basis = [np.random.randint(0, 2) for _ in range(num_qubits)]
    eve_basis = [np.random.randint(0, 2) for _ in range(num_qubits)]

    expected_key = bitstring(
        [alice_state[i] for i in range(num_qubits) if alice_basis[i] == bob_basis[i]]
    )

    # Eve intercepts the qubits

    alice_eve_circuit = make_bb84_circ(num_qubits, alice_basis, eve_basis, alice_state)

    # Run simulations.
    repetitions = 1
    result = cirq.Simulator().run(program=alice_eve_circuit, repetitions=repetitions)
    eve_state = [result.measurements[str(q)].item() for q in qubits]

    eve_bob_circuit = make_bb84_circ(num_qubits, eve_basis, bob_basis, eve_state)

    # Run simulations.
    repetitions = 1
    result = cirq.Simulator().run(program=eve_bob_circuit, repetitions=repetitions)
    result_bitstring = bitstring([result.measurements[str(q)].item() for q in qubits])

    # Take only qubits where bases match
    obtained_key = ''.join(
        [result_bitstring[i] for i in range(num_qubits) if alice_basis[i] == bob_basis[i]]
    )

    assert expected_key != obtained_key, "Keys shouldn't match"

    circuit = alice_eve_circuit + eve_bob_circuit
    print(circuit)
    print_results(alice_basis, bob_basis, alice_state, expected_key, obtained_key)


def make_bb84_circ(num_qubits, alice_basis, bob_basis, alice_state):

    qubits = cirq.LineQubit.range(num_qubits)

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


def print_results(alice_basis, bob_basis, alice_state, expected_key, obtained_key):
    num_qubits = len(alice_basis)
    basis_match = ''.join(
        ['X' if alice_basis[i] == bob_basis[i] else '_' for i in range(num_qubits)]
    )
    alice_basis_str = "".join(['C' if alice_basis[i] == 0 else "H" for i in range(num_qubits)])
    bob_basis_str = "".join(['C' if bob_basis[i] == 0 else "H" for i in range(num_qubits)])

    print(f'Alice\'s basis:\t{alice_basis_str}')
    print(f'Bob\'s basis:\t{bob_basis_str}')
    print(f'Alice\'s bits:\t{bitstring(alice_state)}')
    print(f'Bases match::\t{basis_match}')
    print(f'Expected key:\t{expected_key}')
    print(f'Actual key:\t{obtained_key}')


if __name__ == "__main__":
    main()
