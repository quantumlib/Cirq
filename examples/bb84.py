# Example program to demonstrate BB84 QKD Protocol

################ Example output

# 0: ────X───H───H───M───

# 1: ────M───────────────

# 2: ────X───H───M───────

# 3: ────H───M───────────

# 4: ────X───H───M───────

# 5: ────H───M───────────

# 6: ────M───────────────

# 7: ────X───H───H───M───

# 8: ────X───H───M───────

# 9: ────M───────────────

# 10: ───H───H───M───────

# 11: ───X───H───H───M───

# 12: ───X───M───────────

# 13: ───H───M───────────

# 14: ───H───H───M───────

# 15: ───X───M───────────
# Simulating 5 repetitions...
# Alice's basis:  HCCCHCCHHCHHCHHC
# Bob's basis:    HCHHCHCHCCHHCCHC
# Alice's bits:   1010100110011001
# Bases match::   XX____XX_XXXX_XX
# Expected key:   10____01_0011_01
# Actual key:     1001001101

################

import numpy as np
import cirq


def main(num_qubits=16):

    alice_basis = np.random.randint(2, size=num_qubits)
    alice_state = np.random.randint(2, size=num_qubits)
    bob_basis = np.random.randint(2, size=num_qubits)

    circuit = make_bb84_circ(num_qubits, alice_basis, bob_basis, alice_state)
    print(circuit)

    # Run simulations.
    repetitions = 5
    print('Simulating {} repetitions...'.format(repetitions))
    result = cirq.Simulator().run(program=circuit, repetitions=repetitions)

    # Take only qubits where bases match
    keys = []
    for i in range(num_qubits):
        if alice_basis[i] == bob_basis[
                i]:  # Only choose bits where Alice and Bob chose the same basis
            keys.append(result.measurements[str(i)][:, 0])

    keys = np.array(keys, dtype='int64')
    key_bitstr = [
        np.array2string(keys[:, i], separator='')[1:-1]
        for i in range(repetitions)
    ]

    print_results(alice_basis, bob_basis, alice_state, key_bitstr)


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


def print_results(alice_basis, bob_basis, alice_state, key_bitstr):
    basis_match = ''
    key_expected = ''
    for i, char in enumerate(alice_state):
        basis_match += 'X' if alice_basis[i] == bob_basis[i] else '_'
        key_expected += str(char) if alice_basis[i] == bob_basis[i] else '_'

    print(
        f'Alice\'s basis:\t{"".join(np.where(alice_basis==0, "C","H"))}'
    )
    print(
        f'Bob\'s basis:\t{"".join(np.where(bob_basis==0, "C", "H"))}')
    print(
        f'Alice\'s bits:\t{np.array2string(alice_state, separator="")[1:-1]}'
    )
    print(f'Bases match::\t{basis_match}')
    print(f'Expected key:\t{key_expected}')
    print(f'Actual key:\t{np.unique(key_bitstr)[0]}')


if __name__ == "__main__":
    main()
