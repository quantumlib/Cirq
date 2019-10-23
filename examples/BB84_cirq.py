# Example program to demonstrate BB84 QKD Protocol

################ Example output

    # 0: ────H───H───M───────

    # 1: ────X───M───────────

    # 2: ────X───H───M───────

    # 3: ────M───────────────

    # 4: ────H───M───────────

    # 5: ────H───H───M───────

    # 6: ────X───H───M───────

    # 7: ────X───H───M───────

    # 8: ────H───H───M───────

    # 9: ────M───────────────

    # 10: ───X───H───H───M───

    # 11: ───H───M───────────

    # 12: ───X───M───────────

    # 13: ───X───H───M───────

    # 14: ───M───────────────

    # 15: ───X───M───────────

    # 16: ───X───H───M───────

    # 17: ───X───M───────────

    # 18: ───H───M───────────

    # 19: ───H───M───────────

    # 20: ───H───M───────────

    # 21: ───H───M───────────

    # 22: ───X───H───M───────

    # 23: ───H───H───M───────
    # Simulating 5 repetitions...
    # Alice's Encoding basis: HCHCHHHHHCHCCCCCHCCHHCHH
    # Bob's Encoding basis:   HCCCCHCCHCHHCHCCCCHCCHCH
    # Alice's sent bits:      011000110010110111000010
    # Both bases Match::      XX_X_X__XXX_X_XX_X_____X
    # Expected key bits:      01_0_0__001_1_01_1_____0
    # Actual key bits:        010000110110

################

import numpy as np
import cirq

def main():

    num_qubits = 24

    alice_basis = np.random.randint(2, size=num_qubits)
    alice_state = np.random.randint(2, size=num_qubits)
    bob_basis = np.random.randint(2, size=num_qubits)

    circuit = build_bb84_circ(num_qubits, alice_basis, bob_basis, alice_state)
    print(circuit)

    # Run simulations.
    repetitions = 5
    print('Simulating {} repetitions...'.format(repetitions))
    result = cirq.Simulator().run(program=circuit, repetitions=repetitions)


    # Take only qubits where bases match
    keys = []
    for i in range(num_qubits):
        if alice_basis[i] == bob_basis[i]: # Only choose bits where Alice and Bob chose the same basis
            keys.append(result.measurements[str(i)][:,0])
                
    keys = np.array(keys, dtype='int64')
    key_bitstr = [np.array2string(keys[:,i], separator='')[1:-1] for i in range(repetitions)]

    print_results(alice_basis, bob_basis, alice_state, key_bitstr)

def build_bb84_circ(num_qubits, alice_basis, bob_basis, alice_state):


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
        basis_match += 'X' if alice_basis[i]==bob_basis[i] else '_'
        key_expected += str(char) if alice_basis[i]==bob_basis[i] else '_'

    print(f'Alice\'s Encoding basis:\t{"".join(np.where(alice_basis==0, "C","H"))}')
    print(f'Bob\'s Encoding basis:\t{"".join(np.where(bob_basis==0, "C", "H"))}')
    print(f'Alice\'s sent bits:\t{np.array2string(alice_state, separator="")[1:-1]}')
    print(f'Both bases Match::\t{basis_match}')
    print(f'Expected key bits:\t{key_expected}')
    print(f'Actual key bits:\t{np.unique(key_bitstr)[0]}')

if __name__ == "__main__":
    main()