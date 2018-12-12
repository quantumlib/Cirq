"""Demonstrates the algorithm for solving linear systems by Harrow, Hassidim,
Lloyd (HHL).

The HHL algorithm solves a system of linear equations, specifically equations
of the form Ax = b, where A is a Hermitian matrix, b is a known vector, and
x is the unknown vector. To solve on a quantum system, b must be rescaled to
have magnitude 1, and the equation becomes:

|x> = A**-1 |b> / || A**-1 |b> ||

The algorithm uses 3 sets of qubits: a single ancilla qubit, a register (to
store eigenvalues of A), and memory qubits (to store |b> and |x>). The
following are performed in order:
1) Quantum phase estimation to extract eigenvalues of A
2) Controlled rotations of ancilla qubit
3) Uncomputation with inverse quantum phase estimation

For details about the algorithm, please refer to the paper in the
REFERENCE section below. The following description uses variables defined
in the paper.

This example is an implementation of the HHL algorithm for the equation where
A = [ 1.5  0.5 ]
    [ 0.5  1.5 ]
|b> = [ 1  0 ]^T = |0>
|x> = [ 0.948683  -0.316228 ]
Eigenvalues of A are 1 (with eigenvector |->) and 2 (with eigenvector |+>)

Because there are 2 eigenvalues, exactly 1 register qubit is needed.
As a result, t is set to π, and

exp(iAt) = exp(i(H [ 2  0 ] H)*0.5) = H [ exp(i2π)  0 ] H = HZH = X
                   [ 0  1 ]             [ 0   exp(iπ) ]

|0> for the register qubit corresponds to λ=1, and |1> corresponds to λ=2.
By setting C = 1, the rotations are Ry(π) for λ=1 and for Ry(π/3) λ=2

Instead of outputting the result directly, this example measures various
observables of the result.

=== REFERENCE ===
Coles, Eidenbenz et al. Quantum Algorithm Implementations for Beginners
https://arxiv.org/abs/1804.03719

=== CIRCUIT ===
(0, 0): ───────────────Ry(π)───────Ry(0.333π)───────────────────M───
                       │           │                            │
(1, 0): ───H───@───H───@───────X───@────────────X───H───@───H───┼───
               │                                        │       │
(2, 0): ───────X────────────────────────────────────────X───────M───

"""
import math
import cirq


def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)


def expectation(frequencies):
    # Post select on ancilla qubit = 1
    post_select = {k[1]: v for k, v in frequencies.items() if k[0] == '1'}

    s = sum(post_select.values())
    expected_value = 0
    for output, occurrence in post_select.items():
        v = 1 if output == '0' else -1
        expected_value += v * occurrence / s
    return expected_value


def hhl_circuit(measure_key, observable):
    ancilla_qubit = cirq.GridQubit(0, 0)
    register_qubit = cirq.GridQubit(1, 0) # to store eigenvalues of the matrix
    memory_qubit = cirq.GridQubit(2, 0) # to store input and output vectors
    c = cirq.Circuit()

    # Uncomment 2nd line to set |b> = |+>
    # Uncomment both lines to set |b> = |->
    # c.append(cirq.X(memory_qubit))
    # c.append(cirq.H(memory_qubit))

    # Phase estimation
    c.append([
        cirq.H(register_qubit),
        cirq.CNOT(register_qubit, memory_qubit),
        cirq.H(register_qubit),
    ])

    # Ry(π) for λ=1
    c.append([
        cirq.ControlledGate(cirq.Ry(math.pi))(register_qubit, ancilla_qubit),
    ])

    # Ry(π/3) for λ=2
    c.append([
        cirq.X(register_qubit),
        cirq.ControlledGate(cirq.Ry(math.pi/3))(register_qubit, ancilla_qubit),
        cirq.X(register_qubit),
    ])

    # Inverse phase estimation
    c.append([
        cirq.H(register_qubit),
        cirq.CNOT(register_qubit, memory_qubit),
        cirq.H(register_qubit),
    ])

    # Preparing the observable
    if observable == 'X':
        c.append((cirq.H)(memory_qubit))  # X observable
    elif observable == 'Y':
        c.append((cirq.X**-0.5)(memory_qubit))  # Y observable
    # otherwise assume observable == 'Z'

    c.append(cirq.measure(ancilla_qubit, memory_qubit, key=measure_key))

    return c


def main():
    trial_count_per_observable = 10000
    mkey = 'result'

    # Expected observables:
    # <X> = -0.6
    # <Y> = 0.0
    # <Z> = 0.8
    simulator = cirq.Simulator()
    observables = ['X', 'Y', 'Z']

    print('''
A = [ 1.5  0.5 ]
    [ 0.5  1.5 ]
b = [ 1  0 ]
''')

    for o in observables:
        result = simulator.run(
            hhl_circuit(mkey, o),
            repetitions=trial_count_per_observable)
        frequencies = result.histogram(key=mkey, fold_func=bitstring)
        print('<{}>: {}'.format(o, expectation(frequencies)))


if __name__ == '__main__':
    main()
