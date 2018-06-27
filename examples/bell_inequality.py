"""Creates and simulates a circuit equivalent to a Bell inequality test.

=== EXAMPLE OUTPUT ===

Circuit:
(0, 0): ───H───@───X^-0.25───────X^0.5───M───
               │                 │
(0, 1): ───────┼─────────────H───@───────M───
               │
(1, 0): ───────X─────────────────X^0.5───M───
                                 │
(1, 1): ─────────────────────H───@───────M───

Simulating 75 repetitions...

Results
a: _1__11111___111111_1__1_11_11__111________1___1_111_111_____1111_1_111__1_1
b: _1__1__11_1_1_1__1_1_1_1_11____111_111__11_____1_1__1111_1_11___111_11_1__1
x: ____11______11111__1_1111111____1_111__1111___1111_1__11_1__11_11_11_1__11_
y: 11_111_____1_1_111__11111_111_1____1____11_____11___11_1_1___1_111111_1_1_1
(a XOR b) == (x AND y):
   1111_1_111_11111111111111111_1111111__1111_111_111_11111111_11_11111111_111
Win rate: 84.0%
"""

from typing import Union, Iterable

import numpy as np

import cirq


def main():
    # Create circuit.
    circuit = make_bell_test_circuit()
    print('Circuit:')
    print(circuit)

    # Run simulations.
    print()
    repetitions = 75
    print('Simulating {} repetitions...'.format(repetitions))
    result = cirq.google.XmonSimulator().run(circuit=circuit,
                                             repetitions=repetitions)

    # Collect results.
    a = np.array(result.measurements['a'][:, 0])
    b = np.array(result.measurements['b'][:, 0])
    x = np.array(result.measurements['x'][:, 0])
    y = np.array(result.measurements['y'][:, 0])
    outcomes = a ^ b == x & y
    win_percent = len([e for e in outcomes if e]) * 100 / repetitions

    # Print data.
    print()
    print('Results')
    print('a:', bitstring(a))
    print('b:', bitstring(b))
    print('x:', bitstring(x))
    print('y:', bitstring(y))
    print('(a XOR b) == (x AND y):\n  ', bitstring(outcomes))
    print('Win rate: {}%'.format(win_percent))


def make_bell_test_circuit():
    alice = cirq.google.XmonQubit(0, 0)
    bob = cirq.google.XmonQubit(1, 0)
    alice_referee = cirq.google.XmonQubit(0, 1)
    bob_referee = cirq.google.XmonQubit(1, 1)

    circuit = cirq.Circuit()

    # Prepare shared entangled state.
    circuit.append([
        cirq.H(alice),
        cirq.CNOT(alice, bob),
        cirq.X(alice)**-0.25,
    ])

    # Referees flip coins.
    circuit.append([
        cirq.H(alice_referee),
        cirq.H(bob_referee),
    ])

    # Players do a sqrt(X) based on their referee's coin.
    circuit.append([
        cirq.CNOT(alice_referee, alice)**0.5,
        cirq.CNOT(bob_referee, bob)**0.5,
    ])

    # Then results are recorded.
    circuit.append([
        cirq.measure(alice, key='a'),
        cirq.measure(bob, key='b'),
        cirq.measure(alice_referee, key='x'),
        cirq.measure(bob_referee, key='y'),
    ])

    return circuit


def bitstring(bits: Iterable[Union[bool, int]]) -> str:
    return ''.join('1' if e else '_' for e in bits)


if __name__ == '__main__':
    main()
