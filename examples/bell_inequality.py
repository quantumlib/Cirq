"""
Bell's theorem or inequality proves that entanglement based
observations can't be reproduced with any local realist theory [1].

This example shows Bell's inequality in form of CHSH game where two
players Alice and Bob receive an input bit x and y respectively and
produce an output a and b based on the input bit.
The goal is to maximize the probability to satisfy the condition [2]:
    a XOR b = x AND y

In the classical deterministic case, the highest probability
achievable is 75%. While with quantum correlations, it can
achieve higher success probability. In the quantum case, two players
Alice and Bob start with a shared Bell-pair entangled state. The
random input x and y is provided by referee for Alice and Bob. The
success probability of satisfying the above condition will be
cos(theta/2)^2 if Alice and Bob measure their entangled qubit in
measurement basis V and W where angle between V and W is theta.
Therefore, maximum success probability is cos(pi/8)^2 ~ 85.3%
when theta = pi/4.

In the usual implementation [2], Alice and Bob share the Bell state
with the same value and opposite phase. If the input x (y) is 0, Alice (Bob)
rotates in Y-basis by angle -pi/16 and if the input is 1, Alice (Bob)
rotates by angle 3pi/16. Here, Alice and Bob start with the entangled
Bell state with same value and phase. The same success probability is
achieved by following procedure: Alice rotate in X-basis by angle
-pi/4 followed by controlled-rotation by angle pi/2 in X-basis for
Alice (Bob) based on input x (y).

[1] https://en.wikipedia.org/wiki/Bell%27s_theorem
[2] R. de Wolf. Quantum Computing: Lecture Notes
(arXiv:1907.09415, Section 15.2)

=== EXAMPLE OUTPUT ===
Circuit:
(0, 0): ───H───@───X^-0.25───X────────M('a')───
               │             │
(0, 1): ───H───┼─────────────@^0.5────M('x')───
               │
(1, 0): ───────X───X─────────M('b')────────────
                   │
(1, 1): ───H───────@^0.5─────M('y')────────────

Simulating 75 repetitions...

Results
a: _1___1_1_1111__11__11______11__1______111_______1_______11___11_11__1_1_1_1
b: _1_______11_1_1__1_1111_11_11_1_1_____11___1__111__1_1_1_1__111_11_11_1_1_1
x: 1_1____1______11_1_1_1_11111___111_1__1_1__11_111__1_11_11_11______1____1__
y: ____1__111_______1___11_111__1_111______111___11_11_11__1_1_1111_1111__1_11
(a XOR b) == (x AND y):
   11111_11111_11___11111_111_111_11_111111111_1111111_111_1111111111111111111
Win rate: 84.0%
"""

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
    result = cirq.Simulator().run(program=circuit, repetitions=repetitions)

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
    alice = cirq.GridQubit(0, 0)
    bob = cirq.GridQubit(1, 0)
    alice_referee = cirq.GridQubit(0, 1)
    bob_referee = cirq.GridQubit(1, 1)

    circuit = cirq.Circuit()

    # Prepare shared entangled state.
    circuit.append(
        [
            cirq.H(alice),
            cirq.CNOT(alice, bob),
        ]
    )

    # Referees flip coins.
    circuit.append(
        [
            cirq.H(alice_referee),
            cirq.H(bob_referee),
        ]
    )

    # Players do a sqrt(X) based on their referee's coin.
    circuit.append(
        [
            cirq.X(alice) ** -0.25,
            cirq.CNOT(alice_referee, alice) ** 0.5,
            cirq.CNOT(bob_referee, bob) ** 0.5,
        ]
    )

    # Then results are recorded.
    circuit.append(
        [
            cirq.measure(alice, key='a'),
            cirq.measure(bob, key='b'),
            cirq.measure(alice_referee, key='x'),
            cirq.measure(bob_referee, key='y'),
        ]
    )

    return circuit


def bitstring(bits):
    return ''.join('1' if e else '_' for e in bits)


if __name__ == '__main__':
    main()
