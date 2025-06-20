# pylint: disable=wrong-or-nonexistent-copyright-notice
# -*- coding: utf-8 -*-
"""Creates and simulates basic arithmetic circuits

=== EXAMPLE OUTPUT ===
Execute Adder
0: ───────────@───────────────────────────────────@───────────────@───
              │                                   │               │
1: ───@───@───┼───────────────────────────────────┼───@───@───@───┼───
      │   │   │                                   │   │   │   │   │
2: ───@───X───@───────────────────────────────────@───X───@───X───X───
      │       │                                   │       │
3: ───X───────X───────@───────@───────────────@───X───────X───────────
                      │       │               │
4: ───────────@───@───┼───────┼───@───@───@───┼───────────────────────
              │   │   │       │   │   │   │   │
5: ───────────@───X───@───────@───X───@───X───X───────────────────────
              │       │       │       │
6: ───────────X───────X───@───X───────X───────────────────────────────
                          │
7: ───────────────────@───┼───────────────────────────────────────────
                      │   │
8: ───────────────────X───X───────────────────────────────────────────
000 + 000 = 000
000 + 001 = 001
000 + 010 = 010
000 + 011 = 011
001 + 000 = 001
001 + 001 = 010
001 + 010 = 011
001 + 011 = 100
010 + 000 = 010
010 + 001 = 011
010 + 010 = 100
010 + 011 = 101
011 + 000 = 011
011 + 001 = 100
011 + 010 = 101
011 + 011 = 110

Execute Multiplier
0: ─────────────Adder:0───────────────────────Adder:0───────────────Adder:0─────
                │                             │                     │
1: ─X───────────Adder:1───X───────────────────Adder:1───────────────Adder:1─────
    │           │         │                   │                     │
2: ─┼───────────Adder:2───┼───────────────────Adder:2───────────────Adder:2─────
    │           │         │                   │                     │
3: ─┼───────────Adder:3───┼───────────────────Adder:3───────────────Adder:3─────
    │           │         │                   │                     │
4: ─┼───X───────Adder:4───┼───X───────X───────Adder:4───X───────────Adder:4─────
    │   │       │         │   │       │       │         │           │
5: ─┼───┼───────Adder:5───┼───┼───────┼───────Adder:5───┼───────────Adder:5─────
    │   │       │         │   │       │       │         │           │
6: ─┼───┼───────Adder:6───┼───┼───────┼───────Adder:6───┼───────────Adder:6─────
    │   │       │         │   │       │       │         │           │
7: ─┼───┼───X───Adder:7───┼───┼───X───┼───X───Adder:7───┼───X───X───Adder:7───X─
    │   │   │   │         │   │   │   │   │   │         │   │   │   │         │
8: ─┼───┼───┼───Adder:8───┼───┼───┼───┼───┼───Adder:8───┼───┼───┼───Adder:8───┼─
    │   │   │             │   │   │   │   │             │   │   │             │
9: ─@───┼───┼─────────────@───┼───┼───@───┼─────────────@───┼───@─────────────@─
    │   │   │             │   │   │   │   │             │   │   │             │
10:─┼───@───┼─────────────┼───@───┼───┼───@─────────────┼───@───┼─────────────┼─
    │   │   │             │   │   │   │   │             │   │   │             │
11:─┼───┼───@─────────────┼───┼───@───┼───┼─────────────┼───┼───┼─────────────┼─
    │   │   │             │   │   │   │   │             │   │   │             │
12:─@───@───@─────────────@───@───@───┼───┼─────────────┼───┼───┼─────────────┼─
                                      │   │             │   │   │             │
13:───────────────────────────────────@───@─────────────@───@───┼─────────────┼─
                                                                │             │
14:─────────────────────────────────────────────────────────────@─────────────@─
000 * 000 = 000
000 * 001 = 000
000 * 010 = 000
000 * 011 = 000
001 * 000 = 000
001 * 001 = 001
001 * 010 = 010
001 * 011 = 011
010 * 000 = 000
010 * 001 = 010
010 * 010 = 100
010 * 011 = 110
011 * 000 = 000
011 * 001 = 011
011 * 010 = 110
011 * 011 = 001
"""

from __future__ import annotations

import cirq


class Adder(cirq.Gate):
    """A quantum circuit to calculate a + b

            -----------@---             ---@------------
                       |                   |
            ---@---@---+---             ---+---@---@---         -------@---
    [Carry]:   |   |   |      [Uncarry]:   |   |   |                   |
            ---@---X---@---             ---@---X---@---   [Sum]:---@---+---
               |       |                   |       |               |   |
            ---X-------X---             ---X-------X---         ---X---X---


           -----                                      -------    ---
    c0: --|     |------------------------------------|       |--|   |-----
          |     |                                    |       |  |   |
    a0: --|     |------------------------------------|       |--|Sum|-----
          |Carry|                                    |Uncarry|  |   |
    b0: --|     |------------------------------------|       |--|   |--M--
          |     |   -----           -------    ---   |       |   ---
    c1: --|     |--|     |---------|       |--|   |--|       |------------
           -----   |     |         |       |  |   |   -------
    a1: -----------|     |---------|       |--|Sum|-----------------------
                   |Carry|         |Uncarry|  |   |
    b1: -----------|     |---------|       |--|   |--------------------M--
                   |     |   ---   |       |   ---
    c2: -----------|     |--|   |--|       |------------------------------
                    -----   |   |   -------
    a2: --------------------|Sum|-----------------------------------------
                            |   |
    b2: --------------------|   |--------------------------------------M--
                             ---
    """

    def __init__(self, num_qubits):
        super(Adder, self)
        self._num_qubits = num_qubits

    def num_qubits(self):
        return self._num_qubits

    def carry(self, *qubits):
        c0, a, b, c1 = qubits
        yield cirq.TOFFOLI(a, b, c1)
        yield cirq.CNOT(a, b)
        yield cirq.TOFFOLI(c0, b, c1)

    def uncarry(self, *qubits):
        c0, a, b, c1 = qubits
        yield cirq.TOFFOLI(c0, b, c1)
        yield cirq.CNOT(a, b)
        yield cirq.TOFFOLI(a, b, c1)

    def carry_sum(self, *qubits):
        c0, a, b = qubits
        yield cirq.CNOT(a, b)
        yield cirq.CNOT(c0, b)

    def _decompose_(self, qubits):
        n = int(len(qubits) / 3)
        c = qubits[0::3]
        a = qubits[1::3]
        b = qubits[2::3]
        for i in range(n - 1):
            yield self.carry(c[i], a[i], b[i], c[i + 1])
        yield self.carry_sum(c[n - 1], a[n - 1], b[n - 1])
        for i in range(n - 2, -1, -1):
            yield self.uncarry(c[i], a[i], b[i], c[i + 1])
            yield self.carry_sum(c[i], a[i], b[i])


class Multiplier(cirq.Gate):
    """A quantum circuit to calculate y * x

                       -                         -                 -
    c0: --------------| |-----------------------| |---------------| |---------
                      | |                       | |               | |
    a0: --X-----------| |---X-------------------| |---------------| |---------
          |           | |   |                   | |               | |
    b0: --+-----------| |---+-------------------| |---------------| |------M--
          |           | |   |                   | |               | |
    c1: --+-----------|A|---+-------------------|A|---------------|A|---------
          |           |d|   |                   |d|               |d|
    a1: --+---X-------|d|---+---X-------X-------|d|---X-----------|d|---------
          |   |       |e|   |   |       |       |e|   |           |e|
    b1: --+---+-------|r|---+---+-------+-------|r|---+-----------|r|------M--
          |   |       | |   |   |       |       | |   |           | |
    c2: --+---+-------| |---+---+-------+-------| |---+-----------| |---------
          |   |       | |   |   |       |       | |   |           | |
    a2: --+---+---X---| |---+---+---X---+---X---| |---+---X---X---| |---X-----
          |   |   |   | |   |   |   |   |   |   | |   |   |   |   | |   |
    b3: --+---+---+---| |---+---+---+---+---+---| |---+---+---+---| |---+--M--
          |   |   |    -    |   |   |   |   |    -    |   |   |    -    |
    y0: --@---+---+---------@---+---+---@---+---------@---+---@---------@-----
          |   |   |         |   |   |   |   |         |   |   |         |
    y1: --+---@---+---------+---@---+---+---@---------+---@---+---------+-----
          |   |   |         |   |   |   |   |         |   |   |         |
    y2: --+---+---@---------+---+---@---+---+---------+---+---+---------+-----
          |   |   |         |   |   |   |   |         |   |   |         |
    x0: --@---@---@---------@---@---@---+---+---------+---+---+---------+-----
                                        |   |         |   |   |         |
    x1: --------------------------------@---@---------@---@---+---------+-----
                                                              |         |
    x2: ------------------------------------------------------@---------@-----
    """

    def __init__(self, num_qubits):
        super(Multiplier, self)
        self._num_qubits = num_qubits

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        n = int(len(qubits) / 5)
        # c = qubits[0:n*3:3]
        a = qubits[1 : n * 3 : 3]
        # b = qubits[2::3]
        y = qubits[n * 3 : n * 4]
        x = qubits[n * 4 :]

        for i, x_i in enumerate(x):
            # a = (y*(2**i))*x_i
            for a_qubit, y_qubit in zip(a[i:], y[: n - i]):
                yield cirq.TOFFOLI(x_i, y_qubit, a_qubit)
            # b += a
            yield Adder(3 * n).on(*qubits[: 3 * n])
            # a = 0
            for a_qubit, y_qubit in zip(a[i:], y[: n - i]):
                yield cirq.TOFFOLI(x_i, y_qubit, a_qubit)


def init_qubits(x_bin, *qubits):
    for x, qubit in zip(x_bin, list(qubits)[::-1]):
        if x == '1':
            yield cirq.X(qubit)


def experiment_adder(p, q, n=3):
    a_bin = f'{p:08b}'[-n:]
    b_bin = f'{q:08b}'[-n:]
    qubits = cirq.LineQubit.range(3 * n)
    # c = qubits[0::3]
    a = qubits[1::3]
    b = qubits[2::3]
    circuit = cirq.Circuit(
        init_qubits(a_bin, *a),
        init_qubits(b_bin, *b),
        Adder(n * 3).on(*qubits),
        cirq.measure(*b, key='result'),
    )
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1).measurements['result']
    sum_bin = ''.join(result[0][::-1].astype(int).astype(str))
    print(f'{a_bin} + {b_bin} = {sum_bin}')


def experiment_multiplier(p, q, n=3):
    y_bin = f'{p:08b}'[-n:]
    x_bin = f'{q:08b}'[-n:]
    qubits = cirq.LineQubit.range(5 * n)
    # c = qubits[0:n*3:3]
    # a = qubits[1:n*3:3]
    b = qubits[2 : n * 3 : 3]
    y = qubits[n * 3 : n * 4]
    x = qubits[n * 4 :]

    circuit = cirq.Circuit(
        init_qubits(x_bin, *x),
        init_qubits(y_bin, *y),
        Multiplier(5 * n).on(*qubits),
        cirq.measure(*b, key='result'),
    )
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1)
    sum_bin = ''.join(result.measurements['result'][0][::-1].astype(int).astype(str))
    print(f'{y_bin} * {x_bin} = {sum_bin}')


def main(n=3):
    print('Execute Adder')
    print(cirq.Circuit(cirq.decompose(Adder(3 * n).on(*cirq.LineQubit.range(3 * n)))))
    for p in range(2 * 2):
        for q in range(2 * 2):
            experiment_adder(p, q, n)
    print('')
    print('Execute Multiplier')
    print(cirq.Circuit(cirq.decompose(Multiplier(5 * n).on(*cirq.LineQubit.range(5 * n)))))
    for p in range(2 * 2):
        for q in range(2 * 2):
            experiment_multiplier(p, q, n)


if __name__ == '__main__':
    main()
