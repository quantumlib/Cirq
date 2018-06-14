# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code snippets for the Circuits concepts page."""

import cirq
import cirq.google


def test_intro_snippets(capsys):
    # [START cirq_circuits_initialize_grid]
    qubits = [cirq.google.XmonQubit(x, y) for x in range(3) for y in range(3)]

    print(qubits[0])
    # prints "(0, 0)"
    # [END cirq_circuits_initialize_grid]
    captured = capsys.readouterr()
    assert captured.out == '(0, 0)\n'

    # [START cirq_circuits_gate]
    # This is an Pauli X gate.
    x_gate = cirq.X
    # Applying it to the qubit at location (0, 0) (defined above)
    # turns it into an operation.
    x_op = x_gate(qubits[0])

    print(x_op)
    # prints "X((0, 0))"
    # [END cirq_circuits_gate]
    captured = capsys.readouterr()
    assert captured.out == 'X((0, 0))\n'

    # [START cirq_circuits_moment]
    cz = cirq.CZ(qubits[0], qubits[1])
    x = cirq.X(qubits[2])
    moment = cirq.Moment([x, cz])

    print(moment)
    # prints "X((0, 2)) and CZ((0, 0), (0, 1))"
    # [END cirq_circuits_moment]
    captured = capsys.readouterr()
    assert captured.out == 'X((0, 2)) and CZ((0, 0), (0, 1))\n'

    # [START cirq_circuits_moment_series]
    cz01 = cirq.CZ(qubits[0], qubits[1])
    x2 = cirq.X(qubits[2])
    cz12 = cirq.CZ(qubits[1], qubits[2])
    moment0 = cirq.Moment([cz01, x2])
    moment1 = cirq.Moment([cz12])
    circuit = cirq.Circuit((moment0, moment1))

    print(circuit)
    # prints the text diagram for the circuit:
    # (0, 0): ───@───────
    #            │
    # (0, 1): ───Z───@───
    #                │
    # (0, 2): ───X───Z───
    # [END cirq_circuits_moment_series]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        '(0, 0): ───@───────',
        '           │',
        '(0, 1): ───Z───@───',
        '               │',
        '(0, 2): ───X───Z───',
    )) + '\n'


def test_circuit_construction_snippets(capsys):
    # [START cirq_circuits_construction_append]
    from cirq.ops import CZ, H
    q0, q1, q2 = [cirq.google.XmonQubit(i, 0) for i in range(3)]
    circuit = cirq.Circuit()
    circuit.append([CZ(q0, q1), H(q2)])

    print(circuit)
    # prints
    # (0, 0): ───@───
    #            │
    # (1, 0): ───Z───
    #
    # (2, 0): ───H───
    # [END cirq_circuits_construction_append]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        '(0, 0): ───@───',
        '           │',
        '(1, 0): ───Z───',
        '',
        '(2, 0): ───H───',
    )) + '\n'

    # [START cirq_circuits_construction_append_more]
    circuit.append([H(q0), CZ(q1, q2)])

    print(circuit)
    # prints
    # (0, 0): ───@───H───
    #            │
    # (1, 0): ───Z───@───
    #                │
    # (2, 0): ───H───Z───
    # [END cirq_circuits_construction_append_more]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        '(0, 0): ───@───H───',
        '           │',
        '(1, 0): ───Z───@───',
        '               │',
        '(2, 0): ───H───Z───',
    )) + '\n'

    # [START cirq_circuits_construction_append_all]
    circuit = cirq.Circuit()
    circuit.append([CZ(q0, q1), H(q2), H(q0), CZ(q1, q2)])

    print(circuit)
    # prints
    # (0, 0): ───@───H───
    #            │
    # (1, 0): ───Z───@───
    #                │
    # (2, 0): ───H───Z───
    # [END cirq_circuits_construction_append_all]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        '(0, 0): ───@───H───',
        '           │',
        '(1, 0): ───Z───@───',
        '               │',
        '(2, 0): ───H───Z───',
    )) + '\n'


def test_insert_strategy_snippets(capsys):
    from cirq.ops import CZ, H
    q0, q1, q2 = [cirq.google.XmonQubit(i, 0) for i in range(3)]

    # [START cirq_circuits_insert_strategy_earliest]
    from cirq.circuits import InsertStrategy
    circuit = cirq.Circuit()
    circuit.append([CZ(q0, q1)])
    circuit.append([H(q0), H(q2)], strategy=InsertStrategy.EARLIEST)

    print(circuit)
    # prints
    # (0, 0): ───@───H───
    #            │
    # (1, 0): ───Z───────
    #
    # (2, 0): ───H───────
    # [END cirq_circuits_insert_strategy_earliest]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        '(0, 0): ───@───H───',
        '           │',
        '(1, 0): ───Z───────',
        '',
        '(2, 0): ───H───────',
    )) + '\n'

    # [START cirq_circuits_insert_strategy_new]
    circuit = cirq.Circuit()
    circuit.append([H(q0), H(q1), H(q2)], strategy=InsertStrategy.NEW)

    print(circuit)
    # prints
    # (0, 0): ───H───────────
    #
    # (1, 0): ───────H───────
    #
    # (2, 0): ───────────H───
    # [END cirq_circuits_insert_strategy_new]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        '(0, 0): ───H───────────',
        '',
        '(1, 0): ───────H───────',
        '',
        '(2, 0): ───────────H───',
    )) + '\n'

    # [START cirq_circuits_insert_strategy_inline]
    circuit = cirq.Circuit()
    circuit.append([CZ(q1, q2)])
    circuit.append([CZ(q0, q1), H(q2), H(q0)], strategy=InsertStrategy.INLINE)

    print(circuit)
    # prints
    # (0, 0): ───────@───H───
    #                │
    # (1, 0): ───@───Z───────
    #            │
    # (2, 0): ───Z───H───────
    # [END cirq_circuits_insert_strategy_inline]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        '(0, 0): ───────@───H───',
        '               │',
        '(1, 0): ───@───Z───────',
        '           │',
        '(2, 0): ───Z───H───────',
    )) + '\n'

    # [START cirq_circuits_insert_strategy_new_then_inline]
    circuit = cirq.Circuit()
    circuit.append([H(q0)])
    circuit.append([CZ(q1, q2), H(q0)],
                   strategy=InsertStrategy.NEW_THEN_INLINE)

    print(circuit)
    # prints
    # (0, 0): ───H───H───
    #
    # (1, 0): ───────@───
    #                │
    # (2, 0): ───────Z───
    # [END cirq_circuits_insert_strategy_new_then_inline]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        '(0, 0): ───H───H───',
        '',
        '(1, 0): ───────@───',
        '               │',
        '(2, 0): ───────Z───',
    )) + '\n'


def test_op_tree(capsys):
    from cirq.ops import CZ, H
    q0, q1, q2 = [cirq.google.XmonQubit(i, 0) for i in range(3)]

    # [START cirq_circuits_op_tree]
    def my_layer():
        yield CZ(q0, q1)
        yield [H(q) for q in (q0, q1, q2)]
        yield [CZ(q1, q2)]
        yield [H(q0), [CZ(q1, q2)]]

    circuit = cirq.Circuit()
    circuit.append(my_layer())

    for x in my_layer():
        print(x)
    # prints
    # CZ((0, 0), (1, 0))
    # [Operation(H, (XmonQubit(0, 0),)), Operation(H, (XmonQubit(1, 0),)), \
    #  Operation(H, (XmonQubit(2, 0),))]
    # [Operation(CZ, (XmonQubit(1, 0), XmonQubit(2, 0)))]
    # [Operation(H, (XmonQubit(0, 0),)), [Operation(CZ, (XmonQubit(1, 0), \
    #  XmonQubit(2, 0)))]]

    print(circuit)
    # prints
    # (0, 0): ───@───H───H───────
    #            │
    # (1, 0): ───Z───H───@───@───
    #                    │   │
    # (2, 0): ───────H───Z───Z───
    # [END cirq_circuits_op_tree]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        'CZ((0, 0), (1, 0))',
        (
            '[Operation(H, (XmonQubit(0, 0),)), '
            'Operation(H, (XmonQubit(1, 0),)), '
            'Operation(H, (XmonQubit(2, 0),))]'
        ),
        '[Operation(CZ, (XmonQubit(1, 0), XmonQubit(2, 0)))]',
        (
            '[Operation(H, (XmonQubit(0, 0),)), '
            '[Operation(CZ, (XmonQubit(1, 0), '
            'XmonQubit(2, 0)))]]'
        ),
        '(0, 0): ───@───H───H───────',
        '           │',
        '(1, 0): ───Z───H───@───@───',
        '                   │   │',
        '(2, 0): ───────H───Z───Z───',
    )) + '\n'


def test_from_ops(capsys):
    from cirq.ops import H
    q0, q1 = [cirq.google.XmonQubit(i, 0) for i in range(2)]

    # [START cirq_circuits_from_ops]
    circuit = cirq.Circuit.from_ops(H(q0), H(q1))
    print(circuit)
    # prints
    # (0, 0): ───H───
    #
    # (1, 0): ───H───
    # [END cirq_circuits_from_ops]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        '(0, 0): ───H───',
        '',
        '(1, 0): ───H───',
    )) + '\n'


def test_iterate(capsys):
    from cirq.ops import CZ, H
    q0, q1 = [cirq.google.XmonQubit(i, 0) for i in range(2)]

    # [START cirq_circuits_iterate]
    circuit = cirq.Circuit.from_ops(H(q0), CZ(q0, q1))
    for moment in circuit:
        print(moment)
    # prints
    # H((0, 0))
    # CZ((0, 0), (1, 0))
    # [END cirq_circuits_iterate]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        'H((0, 0))',
        'CZ((0, 0), (1, 0))',
    )) + '\n'


def test_slice(capsys):
    from cirq.ops import CZ, H
    q0, q1 = [cirq.google.XmonQubit(i, 0) for i in range(2)]

    # [START cirq_circuits_slice]
    circuit = cirq.Circuit.from_ops(H(q0), CZ(q0, q1), H(q1), CZ(q0, q1))
    print(circuit[1:3])
    # prints
    # (0, 0): ───@───────
    #            │
    # (1, 0): ───Z───H───
    # [END cirq_circuits_slice]
    captured = capsys.readouterr()
    assert captured.out == '\n'.join((
        '(0, 0): ───@───────',
        '           │',
        '(1, 0): ───Z───H───',
    )) + '\n'
