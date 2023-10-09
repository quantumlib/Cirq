# Copyright 2023 The Cirq Developers
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

import itertools

import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools


def identity_map(n: int):
    """Returns a dict of size `2**n` mapping each integer in range [0, 2**n) to itself."""
    return {i: i for i in range(2**n)}


def test_less_than_gate():
    qubits = cirq.LineQubit.range(4)
    gate = cirq_ft.LessThanGate(3, 5)
    op = gate.on(*qubits)
    circuit = cirq.Circuit(op)
    basis_map = {
        0b_000_0: 0b_000_1,
        0b_000_1: 0b_000_0,
        0b_001_0: 0b_001_1,
        0b_001_1: 0b_001_0,
        0b_010_0: 0b_010_1,
        0b_010_1: 0b_010_0,
        0b_011_0: 0b_011_1,
        0b_011_1: 0b_011_0,
        0b_100_0: 0b_100_1,
        0b_100_1: 0b_100_0,
        0b_101_0: 0b_101_0,
        0b_101_1: 0b_101_1,
        0b_110_0: 0b_110_0,
        0b_110_1: 0b_110_1,
        0b_111_0: 0b_111_0,
        0b_111_1: 0b_111_1,
    }
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
    circuit += op**-1
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(len(qubits)), circuit)
    gate2 = cirq_ft.LessThanGate(4, 10)
    assert gate.with_registers(*gate2.registers()) == gate2
    assert cirq.circuit_diagram_info(gate).wire_symbols == ("In(x)",) * 3 + ("+(x < 5)",)
    assert (gate**1 is gate) and (gate**-1 is gate)
    assert gate.__pow__(2) is NotImplemented


@pytest.mark.parametrize("bits", [*range(8)])
@pytest.mark.parametrize("val", [3, 5, 7, 8, 9])
def test_decompose_less_than_gate(bits: int, val: int):
    qubit_states = list(bit_tools.iter_bits(bits, 3))
    circuit = cirq.Circuit(
        cirq.decompose_once(cirq_ft.LessThanGate(3, val).on(*cirq.LineQubit.range(4)))
    )
    if val < 8:
        initial_state = [0] * 4 + qubit_states + [0]
        output_state = [0] * 4 + qubit_states + [int(bits < val)]
    else:
        # When val >= 2**number_qubits the decomposition doesn't create any ancilla since the
        # answer is always 1.
        initial_state = [0]
        output_state = [1]
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(
        circuit, sorted(circuit.all_qubits()), initial_state, output_state
    )


@pytest.mark.parametrize("n", [*range(2, 5)])
@pytest.mark.parametrize("val", [3, 4, 5, 7, 8, 9])
def test_less_than_consistent_protocols(n: int, val: int):
    g = cirq_ft.LessThanGate(n, val)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(g)
    cirq.testing.assert_equivalent_repr(g, setup_code='import cirq_ft')
    # Test the unitary is self-inverse
    u = cirq.unitary(g)
    np.testing.assert_allclose(u @ u, np.eye(2 ** (n + 1)))


def test_multi_in_less_equal_than_gate():
    qubits = cirq.LineQubit.range(7)
    op = cirq_ft.LessThanEqualGate(3, 3).on(*qubits)
    circuit = cirq.Circuit(op)
    basis_map = {}
    for in1, in2 in itertools.product(range(2**3), repeat=2):
        for target_reg_val in range(2):
            target_bin = bin(target_reg_val)[2:]
            in1_bin = format(in1, '03b')
            in2_bin = format(in2, '03b')
            out_bin = bin(target_reg_val ^ (in1 <= in2))[2:]
            true_out_int = target_reg_val ^ (in1 <= in2)
            input_int = int(in1_bin + in2_bin + target_bin, 2)
            output_int = int(in1_bin + in2_bin + out_bin, 2)
            assert true_out_int == int(out_bin, 2)
            basis_map[input_int] = output_int

    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
    circuit += op**-1
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(len(qubits)), circuit)


@pytest.mark.parametrize("x_bitsize", [*range(1, 5)])
@pytest.mark.parametrize("y_bitsize", [*range(1, 5)])
def test_less_than_equal_consistent_protocols(x_bitsize: int, y_bitsize: int):
    g = cirq_ft.LessThanEqualGate(x_bitsize, y_bitsize)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(g)

    # Decomposition works even when context is None.
    qubits = cirq.LineQid.range(x_bitsize + y_bitsize + 1, dimension=2)
    assert cirq.Circuit(g._decompose_with_context_(qubits=qubits)) == cirq.Circuit(
        cirq.decompose_once(
            g.on(*qubits), context=cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        )
    )

    cirq.testing.assert_equivalent_repr(g, setup_code='import cirq_ft')
    # Test the unitary is self-inverse
    assert g**-1 is g
    u = cirq.unitary(g)
    np.testing.assert_allclose(u @ u, np.eye(2 ** (x_bitsize + y_bitsize + 1)))
    # Test diagrams
    expected_wire_symbols = ("In(x)",) * x_bitsize + ("In(y)",) * y_bitsize + ("+(x <= y)",)
    assert cirq.circuit_diagram_info(g).wire_symbols == expected_wire_symbols
    # Test with_registers
    assert g.with_registers([2] * 4, [2] * 5, [2]) == cirq_ft.LessThanEqualGate(4, 5)


def test_contiguous_register_gate():
    gate = cirq_ft.ContiguousRegisterGate(3, 6)
    circuit = cirq.Circuit(gate.on(*cirq.LineQubit.range(12)))
    basis_map = {}
    for p in range(2**3):
        for q in range(p):
            inp = f'0b_{p:03b}_{q:03b}_{0:06b}'
            out = f'0b_{p:03b}_{q:03b}_{(p * (p - 1))//2 + q:06b}'
            basis_map[int(inp, 2)] = int(out, 2)

    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq_ft')
    # Test the unitary is self-inverse
    gate = cirq_ft.ContiguousRegisterGate(2, 4)
    assert gate**-1 is gate
    u = cirq.unitary(gate)
    np.testing.assert_allclose(u @ u, np.eye(2 ** cirq.num_qubits(gate)))
    # Test diagrams
    expected_wire_symbols = ("In(x)",) * 2 + ("In(y)",) * 2 + ("+(x(x-1)/2 + y)",) * 4
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected_wire_symbols
    # Test with_registers
    assert gate.with_registers([2] * 3, [2] * 3, [2] * 6) == cirq_ft.ContiguousRegisterGate(3, 6)


@pytest.mark.parametrize('n', [*range(1, 10)])
def test_contiguous_register_gate_t_complexity(n):
    gate = cirq_ft.ContiguousRegisterGate(n, 2 * n)
    toffoli_complexity = cirq_ft.t_complexity(cirq.CCNOT)
    assert cirq_ft.t_complexity(gate) == (n**2 + n - 1) * toffoli_complexity


@pytest.mark.parametrize('a,b,num_bits', itertools.product(range(4), range(4), range(3, 5)))
def test_add(a: int, b: int, num_bits: int):
    num_anc = num_bits - 1
    gate = cirq_ft.AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    initial_state = [0] * (2 * num_bits + num_anc)
    initial_state[:num_bits] = list(bit_tools.iter_bits(a, num_bits))[::-1]
    initial_state[num_bits : 2 * num_bits] = list(bit_tools.iter_bits(b, num_bits))[::-1]
    final_state = [0] * (2 * num_bits + num_bits - 1)
    final_state[:num_bits] = list(bit_tools.iter_bits(a, num_bits))[::-1]
    final_state[num_bits : 2 * num_bits] = list(bit_tools.iter_bits(a + b, num_bits))[::-1]
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(
        circuit, qubits + ancillas, initial_state, final_state
    )
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq_ft')
    # Test diagrams
    expected_wire_symbols = ("In(x)",) * num_bits + ("In(y)/Out(x+y)",) * num_bits
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected_wire_symbols
    # Test with_registers
    assert gate.with_registers([2] * 6, [2] * 6) == cirq_ft.AdditionGate(6)


@pytest.mark.parametrize('bitsize', [3])
@pytest.mark.parametrize('mod', [5, 8])
@pytest.mark.parametrize('add_val', [1, 2])
@pytest.mark.parametrize('cv', [[], [0, 1], [1, 0], [1, 1]])
def test_add_mod_n(bitsize, mod, add_val, cv):
    gate = cirq_ft.AddMod(bitsize, mod, add_val=add_val, cv=cv)
    basis_map = {}
    num_cvs = len(cv)
    for x in range(2**bitsize):
        y = (x + add_val) % mod if x < mod else x
        if not num_cvs:
            basis_map[x] = y
            continue
        for cb in range(2**num_cvs):
            inp = f'0b_{cb:0{num_cvs}b}_{x:0{bitsize}b}'
            if tuple(int(x) for x in f'{cb:0{num_cvs}b}') == tuple(cv):
                out = f'0b_{cb:0{num_cvs}b}_{y:0{bitsize}b}'
                basis_map[int(inp, 2)] = int(out, 2)
            else:
                basis_map[int(inp, 2)] = int(inp, 2)

    num_qubits = gate.num_qubits()
    op = gate.on(*cirq.LineQubit.range(num_qubits))
    circuit = cirq.Circuit(op)
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
    circuit += op**-1
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(num_qubits), circuit)
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq_ft')


def test_add_mod_n_protocols():
    with pytest.raises(ValueError, match="must be between"):
        _ = cirq_ft.AddMod(3, 10)
    add_one = cirq_ft.AddMod(3, 5, 1)
    add_two = cirq_ft.AddMod(3, 5, 2, cv=[1, 0])

    assert add_one == cirq_ft.AddMod(3, 5, 1)
    assert add_one != add_two
    assert hash(add_one) != hash(add_two)
    assert add_two.cv == (1, 0)
    assert cirq.circuit_diagram_info(add_two).wire_symbols == ('@', '@(0)') + ('Add_2_Mod_5',) * 3


def test_add_truncated():
    num_bits = 3
    num_anc = num_bits - 1
    gate = cirq_ft.AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits)))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    # Corresponds to 2^2 + 2^2 (4 + 4 = 8 = 2^3 (needs num_bits = 4 to work properly))
    initial_state = [0, 0, 1, 0, 0, 1, 0, 0]
    # Should be 1000 (or 0001 below) but bit falls off the end
    final_state = [0, 0, 1, 0, 0, 0, 0, 0]
    # increasing number of bits yields correct value
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)

    num_bits = 4
    num_anc = num_bits - 1
    gate = cirq_ft.AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    initial_state = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    final_state = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)

    num_bits = 3
    num_anc = num_bits - 1
    gate = cirq_ft.AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    # Corresponds to 2^2 + (2^2 + 2^1 + 2^0) (4 + 7 = 11 = 1011 (need num_bits=4 to work properly))
    initial_state = [0, 0, 1, 1, 1, 1, 0, 0]
    # Should be 1011 (or 1101 below) but last two bits are lost
    final_state = [0, 0, 1, 1, 1, 0, 0, 0]
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


@pytest.mark.parametrize('a,b,num_bits', itertools.product(range(4), range(4), range(3, 5)))
def test_subtract(a, b, num_bits):
    num_anc = num_bits - 1
    gate = cirq_ft.AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    initial_state = [0] * (2 * num_bits + num_anc)
    initial_state[:num_bits] = list(bit_tools.iter_bits_twos_complement(a, num_bits))[::-1]
    initial_state[num_bits : 2 * num_bits] = list(
        bit_tools.iter_bits_twos_complement(-b, num_bits)
    )[::-1]
    final_state = [0] * (2 * num_bits + num_bits - 1)
    final_state[:num_bits] = list(bit_tools.iter_bits_twos_complement(a, num_bits))[::-1]
    final_state[num_bits : 2 * num_bits] = list(
        bit_tools.iter_bits_twos_complement(a - b, num_bits)
    )[::-1]
    all_qubits = qubits + ancillas
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


@pytest.mark.parametrize("n", [*range(3, 10)])
def test_addition_gate_t_complexity(n: int):
    g = cirq_ft.AdditionGate(n)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(g)
    cirq.testing.assert_equivalent_repr(g, setup_code='import cirq_ft')


@pytest.mark.parametrize('a,b', itertools.product(range(2**3), repeat=2))
def test_add_no_decompose(a, b):
    num_bits = 5
    qubits = cirq.LineQubit.range(2 * num_bits)
    op = cirq_ft.AdditionGate(num_bits).on(*qubits)
    circuit = cirq.Circuit(op)
    basis_map = {}
    a_bin = format(a, f'0{num_bits}b')
    b_bin = format(b, f'0{num_bits}b')
    out_bin = format(a + b, f'0{num_bits}b')
    true_out_int = a + b
    input_int = int(a_bin + b_bin, 2)
    output_int = int(a_bin + out_bin, 2)
    assert true_out_int == int(out_bin, 2)
    basis_map[input_int] = output_int
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)


@pytest.mark.parametrize("P,n", [(v, n) for n in range(1, 4) for v in range(1 << n)])
@pytest.mark.parametrize("Q,m", [(v, n) for n in range(1, 4) for v in range(1 << n)])
def test_decompose_less_than_equal_gate(P: int, n: int, Q: int, m: int):
    qubit_states = list(bit_tools.iter_bits(P, n)) + list(bit_tools.iter_bits(Q, m))
    circuit = cirq.Circuit(
        cirq.decompose_once(
            cirq_ft.LessThanEqualGate(n, m).on(*cirq.LineQubit.range(n + m + 1)),
            context=cirq.DecompositionContext(cirq.GreedyQubitManager(prefix='_c')),
        )
    )
    qubit_order = tuple(sorted(circuit.all_qubits()))
    num_ancillas = len(circuit.all_qubits()) - n - m - 1
    initial_state = qubit_states + [0] + [0] * num_ancillas
    output_state = qubit_states + [int(P <= Q)] + [0] * num_ancillas
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(
        circuit, qubit_order, initial_state, output_state
    )


@pytest.mark.parametrize("adjoint", [False, True])
def test_single_qubit_compare_protocols(adjoint: bool):
    g = cirq_ft.algos.SingleQubitCompare(adjoint=adjoint)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(g)
    cirq.testing.assert_equivalent_repr(g, setup_code='import cirq_ft')
    expected_side = cirq_ft.infra.Side.LEFT if adjoint else cirq_ft.infra.Side.RIGHT
    assert g.signature[2] == cirq_ft.Register('less_than', 1, side=expected_side)
    assert g.signature[3] == cirq_ft.Register('greater_than', 1, side=expected_side)

    with pytest.raises(ValueError):
        _ = g**0.5  # type: ignore

    assert g**2 == cirq.IdentityGate(4)
    assert g**1 is g
    assert g**-1 == cirq_ft.algos.SingleQubitCompare(adjoint=adjoint ^ True)


@pytest.mark.parametrize("v1,v2", [(v1, v2) for v1 in range(2) for v2 in range(2)])
def test_single_qubit_compare(v1: int, v2: int):
    g = cirq_ft.algos.SingleQubitCompare()
    qubits = cirq.LineQid.range(4, dimension=2)
    c = cirq.Circuit(g.on(*qubits))
    initial_state = [v1, v2, 0, 0]
    output_state = [v1, int(v1 == v2), int(v1 < v2), int(v1 > v2)]
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(
        c, sorted(c.all_qubits()), initial_state, output_state
    )

    # Check that g**-1 restores the qubits to their original state
    c = cirq.Circuit(g.on(*qubits), (g**-1).on(*qubits))
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(
        c, sorted(c.all_qubits()), initial_state, initial_state
    )


@pytest.mark.parametrize("adjoint", [False, True])
def test_bi_qubits_mixer_protocols(adjoint: bool):
    g = cirq_ft.algos.BiQubitsMixer(adjoint=adjoint)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(g)
    cirq.testing.assert_equivalent_repr(g, setup_code='import cirq_ft')

    assert g**1 is g
    assert g**-1 == cirq_ft.algos.BiQubitsMixer(adjoint=adjoint ^ True)


@pytest.mark.parametrize("x", [*range(4)])
@pytest.mark.parametrize("y", [*range(4)])
def test_bi_qubits_mixer(x: int, y: int):
    g = cirq_ft.algos.BiQubitsMixer()
    qubits = cirq.LineQid.range(7, dimension=2)
    c = cirq.Circuit(g.on(*qubits))
    x_1, x_0 = (x >> 1) & 1, x & 1
    y_1, y_0 = (y >> 1) & 1, y & 1
    initial_state = [x_1, x_0, y_1, y_0, 0, 0, 0]
    result = (
        cirq.Simulator()
        .simulate(c, initial_state=initial_state, qubit_order=qubits)
        .dirac_notation()[1:-1]
    )
    x_0, y_0 = int(result[1]), int(result[3])
    assert np.sign(x - y) == np.sign(x_0 - y_0)

    # Check that g**-1 restores the qubits to their original state
    c = cirq.Circuit(g.on(*qubits), (g**-1).on(*qubits))
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(
        c, sorted(c.all_qubits()), initial_state, initial_state
    )
