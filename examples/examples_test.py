# pylint: disable=wrong-or-nonexistent-copyright-notice
import itertools

import networkx
import numpy as np
import pytest
import matplotlib.pyplot as plt

import cirq
import examples.basic_arithmetic
import examples.bb84
import examples.bell_inequality
import examples.bernstein_vazirani
import examples.bcs_mean_field
import examples.deutsch
import examples.grover
import examples.heatmaps
import examples.hello_qubit
import examples.hhl
import examples.hidden_shift_algorithm
import examples.noisy_simulation_example
import examples.phase_estimator
import examples.qaoa
import examples.quantum_fourier_transform
import examples.quantum_teleportation
import examples.qubit_characterizations_example
import examples.shor
import examples.simon_algorithm
import examples.superdense_coding
import examples.swap_networks
import examples.two_qubit_gate_compilation
from examples.shors_code import OneQubitShorsCode


def test_example_runs_bernstein_vazirani():
    examples.bernstein_vazirani.main(qubit_count=3)

    # Check empty oracle case. Cover both biases.
    a = cirq.NamedQubit('a')
    assert list(examples.bernstein_vazirani.make_oracle([], a, [], False)) == []
    assert list(examples.bernstein_vazirani.make_oracle([], a, [], True)) == [cirq.X(a)]


def test_example_runs_simon():
    examples.simon_algorithm.main()


def test_example_runs_hidden_shift():
    examples.hidden_shift_algorithm.main()


def test_example_runs_deutsch():
    examples.deutsch.main()


def test_example_runs_hello_qubit():
    examples.hello_qubit.main()


def test_example_runs_bell_inequality():
    examples.bell_inequality.main()


def test_example_runs_bb84():
    examples.bb84.main()


def test_example_runs_quantum_fourier_transform():
    examples.quantum_fourier_transform.main()


def test_example_runs_bcs_mean_field():
    examples.bcs_mean_field.main()


def test_example_runs_grover():
    examples.grover.main()


def test_example_runs_basic_arithmetic():
    examples.basic_arithmetic.main(n=2)


def test_example_runs_phase_estimator():
    examples.phase_estimator.main(qnums=(2,), repetitions=2)


@pytest.mark.usefixtures('closefigures')
def test_example_heatmaps():
    pytest.importorskip("cirq_google")
    plt.switch_backend('agg')
    examples.heatmaps.main()


def test_example_runs_qaoa():
    examples.qaoa.main(repetitions=10, maxiter=5, use_boolean_hamiltonian_gate=False)
    examples.qaoa.main(repetitions=10, maxiter=5, use_boolean_hamiltonian_gate=True)


def test_example_qaoa_same_unitary():
    n = 6
    p = 2

    qubits = cirq.LineQubit.range(n)

    graph = networkx.random_regular_graph(3, n)

    betas = np.random.uniform(-np.pi, np.pi, size=p)
    gammas = np.random.uniform(-np.pi, np.pi, size=p)
    circuits = [
        examples.qaoa.qaoa_max_cut_circuit(
            qubits, betas, gammas, graph, use_boolean_hamiltonian_gate
        )
        for use_boolean_hamiltonian_gate in [True, False]
    ]

    assert cirq.allclose_up_to_global_phase(
        cirq.unitary(circuits[0]), cirq.unitary(circuits[1]), atol=1e-8
    )


def test_example_runs_quantum_teleportation():
    _, teleported = examples.quantum_teleportation.main(seed=12)
    assert np.allclose(np.array([0.07023552, -0.9968105, -0.03788921]), teleported)


def test_example_runs_superdense_coding():
    examples.superdense_coding.main()


def test_example_runs_hhl():
    examples.hhl.main()


@pytest.mark.usefixtures('closefigures')
def test_example_runs_qubit_characterizations():
    examples.qubit_characterizations_example.main(
        minimum_cliffords=2, maximum_cliffords=6, cliffords_step=2
    )


def test_example_swap_networks():
    examples.swap_networks.main()


def test_example_noisy_simulation():
    examples.noisy_simulation_example.main()


def test_example_shor_modular_exp_register_size():
    with pytest.raises(ValueError):
        _ = examples.shor.ModularExp(target=[2, 2], exponent=[2, 2, 2], base=4, modulus=5)


def test_example_shor_modular_exp_register_type():
    operation = examples.shor.ModularExp(target=[2, 2, 2], exponent=[2, 2], base=4, modulus=5)
    with pytest.raises(ValueError):
        _ = operation.with_registers([2, 2, 2])
    with pytest.raises(ValueError):
        _ = operation.with_registers(1, [2, 2, 2], 4, 5)
    with pytest.raises(ValueError):
        _ = operation.with_registers([2, 2, 2], [2, 2, 2], [2, 2, 2], 5)
    with pytest.raises(ValueError):
        _ = operation.with_registers([2, 2, 2], [2, 2, 2], 4, [2, 2, 2])


def test_example_shor_modular_exp_registers():
    target = [2, 2, 2]
    exponent = [2, 2]
    operation = examples.shor.ModularExp(target, exponent, 4, 5)
    assert operation.registers() == (target, exponent, 4, 5)

    new_target = [2, 2, 2]
    new_exponent = [2, 2, 2, 2]
    new_operation = operation.with_registers(new_target, new_exponent, 6, 7)
    assert new_operation.registers() == (new_target, new_exponent, 6, 7)


def test_example_shor_modular_exp_diagram():
    target = [2, 2, 2]
    exponent = [2, 2]
    gate = examples.shor.ModularExp(target, exponent, 4, 5)
    circuit = cirq.Circuit(gate.on(*cirq.LineQubit.range(5)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───ModularExp(t*4**e % 5)───
      │
1: ───t1───────────────────────
      │
2: ───t2───────────────────────
      │
3: ───e0───────────────────────
      │
4: ───e1───────────────────────
""",
    )

    gate = gate.with_registers(target, 2, 4, 5)
    circuit = cirq.Circuit(gate.on(*cirq.LineQubit.range(3)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───ModularExp(t*4**2 % 5)───
      │
1: ───t1───────────────────────
      │
2: ───t2───────────────────────
""",
    )


def assert_order(r: int, x: int, n: int) -> None:
    """Assert that r is the order of x modulo n."""
    y = x
    for _ in range(1, r):
        assert y % n != 1
        y *= x
    assert y % n == 1


@pytest.mark.parametrize(
    'x, n', ((2, 3), (5, 6), (2, 7), (6, 7), (5, 8), (6, 11), (6, 49), (7, 810))
)
def test_example_shor_naive_order_finder(x, n):
    r = examples.shor.naive_order_finder(x, n)
    assert_order(r, x, n)


@pytest.mark.parametrize('x, n', ((2, 3), (5, 6), (2, 7), (6, 7)))
def test_example_shor_quantum_order_finder(x, n):
    r = None
    for _ in range(15):
        r = examples.shor.quantum_order_finder(x, n)
        if r is not None:
            break
    assert_order(r, x, n)


@pytest.mark.parametrize('x, n', ((1, 7), (7, 7)))
def test_example_shor_naive_order_finder_invalid_x(x, n):
    with pytest.raises(ValueError):
        _ = examples.shor.naive_order_finder(x, n)


@pytest.mark.parametrize('x, n', ((1, 7), (7, 7)))
def test_example_shor_quantum_order_finder_invalid_x(x, n):
    with pytest.raises(ValueError):
        _ = examples.shor.quantum_order_finder(x, n)


@pytest.mark.parametrize('n', (4, 6, 15, 125, 101 * 103, 127 * 127))
def test_example_shor_find_factor_with_composite_n_and_naive_order_finder(n):
    d = examples.shor.find_factor(n, examples.shor.naive_order_finder)
    assert 1 < d < n
    assert n % d == 0


@pytest.mark.parametrize('n', (4, 6, 15, 125))
def test_example_shor_find_factor_with_composite_n_and_quantum_order_finder(n):
    d = examples.shor.find_factor(n, examples.shor.quantum_order_finder)
    assert 1 < d < n
    assert n % d == 0


@pytest.mark.parametrize(
    'n, order_finder',
    itertools.product(
        (2, 3, 5, 11, 101, 127, 907),
        (examples.shor.naive_order_finder, examples.shor.quantum_order_finder),
    ),
)
def test_example_shor_find_factor_with_prime_n(n, order_finder):
    d = examples.shor.find_factor(n, order_finder)
    assert d is None


@pytest.mark.parametrize('n', (2, 3, 15, 17, 2**89 - 1))
def test_example_runs_shor_valid(n):
    examples.shor.main(n=n)


@pytest.mark.parametrize('n', (-1, 0, 1))
def test_example_runs_shor_invalid(n):
    with pytest.raises(ValueError):
        examples.shor.main(n=n)


def test_example_qec_single_qubit():
    mycode1 = OneQubitShorsCode()
    my_circuit1 = cirq.Circuit(mycode1.encode())
    my_circuit1 += cirq.Circuit(mycode1.correct())
    my_circuit1 += cirq.measure(mycode1.physical_qubits[0])
    sim1 = cirq.DensityMatrixSimulator()
    result1 = sim1.run(my_circuit1, repetitions=1)
    assert result1.measurements['q(0)'] == [[0]]

    mycode2 = OneQubitShorsCode()
    my_circuit2 = cirq.Circuit(mycode2.apply_gate(cirq.X, 0))
    with pytest.raises(IndexError):
        mycode2.apply_gate(cirq.Z, 89)
    my_circuit2 += cirq.Circuit(mycode2.encode())
    my_circuit2 += cirq.Circuit(mycode2.correct())
    my_circuit2 += cirq.measure(mycode2.physical_qubits[0])
    sim2 = cirq.DensityMatrixSimulator()
    result2 = sim2.run(my_circuit2, repetitions=1)
    assert result2.measurements['q(0)'] == [[1]]


@pytest.mark.usefixtures('closefigures')
def test_two_qubit_gate_compilation_example():
    plt.switch_backend('agg')
    examples.two_qubit_gate_compilation.main(samples=10, max_infidelity=0.3)
