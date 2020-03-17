import itertools

import numpy as np
import pytest
import matplotlib.pyplot as plt

import cirq
import examples.basic_arithmetic
import examples.bb84
import examples.bell_inequality
import examples.bernstein_vazirani
import examples.bcs_mean_field
import examples.bristlecone_heatmap_example
import examples.cross_entropy_benchmarking_example
import examples.deutsch
import examples.grover
import examples.hello_qubit
import examples.hhl
import examples.noisy_simulation_example
import examples.phase_estimator
import examples.place_on_bristlecone
import examples.qaoa
import examples.quantum_fourier_transform
import examples.quantum_teleportation
import examples.qubit_characterizations_example
import examples.shor
import examples.superdense_coding
import examples.swap_networks


def test_example_runs_bernstein_vazirani():
    examples.bernstein_vazirani.main(qubit_count=3)

    # Check empty oracle case. Cover both biases.
    a = cirq.NamedQubit('a')
    assert list(examples.bernstein_vazirani.make_oracle([], a, [], False)) == []
    assert list(examples.bernstein_vazirani.make_oracle([], a, [],
                                                        True)) == [cirq.X(a)]


def test_example_runs_deutsch():
    examples.deutsch.main()


def test_example_runs_hello_line():
    examples.place_on_bristlecone.main()


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


def test_example_runs_bristlecone_heatmap():
    plt.switch_backend('agg')
    examples.bristlecone_heatmap_example.main()


def test_example_runs_qaoa():
    examples.qaoa.main(repetitions=10, maxiter=5)


def test_example_runs_quantum_teleportation():
    expected, teleported = examples.quantum_teleportation.main()
    assert np.all(np.isclose(expected, teleported, atol=1e-4))


def test_example_runs_superdense_coding():
    examples.superdense_coding.main()


def test_example_runs_hhl():
    examples.hhl.main()


def test_example_runs_qubit_characterizations():
    examples.qubit_characterizations_example.main(minimum_cliffords=2,
                                                  maximum_cliffords=6,
                                                  cliffords_step=2)


def test_example_swap_networks():
    examples.swap_networks.main()


def test_example_cross_entropy_benchmarking():
    examples.cross_entropy_benchmarking_example.main(repetitions=10,
                                                     num_circuits=2,
                                                     cycles=[2, 3, 4])


def test_example_noisy_simulation():
    examples.noisy_simulation_example.main()


def test_example_shor_modular_exp_register_size():
    with pytest.raises(ValueError):
        _ = examples.shor.ModularExp(target=cirq.LineQubit.range(2),
                                     exponent=cirq.LineQubit.range(2, 5),
                                     base=4,
                                     modulus=5)


def test_example_shor_modular_exp_register_type():
    operation = examples.shor.ModularExp(target=cirq.LineQubit.range(3),
                                         exponent=cirq.LineQubit.range(3, 5),
                                         base=4,
                                         modulus=5)
    with pytest.raises(ValueError):
        _ = operation.with_registers(cirq.LineQubit.range(3))
    with pytest.raises(ValueError):
        _ = operation.with_registers(1, cirq.LineQubit.range(3, 6), 4, 5)
    with pytest.raises(ValueError):
        _ = operation.with_registers(cirq.LineQubit.range(3),
                                     cirq.LineQubit.range(3, 6),
                                     cirq.LineQubit.range(6, 9), 5)
    with pytest.raises(ValueError):
        _ = operation.with_registers(cirq.LineQubit.range(3),
                                     cirq.LineQubit.range(3, 6), 4,
                                     cirq.LineQubit.range(6, 9))


def test_example_shor_modular_exp_registers():
    target = cirq.LineQubit.range(3)
    exponent = cirq.LineQubit.range(3, 5)
    operation = examples.shor.ModularExp(target, exponent, 4, 5)
    assert operation.registers() == (target, exponent, 4, 5)

    new_target = cirq.LineQubit.range(5, 8)
    new_exponent = cirq.LineQubit.range(8, 12)
    new_operation = operation.with_registers(new_target, new_exponent, 6, 7)
    assert new_operation.registers() == (new_target, new_exponent, 6, 7)


def test_example_shor_modular_exp_diagram():
    target = cirq.LineQubit.range(3)
    exponent = cirq.LineQubit.range(3, 5)
    operation = examples.shor.ModularExp(target, exponent, 4, 5)
    circuit = cirq.Circuit(operation)
    cirq.testing.assert_has_diagram(
        circuit, """
0: ───ModularExp(t*4**e % 5)───
      │
1: ───t1───────────────────────
      │
2: ───t2───────────────────────
      │
3: ───e0───────────────────────
      │
4: ───e1───────────────────────
""")

    operation = operation.with_registers(target, 2, 4, 5)
    circuit = cirq.Circuit(operation)
    cirq.testing.assert_has_diagram(
        circuit, """
0: ───ModularExp(t*4**2 % 5)───
      │
1: ───t1───────────────────────
      │
2: ───t2───────────────────────
""")


def assert_order(r: int, x: int, n: int) -> None:
    """Assert that r is the order of x modulo n."""
    y = x
    for _ in range(1, r):
        assert y % n != 1
        y *= x
    assert y % n == 1


@pytest.mark.parametrize('x, n', ((2, 3), (5, 6), (2, 7), (6, 7), (5, 8),
                                  (6, 11), (6, 49), (7, 810)))
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
    d = examples.shor.find_factor(n, examples.shor.naive_order_finder)
    assert 1 < d < n
    assert n % d == 0


@pytest.mark.parametrize(
    'n, order_finder',
    itertools.product(
        (2, 3, 5, 11, 101, 127, 907),
        (examples.shor.naive_order_finder, examples.shor.quantum_order_finder)))
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
