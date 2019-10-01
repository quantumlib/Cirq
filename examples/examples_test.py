import numpy as np
import pytest

import cirq
import examples.basic_arithmetic
import examples.bell_inequality
import examples.bernstein_vazirani
import examples.bcs_mean_field
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
    assert list(examples.bernstein_vazirani.make_oracle(
        [], a, [], False)) == []
    assert list(examples.bernstein_vazirani.make_oracle(
        [], a, [], True)) == [cirq.X(a)]


def test_example_runs_deutsch():
    examples.deutsch.main()


def test_example_runs_hello_line():
    examples.place_on_bristlecone.main()


def test_example_runs_hello_qubit():
    examples.hello_qubit.main()


def test_example_runs_bell_inequality():
    examples.bell_inequality.main()


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
    examples.qubit_characterizations_example.main()


def test_example_swap_networks():
    examples.swap_networks.main()


def test_example_cross_entropy_benchmarking():
    examples.cross_entropy_benchmarking_example.main(repetitions=10,
                                                     num_circuits=2,
                                                     cycles=[2, 3, 4])


def test_example_noisy_simulation():
    examples.noisy_simulation_example.main()


@pytest.mark.parametrize('x, n', ((4, 7), (6, 49), (7, 810)))
def test_example_shor_naive_order_finder(x, n):
    r = examples.shor.naive_order_finder(x, n)
    y = x
    for _ in range(1, r):
        assert y % n != 1
        y *= x
    assert y % n == 1


@pytest.mark.parametrize('x, n', ((4, 7), (6, 49), (7, 810)))
def test_example_shor_quantum_order_finder(x, n):
    with pytest.raises(NotImplementedError):
        _ = examples.shor.quantum_order_finder(x, n)


@pytest.mark.parametrize('x, n', ((1, 7), (7, 7)))
def test_example_shor_naive_order_finder_invalid_x(x, n):
    with pytest.raises(ValueError):
        _ = examples.shor.naive_order_finder(x, n)


@pytest.mark.parametrize('n', (4, 6, 15, 125, 101 * 103, 127 * 127))
def test_example_shor_find_factor_composite(n):
    d = examples.shor.find_factor(n, examples.shor.naive_order_finder)
    assert 1 < d < n
    assert n % d == 0


@pytest.mark.parametrize('n', (2, 3, 5, 11, 101, 127, 907))
def test_example_shor_find_factor_prime(n):
    d = examples.shor.find_factor(n, examples.shor.naive_order_finder)
    assert d is None


@pytest.mark.parametrize('n', (2, 3, 15, 17, 2**89 - 1))
def test_example_runs_shor_valid(n):
    examples.shor.main(n=n)


@pytest.mark.parametrize('n', (-1, 0, 1))
def test_example_runs_shor_invalid(n):
    with pytest.raises(ValueError):
        examples.shor.main(n=n)
