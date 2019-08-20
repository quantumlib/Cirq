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
import examples.phase_estimator
import examples.place_on_bristlecone
import examples.qaoa
import examples.quantum_fourier_transform
import examples.quantum_teleportation
import examples.qubit_characterizations_example
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
    examples.quantum_teleportation.main()


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
