import cirq
import examples.bell_inequality
import examples.bernstein_vazirani
import examples.deutsch
import examples.grover
import examples.place_on_bristlecone
import examples.hello_qubit
import examples.quantum_fourier_transform
import examples.bcs_mean_field
import examples.phase_estimator
import examples.basic_arithmetic
import examples.quantum_teleportation
import examples.superdense_coding


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

def test_example_runs_quantum_teleportation():
    examples.quantum_teleportation.main()

def test_example_runs_superdense_coding():
    examples.superdense_coding.main()
