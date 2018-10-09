import cirq
import examples.bell_inequality
import examples.bernstein_vazirani
import examples.grover
import examples.place_on_bristlecone
import examples.hello_qubit
import examples.quantum_fourier_transform
import examples.bcs_mean_field
import examples.phase_estimator


def test_example_runs_bernstein_vazirani():
    examples.bernstein_vazirani.main()

    # Check empty oracle case. Cover both biases.
    a = cirq.NamedQubit('a')
    assert list(examples.bernstein_vazirani.make_oracle(
        [], a, [], False)) == []
    assert list(examples.bernstein_vazirani.make_oracle(
        [], a, [], True)) == [cirq.X(a)]


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


def test_example_runs_phase_estimator():
    examples.phase_estimator.main()
