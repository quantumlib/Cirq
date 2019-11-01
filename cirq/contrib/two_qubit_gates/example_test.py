from cirq.contrib.two_qubit_gates import example


def test_gate_compilation_example():
    example.main(samples=10, max_infidelity=0.3, verbose=False)
