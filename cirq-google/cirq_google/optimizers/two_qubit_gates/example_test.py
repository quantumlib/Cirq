# pylint: disable=wrong-or-nonexistent-copyright-notice
from matplotlib import pyplot as plt

from cirq_google.optimizers.two_qubit_gates import example


def test_gate_compilation_example():
    plt.switch_backend('agg')
    example.main(samples=10, max_infidelity=0.3)
