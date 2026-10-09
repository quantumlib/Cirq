import pytest

import cirq


@pytest.mark.benchmark(group="greedy_qubit_manager")
def test_greedy_qubit_manager(benchmark):
    qm = cirq.GreedyQubitManager(prefix="bench", size=100)

    def _f():
        qubits = qm.qalloc(100)
        qm.qfree(qubits)

    benchmark(_f)
