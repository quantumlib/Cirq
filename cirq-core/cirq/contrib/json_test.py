# pylint: disable=wrong-or-nonexistent-copyright-notice
import cirq
from cirq.contrib.quantum_volume import QuantumVolumeResult
from cirq.testing import assert_json_roundtrip_works
from cirq.contrib.json import DEFAULT_CONTRIB_RESOLVERS
from cirq.contrib.acquaintance import SwapPermutationGate


def test_quantum_volume():
    qubits = cirq.LineQubit.range(5)
    qvr = QuantumVolumeResult(
        model_circuit=cirq.Circuit(cirq.H.on_each(qubits)),
        heavy_set=[1, 2, 3],
        compiled_circuit=cirq.Circuit(cirq.H.on_each(qubits)),
        sampler_result=0.1,
    )
    assert_json_roundtrip_works(qvr, resolvers=DEFAULT_CONTRIB_RESOLVERS)


def test_swap_permutation_gate():
    gate = SwapPermutationGate(swap_gate=cirq.SWAP)
    assert_json_roundtrip_works(gate, resolvers=DEFAULT_CONTRIB_RESOLVERS)
