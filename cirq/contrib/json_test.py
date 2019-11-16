import cirq
from cirq.contrib.quantum_volume import QuantumVolumeResult
from cirq.testing import assert_json_roundtrip_works
from cirq.contrib.json import contrib_class_resolver


def test_quantum_volume():
    qubits = cirq.LineQubit.range(5)
    qvr = QuantumVolumeResult(
        model_circuit=cirq.Circuit(cirq.H.on_each(qubits)),
        heavy_set=[1, 2, 3],
        compiled_circuit=cirq.Circuit(cirq.H.on_each(qubits)),
        sampler_result=.1)
    assert_json_roundtrip_works(qvr,
                                resolvers=[contrib_class_resolver] +
                                cirq.protocols.json.DEFAULT_RESOLVERS)
