"""Generate correct JSON test data for contrib classes."""

import cirq
from cirq.contrib.acquaintance import SwapPermutationGate
from cirq.contrib.bayesian_network import BayesianNetworkGate
from cirq.contrib.quantum_volume import QuantumVolumeResult
import pathlib

# Create test objects matching the original json_test.py
bayesian_gate = BayesianNetworkGate(
    init_probs=[('q0', 0.125), ('q1', None)],
    arc_probs=[('q1', ('q0',), [0.25, 0.5])]
)

qubits = cirq.LineQubit.range(5)
qvr = QuantumVolumeResult(
    model_circuit=cirq.Circuit(cirq.H.on_each(qubits)),
    heavy_set=[1, 2, 3],
    compiled_circuit=cirq.Circuit(cirq.H.on_each(qubits)),
    sampler_result=0.1,
)

swap_gate = SwapPermutationGate(swap_gate=cirq.SWAP)

# Define output directory
output_dir = pathlib.Path("cirq-core/cirq/contrib/json_test_data")

# Generate and write files
test_objects = [
    ("BayesianNetworkGate", bayesian_gate),
    ("QuantumVolumeResult", qvr),
    ("SwapPermutationGate", swap_gate),
]

for name, obj in test_objects:
    # Write JSON file
    json_path = output_dir / f"{name}.json"
    with open(json_path, 'w') as f:
        f.write(cirq.to_json(obj))
    print(f"Created {json_path}")
    
    # Write repr file
    repr_path = output_dir / f"{name}.repr"
    with open(repr_path, 'w') as f:
        f.write(repr(obj))
    print(f"Created {repr_path}")

print("\nAll test data files created successfully!")