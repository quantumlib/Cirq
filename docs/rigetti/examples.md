# Examples

## Example: Run simple bell circuit

``` python
from pyquil import get_qc
import cirq
from cirq_rigetti import circuit_transformers, circuit_sweep_executors

bell_circuit = cirq.Circuit()
qubits = cirq.LineQubit.range(2)
bell_circuit.append(cirq.H(qubits[0]))
bell_circuit.append(cirq.CNOT(qubits[0], qubits[1]))
bell_circuit.append(cirq.measure(qubits[0], qubits[1], key='m'))

# assign qubits explicitly to hardware or virtual machine qubits.
qubit_id_map = {
   qubits[0]: 4,
   qubits[1]: 5,
}
transformer = circuit_transformers.build(qubit_id_map=qubit_id_map, qubits=qubits)
executor = circuit_sweep_executors.with_quilc_compilation_and_cirq_parameter_resolution

qc = get_qc("9q-square", as_qvm=True, noisy=True)
results = executor(
    quantum_computer=qc,
    circuit=bell_circuit,
    resolvers=[cirq.ParamResolver({})],
    repetitions=10,
    transformer=transformer,
)
print(results[0].histogram(key='m'))
```
