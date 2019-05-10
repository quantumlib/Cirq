import cirq
import cirq.ion as ci
from cirq import Simulator
import itertools
import random
import sys

### length of your hidden string
string_length = 4
### number of qubits needed for implementing the Bernstein-Vazirani Algorithm
qubit_num = string_length+1
### define your qubits as line qubits for a linear ion trap
qubit_list = cirq.LineQubit.range(qubit_num)
### generate all possible strings of length string_length, and randomly choose one as your hidden string
all_strings = ["".join(seq) for seq in itertools.product("01", repeat=string_length)]
hidden_string = random.choice(all_strings)

### make your ion trap device with desired gate times and qubits
us = 1000*cirq.Duration(nanos=1)
ion_device = ci.IonDevice(measurement_duration=100*us,
                        twoq_gates_duration=200*us,
                        oneq_gates_duration=10*us,
                        qubits=qubit_list)

### make the circuit for BV with clifford gates
circuit = cirq.Circuit()
circuit.append([cirq.X(qubit_list[qubit_num-1])])
for i in range(qubit_num):
    circuit.append([cirq.H(qubit_list[i])])
for i in range(qubit_num-1):
    if hidden_string[i] == '1':
        circuit.append([cirq.CNOT(qubit_list[i], qubit_list[qubit_num-1])])
for i in range(qubit_num - 1):
    circuit.append([cirq.H(qubit_list[i])])
    circuit.append([cirq.measure(qubit_list[i])])

print("Doing Bernstein-Vazirani algorithm with hidden string",
      hidden_string, "\n")
print("Clifford Circuit: \n", circuit, "\n")

### convert the clifford circuit into circuit with ion trap native gates
ion_circuit = ion_device.decompose_circuit(circuit)
print("Iontrap Circuit: \n", ion_circuit, "\n")

cirq.merge_single_qubit_gates_into_phased_x_z(ion_circuit)
print("Iontrap Circuit: \n", ion_circuit, "\n")
sys.exit(0)

### run the ion trap circuit
simulator = Simulator()
clifford_result = simulator.run(circuit)
result = simulator.run(ion_circuit)

measurement_results = ''
for i in range(qubit_num-1):
    if result.measurements[str(i)][0][0]:
        measurement_results += '1'
    else:
        measurement_results += '0'

print("Hidden string is:", hidden_string)
print("Measurement results are:", measurement_results)
print("Found answer using Bernstein-Vazirani:", hidden_string == measurement_results)
