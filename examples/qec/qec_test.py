from multiqubit_qec import MultiQubitCode
from shors_code import OneQubitShorsCode

import cirq

# test 1
mycode = OneQubitShorsCode()

my_circuit = cirq.Circuit(mycode.encode())

my_circuit += cirq.Circuit(mycode.correct())

my_circuit += cirq.Circuit(mycode.measure())

print(my_circuit)
sim1 = cirq.DensityMatrixSimulator()
result = sim1.run(my_circuit, repetitions=20)
print(result)

#test2
mycode2 = OneQubitShorsCode()
my_circuit = cirq.Circuit(mycode2.apply_gate(cirq.X, 0))
my_circuit += cirq.Circuit(mycode2.encode())

my_circuit += cirq.Circuit(mycode2.correct())

my_circuit += cirq.Circuit(mycode2.measure())

print(my_circuit)
sim1 = cirq.Simulator()
result = sim1.run(my_circuit, repetitions=20)
print(result)

#test3

original_qubits = cirq.LineQubit.range(3)

original_circuit = cirq.Circuit([
    cirq.Z(original_qubits[0]),
    cirq.X(original_qubits[1]),
    cirq.Y(original_qubits[2])
])
mycode3 = MultiQubitCode(original_qubits, OneQubitShorsCode)

mycode3.encode()
mycode3.operation(original_circuit)
mycode3.apply_error()
mycode3.correct()
my_circuit = mycode3.measure()

print(my_circuit)
sim1 = cirq.Simulator()
result = sim1.run(my_circuit)
print(result)
