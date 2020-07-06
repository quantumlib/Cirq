from multiqubit_qec import MultiQubitCode
from shors_code import OneQubitShorsCode

import cirq

# test 1
mycode1 = OneQubitShorsCode()

my_circuit1 = cirq.Circuit(mycode1.encode())

my_circuit1 += cirq.Circuit(mycode1.correct())

my_circuit1 += cirq.Circuit(mycode1.measure())

print(my_circuit1)
sim1 = cirq.DensityMatrixSimulator()
result1 = sim1.run(my_circuit1, repetitions=20)
print(result1)

#test2
mycode2 = OneQubitShorsCode()
my_circuit2 = cirq.Circuit(mycode2.apply_gate(cirq.X, 0))
my_circuit2 += cirq.Circuit(mycode2.encode())

my_circuit2 += cirq.Circuit(mycode2.correct())

my_circuit2 += cirq.Circuit(mycode2.measure())

print(my_circuit2)
sim2 = cirq.DensityMatrixSimulator()
result2 = sim2.run(my_circuit2, repetitions=20)
print(result2)

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
my_circuit3 = mycode3.measure()

print(my_circuit3)
sim3 = cirq.Simulator()
result3 = sim3.run(my_circuit3)
print(result3)
