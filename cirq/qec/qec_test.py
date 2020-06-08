from qec import MultiQubitCode
from shors_code import OneQubitShorsCode

import cirq

# test 1
mycode = OneQubitShorsCode()

my_circuit = cirq.Circuit(mycode.encode())

my_circuit += cirq.Circuit(mycode.decode())

my_circuit += cirq.Circuit(mycode.measure())

print(my_circuit)
sim1 = cirq.Simulator()
result = sim1.run(my_circuit, repetitions=20)
print(result)

#test2
mycode2 = OneQubitShorsCode()
my_circuit = cirq.Circuit(mycode2.apply_gate(cirq.X, 0))
my_circuit += cirq.Circuit(mycode2.encode())

my_circuit += cirq.Circuit(mycode2.decode())

my_circuit += cirq.Circuit(mycode2.measure())

print(my_circuit)
sim1 = cirq.Simulator()
result = sim1.run(my_circuit, repetitions=20)
print(result)

#test3

mycode3 = MultiQubitCode(cirq.LineQubit.range(4), OneQubitShorsCode)

my_circuit = mycode3.encode()

my_circuit.append(mycode3.decode())

my_circuit.append(mycode3.measure())

print(my_circuit)
sim1 = cirq.Simulator()
result = sim1.run(my_circuit, repetitions=20)
print(result)
