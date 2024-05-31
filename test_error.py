import cirq

q = cirq.q(0)

op_a = cirq.X(q)
op_b = cirq.Z(q)

res = op_b * op_a**3
print(res, type(res))
res = op_b * op_a * op_a * op_a
print(res, type(res))
res = op_a**3 * op_b
print(res, type(res))

res = op_a * op_a**3
print(res, type(res))
res = op_a * op_a * op_a * op_a
print(res, type(res))

res = op_a * op_a**2
print(res, type(res))
res = op_a * op_a * op_a
print(res, type(res))

res = op_b * op_a**2
print(res, type(res))
res = op_b * op_a * op_a
print(res, type(res))


print("\n\n")
a, b = cirq.LineQubit.range(2)
Xa, Zb = cirq.X(a), cirq.Z(b)

print(Xa * Xa * Xa == Xa)
print(Xa * Xa**2 == Xa)
print(Zb * Xa * Xa == Zb)
print(Zb * Xa**2 == Zb)
print(Xa * Xa * Xa * Xa == cirq.PauliString())
print(Xa * Xa**3 == cirq.PauliString())
print(Zb * Xa * Xa * Xa == Zb * Xa)
print(Zb * Xa**3 == Zb * Xa)

#    assert op_a * op_a * op_a == op_a
#    assert op_a * op_a ** 2 == op_a
#    assert op_b * op_a * op_a == op_b
#    assert op_b * op_a ** 2 == op_b
