# IonQ API Circuits

To run against the IonQ API, you must construct circuits that are valid
for the service. In other words, not every `cirq.Circuit` that you can
construct will be able to run on the IonQ API, either against hardware
or on the IonQ simulator.  Here we describe the restrictions on these circuits.

In this section we assume a `cirq_ionq.Service` object has been instantiated and is
called `service` and `cirq` and `cirq_ionq` have been imported:
```python
import cirq
import cirq_ionq as ionq
service = ionq.Service()
```
See [IonQ API Service](service.md) for how to set up the service.

## Qubits

The qubits used for circuits run against the IonQ API must be made
`cirq.LineQubit`s.  Line qubits are identified by a unique integer
identifier.  The number in the `cirq.LineQubit` does not
generically refer to the position of the ion in a chain, as the API
may decide to run your algorithm on different qubits than the number
you specify.  This integer number must be between zero and the number
of qubits on the device minus one, inclusively.  To get the number of
qubits on the device, you can query the calibration:

```python
calibration = service.get_current_calibration()
num_qubits = calibration.num_qubits()
print(num_qubits)
```

A useful way to generate a set of `cirq.LineQubit`s is to use the `range`
method on this class, which functions similar to Python's native `range`.
For example to create three qubits, with indices 0, 1, and 2 you can do
```python
q0, q1, q2 = cirq.LineQubit.range(3)
```

## API Gates

The IonQ API supports a set of gates via the API.  Circuits written with
these gates can be run directly on the API without modifying the circuit.
If the circuit has gates that are not in the this API gate set, then you
must transpile these circuits into the API gate set.

The API gate for the IonQ device is given by
* `cirq.XPowGate`, `cirq.YPowGate`, `cirq.ZPowGate`
    * This includes `cirq.rx`, `cirq.ry`, and `cirq.rz` and Pauli gates
    `cirq.X`, `cirq.Y`, and `cirq.Z`.
* `cirq.H`
* `cirq.XXPowGate`, `cirq.YYPowGate`, `cirq.ZZPowGate`
* `cirq.CNOT`, `cirq.SWAP`
* `cirq.MeasurementGate`: usually via `cirq.measure`.

Here is a nonsense quantum circuit constructed from these API gates,
demonstrating how to use these gates.
```python
q0, q1, q2 = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.X(q0)**0.5, cirq.Y(q1)**0.5, cirq.Z(q2)**0.25, # Pauli Pow gates
    cirq.X(q0), cirq.Y(q1), cirq.Z(q2), # Pauli gates
    cirq.rx(0.1)(q0), cirq.ry(0.1)(q1), cirq.rz(0.1)(q2), # Single qubit rotations
    cirq.H(q1), # Special case of Hadamard
    cirq.CNOT(q0, q1), cirq.SWAP(q2, q1), # Controlled-not and its SWAP cousin
    cirq.XX(q0, q1)**0.2, cirq.YY(q1, q2)**0.2, cirq.ZZ(q2, q0)**0.2, # MS gates
    cirq.measure(q0, key='x'), # Single qubit measurement
    cirq.measure(q1, q2, key='y') # Two qubit measurement
)
print(circuit)
```
which is the circuit
```
0: ───X^0.5───X───Rx(0.032π)───────@───────XX────────────────ZZ───────M('x')───
                                   │       │                 │
1: ───Y^0.5───Y───Ry(0.032π)───H───X───×───XX^0.2───YY───────┼────────M('y')───
                                       │            │        │        │
2: ───T───────Z───Rz(0.032π)───────────×────────────YY^0.2───ZZ^0.2───M────────
```

## Measurement

For the IonQ API, measurement is currently only supported if the measurement is
at the end of the circuit.  Measurement gates have keys which are then used to
batch the results via this key.  For example above we see that there are two
keys, one for measuring the first qubit and one for measuring the second and
third qubit.

## Support for general one and two qubit gates.

If you have a circuit with gates outside of the API native gates, you will
need to convert these gates into the native gates.  For the case in which
these gates are one or two qubit gates which support the `unitary` protocol
(i.e. which support calling `cirq.unitary` on the gate produces the unitary
for the gate), there is support for compiling these into API supported gates.
This conversion may not be optimal, but it does produce a valid API circuit.

This support is given by the `cirq_ionq.IonQAPIDevice` and its
`decompose_operation` method.  On way to use this is to pass the device
to a circuit, and these decompositions will be automatically applied while
the circuit is being constructed:
python
```
q0 = cirq.LineQubit(0)
device = ionq.IonQAPIDevice([q0])
circuit = cirq.Circuit(device=device)
circuit.append(cirq.H(q0)**0.2) # Non-API gate
print(circuit)
```
which produces
```
0: ───Z^(1/14)───X^0.14───Z^(1/14)───
```

Note that the decomposition changes with the `cirq.Moment` structure of the
circuit.

## Next steps

[How to use the service API](jobs.md)

[Get information about QPUs from IonQ calibrations](calibrations.md)

