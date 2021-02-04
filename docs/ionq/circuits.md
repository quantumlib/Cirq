# IonQ API Circuits

To run against the IonQ API, one must construct circuits that are valid
for the service. In other words, not every `cirq.Circuit` that you can
construct will be able to run on the IonQ API, either against hardware
or on the IonQ simulator.  Here we describe the restrictions on these circuits.
Here we assume a `cirq.ionq.Service` object has been instantiated and is
called `service`.  See [service.md](IonQ API Service) for how to set
this up.

## Qubits

The qubits used for circuits run against the IonQ API must be made
`cirq.LineQubit`s.  Line qubits are identified by a unique integer
identifier.  The number in the `cirq.LineQubit` does not
generically refer to the position of the ion in a chain, as the API
may decide to run your algorithm on different qubits than the number
you specify.  This integer number must be between zero and the number
of qubits on the device minus one, inclusively.  To get the number of
qubits on the device, one can query the calibration:

```python
calibration = service.get_current_calibration()
num_qubits = calibration.num_qubits()
print(num_qubits)
```

A useful way to generate a set of `cirq.LineQubit`s is to use the `range`
method on this class, which functions similar to Python's native `range`.
For example to create three qubits, with indices 0, 1, and 2 one can do
```python
q0, q1, q2 = cirq.LineQubit.range(3)
```


## Gates

The IonQ API supports.
