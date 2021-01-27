# IonQ API Circuits

To run against the IonQ API, one must construct circuits that are valid
for the service. In other words, not every `cirq.Circuit` that you can
construct will be able to run on the IonQ API, either against hardware
or on the IonQ simulator.  Here we describe the restrictions on these
circuits.

## Qubits

The qubits used for circuits run against the IonQ API must be made
`cirq.LineQubit`s.  Line qubits are identified by a unique integer
identifier.  The number in the `cirq.LineQubit` does not
generically refer to the position of the ion in a chain, as the API
may decide to run your algorithm on different qubits

Here, for example is an easy way to generate three q



## Gates

The IonQ API supports.
