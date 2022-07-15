# Device specifications

The device specification proto defines basic layout of a device as well as the
gate set and serialized ids that can be used.  This specification can be used
to find out specific characteristics of the device.

Though several standard [Google devices](devices.md) are defined for your
convenience, specific devices may have specialized layouts particular to that
processor.  For instance, there may be one or more qubit "drop-outs" that are
non-functional for whatever reason.   There could also be new or experimental
features enabled on some devices but not on others.

This specification is defined in the Device proto within `cirq_google.api.v2`.

## Gate Set Specifications

Most devices can only accept a limited set of gates.  This is known as the
gate set of the device.   Any circuits sent to this device must only use gates
within this set.  The gate set portion of the protocol buffer defines which
gate set(s) are valid on the device, and which gates make up that set.

### Gate Definitions

Each gate in the gate set will have a definition that defines the id that
the gate is serialized as, the number of qubits for the gates, the arguments
to the gate, the duration, and which qubits it can be applied to.

This definition uses "target sets" to specify which qubits the operation can
be applied to.  See the section below for more information.

### Gate Durations

The time it takes the device to perform each gate is stored within the device
specification.  This time is stored as an integer number of picoseconds.

Example code to print out the gate durations for every gate supported by the
device is shown below:

```
import cirq

# Create an Engine object to use.
engine = cirq_google.Engine(project_id='your_project_id')

# Replace the processor id to get the device specification with that id.
spec = engine.get_processor('processor_id').get_device_specification()

# Iterate through each gate set valid on the device.
for gateset in spec.valid_gate_sets:
    print(gateset.name)
    print('-------')
    # Prints each gate valid in the set with its duration
    for gate in gateset.valid_gates:
        print('%s %d' % (gate.id, gate.gate_duration_picos))
```

Note that, by convention, measurement gate duration includes both the duration
of "read-out" pulses to measure the qubit as well as the "ring-down" time that
it takes the measurement resonator to reset to a ground state.

## Target Sets

Generally, most gates apply to the same set of qubits.  To avoid repeating
these qubits (or pairs of qubits) for each gate, each gate instead uses a
target set to define the set of qubits that are valid.

Each target set contains a list of valid targets.  A target is a list of qubits.
For one-qubit gates, a target is simply a single qubit.  For two qubit gates,
a target is a pair of qubits.

The type of a target set defines how the targets are interpreted.  If the
target set is set to SYMMETRIC, the order of each target does not matter (e.g.
if `gate.on(q1, q2)` is valid, then so is `gate.on(q2, q1)`).  If the target
type is set to ASYMMETRIC, then the order of qubits does matter, and other
orderings of the qubits that are not specified in the definition cannot be
assumed to be valid.

The last type is PERMUTATION_SET.  This type specified that any permutation of
the targets is valid.  This is typically used for measurement gates.  If `q0`,
`q1` and `q2` are all specified as valid targets for a permutation set of the
gate, then `gate.on(q0)`, `gate.on(q1)`, `gate.on(q2)`, `gate.on(q0, q1)`,
`gate.on(q0, q2)`, `gate.on(q1, q2)` and `gate.on(q0, q1, q2)` are all valid
uses of the gate.

### Developer Recommendations

This is a free form text field for additional recommendations and soft
requirements that should be followed for proper operation of the device that
are not captured by the hard requirements above.

For instance, "Do not apply two CZ gates in a row."

## Conversion to cirq.Device

The `cirq_google.GridDevice` class allows someone to take this
device specification and turn it into a `cirq.Device` that can be used to
verify a circuit.

The following example illustrates retrieving the device specification live
from the engine and then using it to validate a circuit.

```
import cirq
import cirq_google as cg

# Create an Engine object to use.
engine = cg.Engine(project_id='your_project_id',
                   proto_version=cirq_google.ProtoVersion.V2)

# Replace the processor id to get the device with that id.
device = engine.get_processor('processor_id').get_device()

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.CZ(q0, q1))

# Raises a ValueError, since CZ is not a supported gate.
device.validate_circuit(circuit)
```

Note that, if network traffic is undesired, the `DeviceSpecification` can
easily be stored in either binary format or TextProto format for later usage.
