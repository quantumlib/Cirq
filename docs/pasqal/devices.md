# Pasqal Devices

This section describes the devices in Cirq for Pasqal hardware devices and their usage.
While our hardware is currently under development, this information should provide a
better understanding of the capabilities of future Pasqal devices, where neutral atoms
are controlled by lasers. Please contact Pasqal to obtain the latest information
on devices that you plan to use.

Currently, there are two devices to choose from: `cirq_pasqal.PasqalDevice` and `cirq_pasqal.PasqalVirtualDevice`. Let us look at the role each of them plays.

## `PasqalDevice`

The `cirq_pasqal.PasqalDevice` class represents the most general of Pasqal devices, enforcing only restrictions expected to be shared by all future devices.

### Gate set
One of the restrictions is on the supported gate set, which is composed of the following gates (all present in the `cirq.ops` module):

  * Single-qubit gates:

    * `IdentityGate`
    * `MeasurementGate`
    * `PhasedXPowGate`
    * `XPowGate`
    * `YPowGate`
    * `ZPowGate`
    * `HPowGate(exponent=1)`

  * Multi-qubit gates

    * `CNotPowGate(exponent=1)`
    * `CZPowGate(exponent=1)`
    * `CCXPowGate(exponent=1)`
    * `CCZPowGate(exponent=1)`

Any gate appended to a `cirq.Circuit` associated with `PasqalDevice` that is not within this gate set will be, whenever possible, decomposed by Cirq's decomposer or otherwise rejected.

### Measurement limitation

The other restriction is on the measurement operation, which has to occur only once, at the end of the circuit, and simultaneously on all the qubits of interest. It can correspond to a single `cirq.ops.GateOperation` with a `cirq.ops.MeasurementGate` applied on all the qubits of interest (recommended) or multiple measurement operations, as long as they are all in the same moment.

### Usage

The `PasqalDevice` is intended to serve as the parent class of future Pasqal devices' classes. However, it can also be used as the host of a nearly unconstrained quantum circuit.

When using the `PasqalDevice` class as the device itself, the qubits have to be of the type `cirq.NamedQubit` and assumed to be all connected, the idea behind it being that after submission, all optimization and transpilation necessary for its execution on the specified device are handled internally by Pasqal.

Therefore, when a `PasqalDevice` hosts a circuit, the user submits a quantum circuit that is intentionally unaffected by the physical limitations of the device, leaving it up to Pasqal's compiler to adapt it so that it can be executed.

## `PasqalVirtualDevice`

Contrary to `PasqalDevice`, `PasqalVirtualDevice` allows the user to create a circuit that is then executed without modification on a physical device. It achieves this by imposing the expected restrictions that come with designing a quantum circuit for execution on a physical device. The added restrictions are imposed throughout the creation of a circuit so that, at any point, it could be executed without any changes.


### Qubit placement and connectivity

Qubits on Pasqal devices can be in arbitrary positions, either in 1D, 2D or 3D, and can be
created via the `cirq_pasqal.ThreeDQubit`, `cirq_pasqal.TwoDQubit`, `cirq.GridQubit` or `cirq.LineQubit` classes.

```python
from cirq_pasqal import TwoDQubit

# An array of 9 Pasqal qubits on a square lattice in 2D
p_qubits = [TwoDQubit(i, j) for i in range(3) for j in range(3)]

# Equivalently, using one of Pasqal's qubit classes' static methods for qubit register creation
p_qubits = TwoDQubit.square(3)  # Initializes qubits in a square grid of side 3

```

Connectivity is limited to qubits that are closer than a control radius. In the current
version, the control radius can be chosen by the user and passed as a parameter of the
`cirq_pasqal.PasqalVirtualDevice`; it is constrained to be at most three times the minimum
distance between all qubits in the layout.

```python
from cirq_pasqal import PasqalVirtualDevice

# A PasqalVirtualDevice with a control radius of 2.0 times the lattice spacing.
p_device = PasqalVirtualDevice(control_radius=2.0, qubits=p_qubits)

```

### Gate set restrictions

To the gate set allowed by `PasqalDevice`, `PasqalVirtualDevice` removes the `CNotPowGate`, `CCXPowGate` and `CCZPowGate(exponent=1)`, which are not expected to be available in the first generation of devices.

### Timing restrictions

Currently, no parallelization is allowed by `PasqalVirtualDevice`, which means that each gate is forced to be the only one in its `Moment` (except for Measurement gates on different qubits, which have to coexist in the final `Moment` of the circuit).
