# Pasqal Devices

This section describes the devices in Cirq for Pasqal hardware devices and their usage.
While our hardware is currently under development, this information should give a
better understanding of the capabilities of future Pasqal devices, powered by neutral atoms
controlled by lasers. Please work contact Pasqal to obtain the latest information
on devices that you plan to use.

## General limitations

Qubits on Pasqal devices are laid out in a grid structure, either in 2D or 3D, and can be
created via the `cirq.pasqal.ThreeDGridQubit` class.

```python
from cirq.pasqal import ThreeDGridQubit

# An array of 9 Pasqal qubit on a square lattice in 2D
p_qubits = [ThreeDGridQubit(i, j, 0) for i in range(3) for j in range(3)]

```

Connectivity is limited to qubits that are closer than a control radius. In the current
version, the control radius can be chosen by the user and passed as a parameter of the
`cirq.pasqal.PasqalDevice` class. In the future, specific versions of the devices
may impose some constraints on that value.

```python
from cirq.pasqal import PasqalDevice

# A PasqalDevice with a control radius of 2.0 times the lattice spacing.
p_device = PasqalDevice(control_radius=2.0, qubits=p_qubits)

```


## Gates

The Pasqal device class currently does not correspond to a specific existing device, and
therefore accepts a broad set of gates. In the future, we will create specific devices with
their own limited gate set. We currently specify a unique duration  of 2 micro-seconds for
quantum gates in the device. Measurement takes much longer than other gates. Currently, the only
supported configuration is to have terminal measurement in the final moment of a circuit.
