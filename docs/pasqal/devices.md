# Pasqal Devices

This section describes the devices in Cirq for Pasqal hardware devices and their usage.
While our hardware is constantly under development, this information should give a
better understanding of the capabilities of Pasqal devices, powered by neutral atoms
controlled by lasers. Please work contact Pasqal to obtain the latest information
on devices that you plan to use.

## General limitations

Qubits on Pasqal devices are laid out in a grid structure, either in 2D or 3D.
Connectivity is limited to qubits that are closer than a control radius. In its current
version, the control radius can be chosen by the user and passed as a parameter of the
PasqalDevice class. In the future, specific versions of the devices may impose some
constraints on that value.

Measurement takes much longer than other gates. Currently, the only supported
configuration is to have terminal measurement in the final moment of a circuit.
