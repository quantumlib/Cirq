# Google devices

This section describes the devices in Cirq for Google hardware devices and their usage.
Since quantum hardware is an active area of research, hardware specifications and best
practices are constantly evolving in an attempt to continuously improve performance.
While this information should be a solid base for beginning your quantum application,
please work with your Google sponsor to obtain the latest information on devices that
you plan to use.

## General limitations

Qubits on Google devices are laid out in a grid structure. Connectivity is limited to
adjacent qubits, either horizontally or vertically.

Measurement takes much longer than other gates. Currently, the only supported
configuration is to have terminal measurement in the final moment of a circuit.

Most devices have a limited set of gates that can be applied. Gates not in that set
must be decomposed into an equivalent circuit using gates within the set.
See below for those restrictions.

There are some limitations to the total circuit length due to hardware limitations.
Several factors can influence this limit, but this can be estimated at about 40 microseconds
of total circuit run-time. Circuits that exceed this limit will return a
"Program too long" error code.


### Moment structure

The hardware will attempt to run your circuit as it exists in cirq to the
extent possible.  The device will respect the moment structure of your circuit
and will execute successive moments in a serial fashion.

The only exception is that identity gates (such as `cirq.Z ** 0`) and virtual
Z gates (see below) will not actually perform any hardware actions.
If a moment contains only these "virtual" gates, it will disappear.

For example, this circuit will execute the two gates in parallel:

```
cirq.Circuit(
  cirq.Moment(cirq.X(cirq.GridQubit(4,4)), cirq.X(cirq.GridQubit(4,5)))
)
```

This circuit will execute the two gates in serial:


```
cirq.Circuit(
  cirq.Moment(cirq.X(cirq.GridQubit(4,4))),
  cirq.Moment(cirq.X(cirq.GridQubit(4,5)))
)
```

Lastly, this circuit will only execute one gate, since the first gate
is virtual and its moment will disappear:

```
cirq.Circuit(
  cirq.Moment(cirq.Z(cirq.GridQubit(4,4))),
  cirq.Moment(cirq.X(cirq.GridQubit(4,5)))
)
```

The duration of a moment is the time of its longest gate.  For example,
if a moment has gates of duration 12ns, 25ns, and 32ns, the entire moment
will take 32ns.  Qubits executing the shorter gtes will idle during the rest
of the time.  To minimize the duration of the circuit, it is best to align
gates of the same duration together when possible.  See the
[best practices](./best_practices.ipynb) for more details.

## Gates supported

The following lists the gates supported by Google devices.
Please note that gate durations are subject to change as hardware is
updated and modified, so please refer to the
[device specification](./specification.md)
to get up-to-date information on supported gates and durations for
specific processors.

In addition, please note that all gates will have variations and
errors that vary from device to device and from qubit to qubit.
This can include both incoherent as well as coherent error.

Note: Gate durations are subject to change based on device or
configuration.  To get gates durations for a specific device, see the
[Device specification](./specification.md#gate-durations) page.  Also
note that some gates (such as Z gates or Fsim gates) have multiple
variations that can have different durations.

### One qubit gates

Google devices support arbitrary one-qubit gates of any rotation.
The full complement of these rotations can be accessed by using the
`cirq.PhasedXZGate`.  More restrictive one-qubit gates, such as
the Pauli gates `cirq.X`, `cirq.Y`, `cirq.Z`, as well as the
gate `cirq.PhasedXPowGate` can also be natively executed.
One qubit rotations have a duration of 25 ns on most Google devices.

#### Virtual Z gates

Rotation around the Z axis is not a hardware operation on its own.
Instead, the compilation keeps track of the Z phase rotations,
commuting them forward through the circuit until a non-commuting
gate is reached.  This compilation is handled automatically for you.
Adding a Z gate will generally not add any duration to the circuit, though
it may modify how the other gates are applied.

What this means is that `cirq.Z` and `cirq.ZPowGate` gates will
have zero duration on the device.  Any moments containing only
these gates will silently disappear from the circuit.  Even when
this gate is absorbed by non-commuting gates, such as the square
root of iSWAP, already have physical Z gates, so this absorption
still does not add duration to the circuit.

#### Physical Z gates

While most applications prefer shorter circuit durations and
virtual Z gates, Google hardware does offer the possibility of
applying a physical Z gate that performs a hardware operation
to affect the frequency of the qubit.

This can be done by applying a PhysicalZTag to the Z gate,
such as in the following example:

```
cirq.Z(cirq.GridQubit(5, 5)).with_tags(cirq_google.PhysicalZTag())
```

Physical Z gates have a duration of 20 ns on most Google devices.

### Two Qubit Gates

Google devices provide several options for two-qubit gates.
The availability of these gates is controlled by the gateset
parameter that is used.

Note that current hardware gates are noisy and not homogenous
across the device.  Unitaries provided below are for the ideal case.
Real unitaries applied to the qubit may vary from qubit to qubit
and may drift over time.

#### Sycamore Gate

The hardware provides a gate known as the Sycamore gate that can
be accessed using `cirq_google.SYC`.  This gate is equivalent to
an FSimGate(π/2, π/6).  That is, it does both a partial swap and
controlled phase rotation of the |11⟩ state.

The unitary of this gate, which can also be found via the `cirq.unitary`
function, is:

$$
\left[
\begin{matrix}
1 & 0 & 0 & 0 \\
0 & 0 & -i & 0 \\
0 & -i & 0 & 0 \\
0 & 0 & 0 & e^{-i \frac{\pi}{6}}
\end{matrix}
\right]
$$

This gate has a duration of 12ns and can be used in `cirq_google.SYC_GATESET`
or in the `cirq_google.FSIM_GATESET`.

#### Square root of iSWAP

The hardware provides the square root of the iSWAP gate
using `cirq.ISWAP ** 0.5`.  This gate is equivalent to an FSimGate(-π/4, 0).
The inverse (`cirq.ISWAP ** -0.5`) is also available.

The unitary of this gate, which can also be found via the `cirq.unitary`
function, is:

$$
\left[
\begin{matrix}
1 & 0 & 0 & 0 \\
0 & \frac{1}{\sqrt{2}} & \frac{i}{\sqrt{2}} & 0 \\
0 & \frac{i}{\sqrt{2}}& \frac{1}{\sqrt{2}} & 0 \\
0 & 0 & 0 & 1
\end{matrix}
\right]
$$

This gate has a duration of 32ns and can be used in
`cirq_google.SQRT_ISWAP_GATESET` or in the `cirq_google.FSIM_GATESET`.

This gate is implemented by using an entangling gate surrounding by
Z gates.  The preceding Z gates are physical Z gates and will absorb
any phases that have accumulated through the use of Virtual Z gates.
Following the entangler are virtual Z gates to match phases back.  All
of this is computed and handled for the user automatically.

Users should note that this gate is approximate and calibrated for
average performance across the entire processor.  In particular,
average variations of the 'phi' angle of about π/24 have been observed
on some devices.

#### CZ gate

The controlled-Z gate `cirq.CZ` is experimentally available on some
devices.  Be sure to check with your sponsor or in the device specification
to see if it is available on the processor you are using.

This gate is equivalent to FSimGate(0, π).  It has an approximate duration
of 26ns.

#### FSim gateset

The `cirq.FSIM_GATESET` provides all three of the above gates in one set.
In addition, by using this combined gate set, the FSimGate can be parameterized,
which allows for efficient sweeps across varying two-qubit gates.
Note that providing a theta/phi combination that
is not one of the above gates will cause an error when run on hardware.


### Wait gate

For decay experiments and other applications, a WaitGate is provided
that causes the device to idle for a specified amount of time.
This can be accomplished by specifying a `cirq.WaitGate`.


### Subcircuits

Circuits with a repetitive structure can benefit from using
`cirq.CircuitOperation` to specify "subcircuits" within the overall circuit.
Using this type condenses the serialized representation of the circuit, which
may help for circuits that would otherwise run into size limitations.

### Parameterized Gates

Circuits for Google devices could contain gates parameterized by Sympy
expressions, but only a subset of Sympy expression types are supported:
`sympy.Symbol`, `sympy.Add`, `sympy.Mul`, and `sympy.Pow`.

## Specific Device Layouts
The following devices are provided as part of cirq and can help you get your
circuit ready for running on hardware by verifying that you are using
appropriate qubits.

Note that real hardware does not always have all qubits enabled, and it
is important to check the device specification for the processor that you
will attempt to run on to make sure that the qubits your circuit uses
are actually active.  Regular calibration and maintenance can disable
and enable misbehaving qubits, so the grid configuration can change on a
daily basis.

### Sycamore

The Sycamore device is a 54 qubit device introduced in 2019 with a
[publication in Nature](https://www.nature.com/articles/s41586-019-1666-5).
Note that the supremacy result in the paper utilized a device that had 53 qubits since
one qubit had malfunctioned.

It can be accessed using `cirq.GridQubit(row, col)` using grid coordinates specified below.

```
  0123456789
0 -----AB---
1 ----ABCD--
2 ---ABCDEF-
3 --ABCDEFGH
4 -ABCDEFGHI
5 ABCDEFGHI-
6 -CDEFGHI--
7 --EFGHI---
8 ---GHI----
9 ----I-----
```

It can be accessing by using `cirq_google.Sycamore`. This device has two possible
two-qubits gates that can be used.

*  Square root of ISWAP. The gate `cirq.ISWAP ** 0.5` or `cirq.ISWAP ** -0.5` can be
used on `cirq_google.optimized_for_sycamore` with optimizer type `sqrt_iswap`
*  Sycamore gate. This gate, equivalent to FSimGate(π/2, π/6) can be used as `cirq_google.SYC`
or by using `cirq.FsimGate(numpy.pi/2,numpy.pi/6)`. Circuits can be compiled to use this gate
by using `cirq_google.optimized_for_sycamore` with optimizer type `sycamore`


### Sycamore23

The Sycamore23 chip is a 23-qubit subset of the Sycamore chip that is easier to work
with and presents less hardware-related complications than using the full Sycamore device.


```
  0123456789
0 ----------
1 ----------
2 ----------
3 --A-------
4 -ABC------
5 ABCDE-----
6 -CDEFG----
7 --EFGHI---
8 ---GHI----
9 ----I-----
```

This grid can be accessed using `cirq_google.Sycamore23` and uses the same gate sets and
compilation as the Sycamore device.


### Bristlecone

The Bristlecone processor is a 72 qubit device
[announced by Google in 2018](https://ai.googleblog.com/2018/03/a-preview-of-bristlecone-googles-new.html).

The device is arrayed on a grid in a diamond pattern like this.

```
            11
  012345678901
0 -----AB-----
1 ----ABCD----
2 ---ABCDEF---
3 --ABCDEFGH--
4 -ABCDEFGHIJ-
5 ABCDEFGHIJKL
6 -CDEFGHIJKL-
7 --EFGHIJKL--
8 ---GHIJKL---
9 ----IJKL----
10-----KL-----
```

It can be accessing by using `cirq_google.Bristlecone`. Circuits can be compiled to it by using
`cirq_google.optimized_for_xmon` or by using `cirq_google.optimized_for_sycamore` with
optimizer_type `xmon`.

