# Google Devices

This section describes the devices in Cirq for Google hardware devices and their usage.
Since quantum hardware is an active area of research, hardware specifications and best
practices are constantly evolving in an attempt to continuously improve performance.
While this information should be a solid base for beginning your quantum application,
please work with your Google sponsor to obtain the latest information on devices that
you plan to use. 

## General limitations

Qubits on Google devices are laid out in a grid structure.  Connectivity is limited to
adjacent qubits, either horizontally or vertically.  

Measurement takes much longer than other gates.  Currently, the only supported
configuration is to have terminal measurement in the final moment of a circuit.

Most devices have a limited number of gates that can applied to the device.  Gates not
in that set must be decomposed into an equivalent circuit using gates within the set.
See below for those restrictions. 

## Best Practices

This section lists some best practices for creating a circuit that performs well on hardware
devices.  This is an area of active research, so users are encouraged to try multiple
approaches to improve results.

### Use built-in optimizers as a first pass

Using built-in optimizers will allow you to compile to the correct .  As they are
automated solutions, they will not always perform as well as a hand-crafted solution, but
it provides a good starting point for creating a circuit that is likely to run successfully
on hardware.

```python
import cirq
import cirq.google as cg


# Create your circuit here
my_circuit = cirq.Circuit()

# Convert the circuit onto a Google device.
# Specifying a device will verify that the circuit satisifies constraints of the device
# Specifying an optimizer type (e.g. 'sqrt_iswap' or 'sycamore') will convert to the appropriate
# gate set and also do basic optimization, such as combining successive one-qubit gates. 
sycamore_circuit = cg.optimized_for_sycamore(my_circuit, device=cg.Sycamore, optimizer_type='sqrt_iswap')
```

### Good moment structure

In an attempt to fulfill the user's desire, quantum devices will respect the moment
structure of circuits to the greatest extent possible.

To this end, it is important that the moment structure of the circuit is kept as
short and concise as possible.  The length of a moment will generally be the length
of the longest gate in the moment, so keeping gates with similar durations together
will generally benefit the performance.

In particular, keep measurement gates in the same moment and make sure that any
circuit optimizers run do not alter this by pushing measurements forward.  This
behavior can be avoided by measuring all qubits with a single gate or by adding
the measurement gate after all optimizers have run.

### Short gate depth

In the current NISQ (noisy intermediate scale quantum) era, gates and devices still
have significant error.  Both gate errors and T1 decay rate can cause long circuits
to have noise that overwhelms any signal in the circuit.  Also, circuits that are
much too long can even be rejected due to various limits in the system.

The recommended gate depths vary significantly with the structure of the circuit itself
and will likely increase as the devices improve.  However, circuits with more than
50 moments will generally struggle to retain any information.

### Keep qubits busy
 
Qubits that remain idle for long periods tend to dephase and decoher.  Inserting a
[Spin Echo](https://en.wikipedia.org/wiki/Spin_echo) into your circuit, such as a pair
of involutions, such as two successive Pauli Y gates, will generally increase
performance of the circuit.

### Alternate single-qubit and two-qubit layers

Devices are generally calibrated to circuits that alternate single-qubit gates with
two-qubit gates in each layer.  Staying close to this paradigm will often improve
performance of circuits.

Devices generally operate in the Z basis, so that rotations around the Z axis will become
book-keeping measures rather than physical operations on the device.  These operations
should be aggregated into their own moment, if possible.

## Specific Device Layouts

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

It can be accessing by using `cirq.google.Sycamore`.  This device has two possible
two-qubits that can be used.

*  Square root of ISWAP.  The gate `cirq.ISWAP ** 0.5` or `cirq.ISWAP ** -0.5` can be
used on `cirq.google.optimized_for_sycamore` with optimizer type `sqrt_iswap`
*  Sycamore gate.  This gate, equivalent to FSimGate(π/2, π/6) can be used as `cirq.google.SYC`
or by using `cirq.FsimGate(numpy.pi/2,numpy.pi/6)`.  Circuits can be compiled to use this gate
by using `cirq.google.optimized_for_sycamore` with optimizer type `sycamore`


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

This grid can be accessed using `cirq.google.Sycamore23` and uses the same gate sets and
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
0 -----KL-----
```

It can be accessing by using `cirq.google.Bristlecone`.  Circuits can be compiled to it by using
`cirq.google.optimized_for_xmon` or by using `cirq.google.optimized_for_sycamore` with
optimizer_type `xmon`.

### Foxtail

The Foxtail device is a 2 by 11 XMON device arranged in a bilinear array,
addressable by using grid qubits `({0,1}, {0-11})`.  It was one of the first
super-conducting quantum devices announced by Google.  Due to the small number of qubits
and limited connectivity, it is still interesting for exploring the space of constrained
algorithms on NISQ devices.

It can be accessing by using `cirq.google.Foxtail`.  Circuits can be compiled to it by using
`cirq.google.optimized_for_xmon` or by using `cirq.google.optimized_for_sycamore` with
optimizer_type `xmon`.





