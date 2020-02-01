# Google Devices

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

## Best Practices

This section lists some best practices for creating a circuit that performs well on hardware
devices. This is an area of active research, so users are encouraged to try multiple
approaches to improve results.

### Use built-in optimizers as a first pass

Using built-in optimizers will allow you to compile to the correct gate set. As they are
automated solutions, they will not always perform as well as a hand-crafted solution, but
they provide a good starting point for creating a circuit that is likely to run successfully
on hardware. Best practice is to inspect the circuit after optimization to make sure
that it has compiled without unintended consequences.

```python
import cirq
import cirq.google as cg


# Create your circuit here
my_circuit = cirq.Circuit()

# Convert the circuit onto a Google device.
# Specifying a device will verify that the circuit satisifies constraints of the device
# The optimizer type (e.g. 'sqrt_iswap' or 'sycamore') specifies which gate set
# to convert into and which optimization routines are appropriate.
# This can include combining successive one-qubit gates and ejecting virtual Z gates. 
sycamore_circuit = cg.optimized_for_sycamore(my_circuit, new_device=cg.Sycamore, optimizer_type='sqrt_iswap')
```

### Good moment structure

Quantum Engine will execute a circuit as faithfully as possible.
This means moment structure will be preserved. That is, all gates in a moment are
guaranteed to be executed before those in any later moment and after gates in
previous moments. 

To this end, it is important that the moment structure of the circuit is kept as
short and concise as possible. The length of a moment will generally be the length
of the longest gate in the moment, so keeping gates with similar durations together
will shorten the duration of the circuit and likely reduce the noise incurred.

In particular, keep measurement gates in the same moment and make sure that any
circuit optimizers do not alter this by pushing measurements forward. This
behavior can be avoided by measuring all qubits with a single gate or by adding
the measurement gate after all optimizers have run.

### Short gate depth

In the current NISQ (noisy intermediate scale quantum) era, gates and devices still
have significant error. Both gate errors and T1 decay rate can cause long circuits
to have noise that overwhelms any signal in the circuit. 

The recommended gate depths vary significantly with the structure of the circuit itself
and will likely increase as the devices improve. Total circuit fidelity can be roughly
estimated by multiplying the fidelity for all gates in the circuit. For example,
using a error rate of 0.5% per gate, a circuit of depth 20 and width 20 could be estimated
at 0.995^(20*20) = 0.135. Using separate error rates per gates (i.e. based on calibration
metrics) or a more complicated noise model can result in more accurate error estimation.


### Keep qubits busy
 
Qubits that remain idle for long periods tend to dephase and decohere. Inserting a
[Spin Echo](https://en.wikipedia.org/wiki/Spin_echo) into your circuit, such as a pair
of involutions, such as two successive Pauli Y gates, will generally increase
performance of the circuit.

### Alternate single-qubit and two-qubit layers

Devices are generally calibrated to circuits that alternate single-qubit gates with
two-qubit gates in each layer. Staying close to this paradigm will often improve
performance of circuits.

Devices generally operate in the Z basis, so that rotations around the Z axis will become
book-keeping measures rather than physical operations on the device. The EjectZ optimizer
included in optimizer lists for each device will generally compile these operations out
of the circuit by pushing them back to the next non-commuting operator. If the resulting
circuit still contains Z operations, they should be aggregated into their own moment,
if possible.

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

It can be accessing by using `cirq.google.Sycamore`. This device has two possible
two-qubits gates that can be used.

*  Square root of ISWAP. The gate `cirq.ISWAP ** 0.5` or `cirq.ISWAP ** -0.5` can be
used on `cirq.google.optimized_for_sycamore` with optimizer type `sqrt_iswap`
*  Sycamore gate. This gate, equivalent to FSimGate(π/2, π/6) can be used as `cirq.google.SYC`
or by using `cirq.FsimGate(numpy.pi/2,numpy.pi/6)`. Circuits can be compiled to use this gate
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
10-----KL-----
```

It can be accessing by using `cirq.google.Bristlecone`. Circuits can be compiled to it by using
`cirq.google.optimized_for_xmon` or by using `cirq.google.optimized_for_sycamore` with
optimizer_type `xmon`.

### Foxtail

The Foxtail device is a 2 by 11 XMON device arranged in a bilinear array,
addressable by using grid qubits `({0,1}, {0-11})`. It was one of the first
super-conducting quantum devices announced by Google. Due to the small number of qubits
and limited connectivity, it is still interesting for exploring the space of constrained
algorithms on NISQ devices.

It can be accessing by using `cirq.google.Foxtail`. Circuits can be compiled to it by using
`cirq.google.optimized_for_xmon` or by using `cirq.google.optimized_for_sycamore` with
optimizer_type `xmon`.





