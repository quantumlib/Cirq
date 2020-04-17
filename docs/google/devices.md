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





