# Gates

A `Gate` is an effect that can be applied to a collection of
qubits (objects with a ``Qid``).  `Gates` can be applied
to qubits by calling their ``on`` method, or, alternatively
calling the gate on the qubits.  The object created by such calls
is an ``Operation``.  Alternatively, a `Gate` can be thought of
as a [factory](https://en.wikipedia.org/wiki/Factory_method_pattern)
that, given input qubits, generates an associated
`GateOperation` object.

## Gates versus Operations

![Structures in Cirq](moments.png)

The above example shows the first half of a quantum teleportation circuit,
found in many quantum computation textbooks.  This example uses three different
gates: a Hadamard (H), controlled-Not (CNOT), and measurement.  These are
represented in cirq by ```cirq.H```, ```cirq.CNOT```, and ``cirq.measure``,
respectively.

In this example, a Hadamard is an example of a `Gate` object that can be applied
in many different circumstances and to many different qubits.  Note that the
above example has two instances of an H gate but applied to different qubits.
This is an example of one `Gate` type with two `Operation` instances, one
applied to the qubit '|b⟩' and the other applied to qubit '|a⟩'.

Gates can generally be applied to any type of qubit (``NamedQubit``,
``LineQubit``, ``GridQubit``, etc) to create an Operation.
However, depending on the application, you may prefer a specific type
of qubit.  For instance, [Google devices](google/devices.md) generally use
``GridQubits``.  Other [devices](devices.md) may have connectivity constraints
that further restrict the set of qubits that can be used, especially in multi-
qubit operations. 

The following example shows how to construct each of these gates and operations.

```python
import cirq

# This examples uses named qubits to remain abstract.
# However, we can easily also LineQubits or GridQubits to specify a geometry
a = cirq.NamedQubit('a')
b = cirq.NamedQubit('b')
c = cirq.NamedQubit('c')

# Example Operations, that correspond to the moments above
print(cirq.H(b))
print(cirq.CNOT(b, c))
print(cirq.CNOT(a, b))
print(cirq.H(a))
print(cirq.measure(a,b))
# prints
# H(b)
# CNOT(b, c)
# CNOT(a, b)
# H(a)
# cirq.MeasurementGate(2, 'a,b', ())(a, b)
```

This would create the operations needed to comprise the circuit from the above
diagram.  The next step would be composing these operations into moments and
circuits.  For more on those types, see the documentation on
[Circuits](circuits.md).

## Other gate features

Most ``Gate``s operate on a specific number of qubits, which can be accessed
by the ``num_qubits()`` function.  One notable exception is the
``MeasurementGate`` which can be applied to a variable number of qubits.

Most gates also have a unitary matrix representation, which can be accessed
by ``cirq.unitary(gate)``.  

Not all `Gate`s correspond to unitary evolution. They may represent a
probabilistic mixture of unitaries, or a general quantum channel. The component
unitaries and associated probabilities of a mixture can be accessed by
`cirq.mixture(gate)`. The Kraus operator representation of a channel can be
accessed by `cirq.channel(gate)`. Non-unitary gates are often used in the
simulation of noise. See [noise documentation](noise.md) for more details.

Many arithmetic operators will work in the expected way when applied to
gates.  For instance, ``cirq.X**0.5`` represents a square root of X gate.
These can also be applied to Operators for a more compact representation, such
as ``cirq.X(q1)**0.5`` will be a square root of X gate applied to the q1
qubit.  This functionality depends on the "magic methods" of the gate
being defined (see below for details).

Gates can be converted to a controlled version by using `Gate.controlled()`.
In general, this returns an instance of a `ControlledGate`.  However, for
certain special cases where the controlled version of the gate is also a
known gate, this returns the instance of that gate.
For instance, `cirq.X.controlled()` returns a `cirq.CNOT` gate.
Operations have similar functionality `Operation.controlled_by()`, such as
`cirq.X(q0).controlled_by(q1)`.

## Common gates

Cirq supports a number of gates natively, with the opportunity to extend these
gates for more advanced use cases.

### Measurement gate

**cirq.MeasurementGate** This is a measurement in the computational basis.
This gate can be applied to a variable number of qubits.  The function
`cirq.measure(q0, q1, ...)` can also be used as a short-hand to create a
`MeasurementGate` .

### Single qubit gates

Most single-qubit gates can be thought of as rotation around an axis in the
[Bloch Sphere](https://en.wikipedia.org/wiki/Bloch_sphere) representation and
are usually referred to by their axis of rotation.


**cirq.Z / cirq.ZPowGate / cirq.rz** Rotations about the Pauli ``Z`` axis.
The matrix of `cirq.Z**t` and the equivalent representation
`cirq.ZPowGate(exponent=t)` is ``exp(i pi |1⟩⟨1| t)`` whereas the matrix of
`cirq.rz(θ)` is `exp(-i Z θ/2)`.  Since computation is often done in the
Z-basis, these are implemented as phase changes on later operations on
[Google devices](google/devices.md) instead of a physical modification applied
to the qubits.

**cirq.Y / cirq.YPowGate / cirq.ry** Rotations about the Pauli ``Y`` axis.
The gate `cirq.Y**t` and the equivalent representation
`cirq.YPowGate(exponent=t)` are measured in terms of 180 degree turns
(half turns), while the function `cirq.ry(θ)` uses a radian angle of rotation.

**cirq.X / cirq.XPowGate / cirq.rx** Rotations about the Pauli ``X`` axis.
The gate `cirq.X**t` and the equivalent representation
`cirq.XPowGate(exponent=t)` are measured in terms of 180 degree turns
(half turns), while the function `cirq.rx(θ)` uses a radian angle of rotation.

**cirq.PhasedXPowGate**
This gate is a rotation about an axis in the XY plane of the Bloch sphere.
The ``PhasedXPowGate`` takes two parameters, ``exponent`` and ``phase_exponent``.
The gate is equivalent to the circuit `───Z^-p───X^t───Z^p───` where `p` is the
`phase_exponent` and `t` is the `exponent`.

**cirq.H / cirq.HPowGate** The Hadamard gate is a rotation around the X+Z axis.
`cirq.HPowGate(exponent=t)` is a variable rotation of t turns around this
axis. `cirq.H` is a π rotation and is equivalent to
`cirq.HPowGate(exponent=1)`

**S** The square root of Z gate, equivalent to `cirq.Z**0.5`

**T** The fourth root of Z gate, equivalent to `cirq.Z**0.25`.


### Two qubit gates

**cirq.CZ / cirq.CZPowGate** The controlled-Z gate.  A two qubit gate that
phases the |11⟩ state.  `cirq.CZPowGate(exponent=y)` is equivalent to
`cirq.CZ**t` and has a matrix representation of ``exp(i pi |11⟩⟨11| t)``.

**cirq.CNOT / cirq.CNotPowGate** The controlled-X gate.  This gate swaps the
|11⟩ and |10⟩ states.  `cirq.CNotPowGate(exponent=t)` is equivalent
to `cirq.CNOT**t` .

**cirq.SWAP / cirq.SwapPowGate** The swap gate swaps the |01⟩ and |10⟩ states.
`cirq.SWAP**t` is the same as `cirq.SwapPowGate(exponent = t)`

**cirq.ISWAP / cirq.ISwapPowGate**  The iSwap gate swaps the |01⟩ and |10⟩
states and adds a relative phase of i.  `cirq.ISWAP**t` is the same as
`cirq.ISwapPowGate(exponent = t)`

**Parity gates**: The gates cirq.XX, cirq.YY, and cirq.ZZ are equivalent to
performing the equivalent one-qubit Pauli gates on both qubits.  The gates
cirq.XXPowGate, cirq.YYPowGate, and cirq.ZZPowGate are the powers of
these gates.  


### Other Gates

**cirq.MatrixGate**: A gate defined by its unitary matrix in the form of a
numpy ndarray.

**cirq.WaitGate**:  This gate does nothing for a specified `cirq.Duration`
amount of time.  This is useful for conducting T1 and T2 decay experiments. 

**cirq.CCNOT, cirq.CCX, cirq.TOFFOLI, cirq.CCXPowGate**: Three qubit gates
representing the controlled-controlled-X gates.

**cirq.CCZ, cirq.CCZPowGate**: Three qubit gates representing a
controlled-controlled-Z gate.

**CSWAP, CSwapGate, FREDKIN**: Three qubit gates representing a controlled-SWAP
gate.


## Advanced: Creating your own gates

If the above gates are not sufficient for your use case, it is fairly simple
to create your own gate.   In order to do so, you can define your class and
inherit the `cirq.Gate` class and define the functionality in your class.

At minimum, you will need to define either the ``_num_qubits_`` or
``_qid_shape_`` magic method to define the number of qubits (or qudits) used
in the gate.  For convenience one can use the ``SingleQubitGate``,
``TwoQubitGate``, and ``ThreeQubitGate`` classes for these common gate sizes.

Much of cirq relies on "magic methods", which are methods prefixed with one or
two underscores and used by cirq's protocols or built-in python methods.
For instance,  python translates `cirq.Z**0.25` into
`cirq.Z.__pow__(0.25)`.  Other uses are specific to cirq and are found in the
protocols subdirectory.  They are defined below.


### Magic Methods


#### Standard python magic methods

There are many standard magic methods in python.  Here are a few of the most
important ones used in cirq:
  * `__str__` for user-friendly string output and  `__repr__` which should be
  able to be evaluated by python to the object itself.
  * `__eq__` and `__hash__` which define whether objects are equal or not.  You
  can also use `cirq.value.value_equality` for objects that have a small list
  of sub-values that can be compared for equality.
  * Arithmetic functions such as `__pow__`, `__mul__`, `__add__` define the
  action of `**`, `*`, and `+` respectively.
   
#### `cirq.num_qubits` and `def _num_qubits_`

A `Gate` must implement the `_num_qubits_` (or `_qid_shape_`) method.
This method returns an integer and is used by `cirq.num_qubits` to determine
how many qubits this gate operates on.

#### `cirq.qid_shape` and `def _qid_shape_`

A qudit gate or operation must implement the `_qid_shape_` method that returns a
tuple of integers.  This method is used to determine how many qudits the gate or
operation operates on and what dimension each qudit must be.  If only the
`_num_qubits_` method is implemented, the object is assumed to operate only on
qubits. Callers can query the qid shape of the object by calling
`cirq.qid_shape` on it. See [qudit documentation](qudits.md) for more
information.

#### `cirq.unitary` and `def _unitary_`

When an object can be described by a unitary matrix, it can expose that unitary
matrix by implementing a `_unitary_(self) -> np.ndarray` method.
Callers can query whether or not an object has a unitary matrix by calling
`cirq.unitary` on it.
The `_unitary_` method may also return `NotImplemented`, in which case
`cirq.unitary` behaves as if the method is not implemented.

#### `cirq.decompose` and `def _decompose_`

Operations and gates can be defined in terms of other operations by implementing
a `_decompose_` method that returns those other operations. Operations implement
`_decompose_(self)` whereas gates implement `_decompose_(self, qubits)`
(since gates don't know their qubits ahead of time).

The main requirements on the output of `_decompose_` methods are:

1. DO NOT CREATE CYCLES. The `cirq.decompose` method will iterative decompose until it finds values satisfying a `keep` predicate. Cycles cause it to enter an infinite loop.
2. Head towards operations defined by Cirq, because these operations have good decomposition methods that terminate in single-qubit and two qubit gates.
These gates can be understood by the simulator, optimizers, and other code.
3. All that matters is functional equivalence.
Don't worry about staying within or reaching a particular gate set; it's too hard to predict what the caller will want. Gate-set-aware decomposition is useful, but *this is not the protocol that does that*.
Gate-set-aware decomposition may be added in the future, but doesn't exist within Cirq at the moment.

For example, `cirq.CCZ` decomposes into a series of `cirq.CNOT` and `cirq.T` operations.
This allows code that doesn't understand three-qubit operation to work with `cirq.CCZ`; by decomposing it into operations they do understand.
As another example, `cirq.TOFFOLI` decomposes into a `cirq.H` followed by a `cirq.CCZ` followed by a `cirq.H`.
Although the output contains a three qubit operation (the CCZ), that operation can be decomposed into two qubit and one qubit operations.
So code that doesn't understand three qubit operations can deal with Toffolis by decomposing them, and then decomposing the CCZs that result from the initial decomposition.

In general, decomposition-aware code consuming operations is expected to recursively decompose unknown operations until the code either hits operations it understands or hits a dead end where no more decomposition is possible.
The `cirq.decompose` method implements logic for performing exactly this kind of recursive decomposition.
Callers specify a `keep` predicate, and optionally specify intercepting and fallback decomposers, and then `cirq.decompose` will repeatedly decompose whatever operations it was given until the operations satisfy the given `keep`.
If `cirq.decompose` hits a dead end, it raises an error.

Cirq doesn't make any guarantees about the "target gate set" decomposition is heading towards.
`cirq.decompose` is not a method
Decompositions within Cirq happen to converge towards X, Y, Z, CZ, PhasedX, specified-matrix gates, and others.
But this set will vary from release to release, and so it is important for consumers of decompositions to look for generic properties of gates,
such as "two qubit gate with a unitary matrix", instead of specific gate types such as CZ gates.

#### `cirq.inverse` and `__pow__`

Gates and operations are considered to be *invertable* when they implement a `__pow__` method that returns a result besides `NotImplemented` for an exponent of -1.
This inverse can be accessed either directly as `value**-1`, or via the utility method `cirq.inverse(value)`.
If you are sure that `value` has an inverse, saying `value**-1` is more convenient than saying `cirq.inverse(value)`.
`cirq.inverse` is for cases where you aren't sure if `value` is invertable, or where `value` might be a *sequence* of invertible operations.

`cirq.inverse` has a `default` parameter used as a fallback when `value` isn't invertable.
For example, `cirq.inverse(value, default=None)` returns the inverse of `value`, or else returns `None` if `value` isn't invertable.
(If no `default` is specified and `value` isn't invertible, a `TypeError` is raised.)

When you give `cirq.inverse` a list, or any other kind of iterable thing, it will return a sequence of operations that (if run in order) undoes the operations of the original sequence (if run in order).
Basically, the items of the list are individually inverted and returned in reverse order.
For example, the expression `cirq.inverse([cirq.S(b), cirq.CNOT(a, b)])` will return the tuple `(cirq.CNOT(a, b), cirq.S(b)**-1)`.

Gates and operations can also return values beside `NotImplemented` from their `__pow__` method for exponents besides `-1`.
This pattern is used often by Cirq.
For example, the square root of X gate can be created by raising `cirq.X` to 0.5:

```python
import cirq
print(cirq.unitary(cirq.X))
# prints
# [[0.+0.j 1.+0.j]
#  [1.+0.j 0.+0.j]]

sqrt_x = cirq.X**0.5
print(cirq.unitary(sqrt_x))
# prints
# [[0.5+0.5j 0.5-0.5j]
#  [0.5-0.5j 0.5+0.5j]]
```

The Pauli gates included in Cirq use the convention ``Z**0.5 ≡ S ≡ np.diag(1, i)``, ``Z**-0.5 ≡ S**-1``, ``X**0.5 ≡ H·S·H``, and the square root of ``Y`` is inferred via the right hand rule.


#### `_circuit_diagram_info_(self, args)` and `cirq.circuit_diagram_info(val, [args], [default])`

Circuit diagrams are useful for visualizing the structure of a `Circuit`.
Gates can specify compact representations to use in diagrams by implementing a `_circuit_diagram_info_` method.
For example, this is why SWAP gates are shown as linked '×' characters in diagrams.

The `_circuit_diagram_info_` method takes an `args` parameter of type `cirq.CircuitDiagramInfoArgs` and returns either
a string (typically the gate's name), a sequence of strings (a label to use on each qubit targeted by the gate), or an
instance of `cirq.CircuitDiagramInfo` (which can specify more advanced properties such as exponents and will expand
in the future).

You can query the circuit diagram info of a value by passing it into `cirq.circuit_diagram_info`.
