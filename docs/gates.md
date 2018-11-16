## Gates

A ``Gate`` is an operation that can be applied to a collection of 
qubits (objects with a ``QubitId``).  ``Gates`` can be applied
to qubits by calling their ``on`` method, or, alternatively
calling the gate on the qubits.  The object created by such calls
is an ``Operation``.
```python
from cirq.ops import CNOT
from cirq.devices import GridQubit
q0, q1 = (GridQubit(0, 0), GridQubit(0, 1))
print(CNOT.on(q0, q1))
print(CNOT(q0, q1))
# prints
# CNOT((0, 0), (0, 1))
# CNOT((0, 0), (0, 1))
```

### Magic Methods

A class that implements ``Gate`` can be applied to qubits to produce an ``Operation``.
In order to support functionality beyond that basic task, it is necessary to implement several *magic methods*.

Standard magic methods in python are `__add__`, `__eq__`, and `__len__`.
Cirq defines several additional magic methods, for functionality such as parameterization, diagramming, and simulation.
For example, if a gate specifies a `_unitary_` method that returns a matrix for the gate, then simulators will be able to simulate applying the gate.
Or, if a gate specifies a `__pow__` method that works for an exponent of -1, then `cirq.inverse` will start to work on lists including the gate.

We describe some magic methods below.

#### `cirq.unitary` and `def _unitary_` 

When an object can be described by a unitary matrix, it can expose that unitary
matrix by implementing a `_unitary_(self) -> np.ndarray` method.
Callers can query whether or not an object has a unitary matrix by calling
`cirq.unitary` on it.
The `_unitary_` method may also return `NotImplemented`, in which case
`cirq.unitary` behaves as if the method is not implemented.

#### `cirq.decompose` and `def _decompose_`

Operations and gates can be defined in terms of other operations by implementing a `_decompose_` method that returns those other operations.
Operations implement `_decompose_(self)` whereas gates implement `_decompose_(self, qubits)` (since gates don't know their qubits ahead of time).

The main requirements on the output of `_decompose_` methods are:

1. DO NOT CREATE CYCLES. The `cirq.decompose` method will iterative decompose until it find values satisfying a `keep` predicate. Cycles cause it to enter an infinite loop.
2. Head towards operations defined by cirq, because these operations have good decomposition methods that terminate in single-qubit and two-qubit gates that can be understood by essentially all code.
3. All that matters is functional equivalence. Don't worry about staying within or reaching a particular gate set; it's too hard to predict what the caller will want. Gate-set-aware decomposition is useful, but *this is not the protocol that does that*. 

For example, `cirq.CCZ` decomposes into a series of `cirq.CNOT` and `cirq.T` operations.
This allows code that doesn't understand three-qubit operation to work with `cirq.CCZ`; by decomposing it into operations they do understand.
As another example, `cirq.TOFFOLI` decomposes into a `cirq.H` followed by a `cirq.CCZ` followed by a `cirq.H`.
Although the output contains a three qubit operation (the CCZ), that operation can be decomposed into two qubit and one qubit operations.
So code that doesn't understand three qubit operations can deal with Toffolis by decomposing it, and then decomposing the CCZ from the decomposition.

In general, code is expected to recursively decompose unknown operations until the code either hits operations it understands or hits a dead end where no more decomposition is possible.
The `cirq.decompose` method implements logic for performing exactly this kind of recursive decomposition.
Callers specify a `keep` predicate, and optionally specify intercepting and fallback decomposers, and then `cirq.decompose` will repeatedly decompose whatever operations it was given until the operations satisfy the given `keep`.
If `cirq.decompose` hits a dead end, it raises an error.

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

#### ExtrapolatableEffect

Represents an effect which can be scaled continuously up or down, or negated.
Implementing gates and operations implement the ``extrapolate_effect`` method, which takes a single float parameter ``factor``.
This factor is the amount to scale the gate by.
Roughly, one can think about this as applying the effect ``factor`` times.
There is some  subtlety in this definition since, for example, there are often two ways to define the square root of a gate.
It is up to the implementation to define which root is chosen.

The primary use of ``ExtrapolatableEffect`` is to allow
easy *powering* of gates.  That is one can define
for these gates a power
```python
import cirq
import numpy as np
print(np.around(cirq.unitary(cirq.X)))
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
Note that it is often the case that ``(g**a)**b != g**(a * b)``, due to the intermediate values normalizing rotation angles into a canonical range.

#### `_circuit_diagram_info_(self, args)` and `cirq.circuit_diagram_info(val, [args], [default])`

Circuit diagrams are useful for visualizing the structure of a `Circuit`.
Gates can specify compact representations to use in diagrams by implementing a `_circuit_diagram_info_` method.
For example, this is why SWAP gates are shown as linked '×' characters in diagrams.

The `_circuit_diagram_info_` method takes an `args` parameter of type `cirq.CircuitDiagramInfoArgs` and returns either
a string (typically the gate's name), a sequence of strings (a label to use on each qubit targeted by the gate), or an
instance of `cirq.CircuitDiagramInfo` (which can specify more advanced properties such as exponents and will expand
in the future).

You can query the circuit diagram info of a value by passing it into `cirq.circuit_diagram_info`.

### XmonGates

Google's Xmon devices support a specific gate set. Gates
in this gate set operate on ``GridQubit``s, which are qubits
arranged on a square grid and which have an ``x`` and ``y``
coordinate.

The ``XmonGates`` are

**cirq.PhasedXPowGate**
This gate is a rotation about an axis in the XY plane of the Bloch sphere.
The ``PhasedXPowGate`` takes two parameters, ``exponent`` and ``phase_exponent``.
The gate is equivalent to the circuit `───Z^-p───X^t───Z^p───` where `p` is the `phase_exponent` and `t` is the `exponent`.

**cirq.Z / cirq.Rz** Rotations about the Pauli ``Z`` axis.
The matrix of `cirq.Z**t` is ``exp(i pi |1><1| t)`` whereas the matrix of `cirq.Rz(θ)` is `exp(-i Z θ/2)`.
Note that in quantum computing hardware, this gate is often implemented in the
classical control hardware as a phase change on later operations, instead of as
a physical modification applied to the qubits.
(TODO: explain this in more detail)

**cirq.CZ** The controlled-Z gate.
A two qubit gate that phases the ``|11>`` state.
The matrix of `cirq.CZ**t` is ``exp(i pi |11><11| t)``.

**cirq.MeasurementGate** This is a single qubit measurement
in the computational basis. 

### CommonGates

``XmonGates`` are hardware specific.  In addition Cirq has a
number of more commonly named gates that are then implemented
as ``XmonGates`` via decomposition or known unitaries. Some
of these are our old friends:

**CNOT** the controlled-X gate, and **SWAP** the swap gate.
As well as some other Clifford friends, **H** and **S**,
and our error correcting friend **T**.

TODO: describe these in more detail.  
