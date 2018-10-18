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

#### `cirq.unitary` and `def _unitary_` 

We've seen this above.
These are ``Gate`` or ``Operation`` instances which may be described by a
unitary matrix.
They implement the ``_unitary_`` method,
which returns a numpy ``ndarray`` matrix which is the unitary gate for the
gate/operation.
The method may also return `NotImplemented`, in which case `cirq.unitary`
behaves as if the method is not implemented.

#### CompositeGate and CompositeOperation

A ``CompositeGate`` is a gate which consists of multiple gates
that can be applied to a given set of qubits.  This is a manner
in which one can decompose one gate into multiple gates.  In
particular ``CompositeGates`` implement the method 
``default_decompose`` which acts on a sequence of qubits, and
returns a list of the operations acting on these qubits for
the constituents gates.  

One thing about ``CompositeGates`` is that sometimes you want
to modify the decomposition.  Algorithms that allow this can
take an ``Extension`` which allows for overriding the 
``CompositeGate``.  An example of this is for in 
``Simulators`` where an optional extension can be supplied
that can be used to override the CompositeGate.

A ``CompositeOperation`` is just like a ``CompositeGate``, except it already knows the qubits it should be applied to.

#### TextDiagrammable

Text diagrams of ``Circuits`` are actually quite useful for visualizing the moment structure of a ``Circuit``.
Gates that implement this feature can specify compact representations to use in the diagram (e.g. '×' instead of 'SWAP').

### XmonGates

Google's Xmon devices support a specific gate set. Gates
in this gate set operate on ``GridQubit``s, which are qubits
arranged on a square grid and which have an ``x`` and ``y``
coordinate.

The ``XmonGates`` are

**ExpWGate** This gate is a rotation about a combination of
a Pauli `X` and Pauli `Y` gates.  The ``ExpWGate`` takes
two parameters, ``half_turns`` and ``axis_half_turns``.  The
later describes the angle of the operator that is being
rotated about in the ``XY`` plane.  In particular if we define
``W(theta) = cos(pi theta) X + sin (pi theta) Y`` then
``axis_half_turns`` is ``theta``.  And the full gate is
``exp(-i pi half_turns W(axis_half_turns) / 2)``.

**ExpZGate** This gate is a rotation about the Pauli ``Z``
axis.  The gate is ``exp(-i pi Z half_turns / 2)`` where
``half_turns`` is the supplied parameter.  Note that in 
quantum computing hardware, this gate is often compiled
out of the circuit (TODO: explain this in more detail)

**cirq.Rot11Gate** This is a two qubit gate and is a rotation
about the ``|11><11|`` projector.  It takes a single parameter 
``half_turns`` and is the gate ``exp(i pi |11><11| half_turns)``.

**cirq.MeasurementGate** This is a single qubit measurement
in the computational basis. 

### CommonGates

``XmonGates`` are hardware specific.  In addition Cirq has a
number of more commonly named gates that are then implemented
as ``XmonGates`` via an extension or composite gates.  Some
of these are our old friends:

**RotXGate**, **RotYGate**, **RotZGate**, **Rot11Gate**. 
These are gates corresponding to the  Pauli rotations or
(in the case of ``Rot11Gate`` a two qubit rotation).

Our old friends the Paulis: **X**, **Y**, and **Z**. 
Some other two qubit fiends, **CZ** the controlled-Z gate,
**CNOT** the controlled-X gate, and **SWAP** the swap gate.
As well as some other Clifford friends, **H** and **S**,
and our error correcting friend **T**.

TODO: describe these in more detail.  

### Extensions

TODO
