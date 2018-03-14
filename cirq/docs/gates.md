## Gates

A ``Gate`` is an operation that can be applied to a collection of 
qubits (objects with a ``QubitId``).  ``Gates`` can be applied
to qubits by calling their ``on`` method, or, alternatively
calling the gate on the qubits.  The object created by such calls
is an ``Operation``.
```python
from cirq.ops import CNOT
from cirq.google import XmonQubit
q0, q1 = (XmonQubit(0, 0), XmonQubit(0, 1))
print(CNOT.on(q0, q1))
print(CNOT(q0, q1))
# prints
# CNOT((0, 0), (0, 1))
# CNOT((0, 0), (0, 1))
```

### Gate Features

The raw ``Gate`` class itself simply describes that a ``Gate``
can be applied to qubits to produce an ``Operation``. We then
use marker classes for ``Gates`` indicated what additional
features a ``Gate`` has.  

For example, one feature is ``ReversibleGate``.  A ``Gate``
that implements this method is required to implement
the method ``inverse`` which returns the inverse gate.
Algorithms that then operate on gates can use 
``isinstance(gate, ReversibleGate)`` to determine whether
this gate implements this method, and use this method
in the algorithm. Note that sometimes you have been provided
a ``Gate`` that does not implement a feature that you care
about.  For this case we use the notion of an ``Extension``,
see below.  

We describe some gate features below.

#### ReversibleGate, SelfInverseGate

As described above a ``ReversibleGate`` implements th
``inverse`` method which returns a ``Gate`` that is the
inverse of the given ``Gate``.  ``SelfInverseGate`` is
a ``Gate`` for which the ``inverse`` is simply the ``Gate``
itself (so the feature ``SelfInverseGate`` doesn't need
to implement ``inverse``, it already just returns ``self``.)

#### ExtrapolatableGate

This is a gate which can be scaled *continuously* up 
or down.  These gates must implement the ``extrapolate_effect``
method which takes a single parameter ``factor`` which 
is a float. This factor is simply the amount to scale
the gate by. Roughly one can think about this as applying the
``Gate`` ``factor`` times.  Of course there is some 
sublty in this definion, since ``factor`` is a float, and,
for example, there are often two ways to define the square
root of a gate.  It is up to the implementation to 
define how this works.

The primary use of ``ExtrapolatableGate`` is to allow
easy *powering* of gates.  That is one can define
for these gates a power
```python
import numpy as np
from cirq.ops import X
print(np.around(X.matrix()))
# prints
# [[0.+0.j 1.-0.j]
#  [1.-0.j 0.+0.j]]

sqrt_x = X**0.5
print(sqrt_x.matrix())
# prints
# [[0.5+0.5j 0.5-0.5j]
#  [0.5-0.5j 0.5+0.5j]]
```

Note that it is often the case that ``(g**a)**b != g**(a * b)``,
since gates that have rotation angles often normalize these 
angles.

#### KnownMatrixGate

We've seen this above.  These are ``Gates`` which implement
the ``matrix`` method. This returns a numpy ``ndarray`` matrix
which is the unitary gate for this ``Gate``.

#### CompositeGate

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
``Simulators`` where an optional 

#### AsciiDiagramableGate

Ascii diagrams of ``Circuits`` are actually quite useful for 
visualizing the moment structure of a ``Circuit``. In order
for this to display in a compact form, it is best practice
to implement his feature.

### XmonGates

Google's Xmon qubits support a specific gate set. Gates
in this gate set operate on ``XmonQubit``s, which are qubits
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
``exp(i pi half_turns W(axis_half_turns) / 2)``.

**ExpZGate** This gate is a rotation about the Pauli ``Z``
axis.  The gate is ``exp(i pi Z half_turns / 2)`` where
``half_turns`` is the supplied parameter.  Note that in 
quantum computing hardware, this gate is often compiled
out of the circuit (TODO: explain this in more detail)

**Exp11Gate** This is a two qubit gate and is a rotation
about the ``|11><11|`` projector.  It takes a single parameter 
``half_turns`` and is the gate ``exp(i pi 11 half_turns)``.

**XmonMeasurementGate** This is a single qubit measurement
in the computational basis. 

### CommonGates

``XmonGates`` are hardware specific.  In addition Cirq has a
number of more commonly named gates that are then implemented
as ``XmonGates`` via an extension or composite gates.  Some
of these are our old friends:

**RotXGate**, **RotYGate**, **RotZGate**, **Rot11Gate**. 
These are non-parameterized gates corresponding to the 
Pauli rotations or (in the case of ``Rot11Gate`` a two
qubit rotation).  When not using ``ParameterizedValue``s
prefer to use these gates.

Our old friends the Paulis: **X**, **Y**, and **Z**. 
Some other two qubit fiends, **CZ** the controlled-Z gate,
**CNOT** the controlled-X gate, and **SWAP** the swap gate.
As well as some other Clifford friends, **H** and **S**,
and our error correcting friend **T**.

TODO: describe these in more detail.  

### Extensions

TODO
