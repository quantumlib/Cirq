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


#### `cirq.unitary` and `def _unitary_` 

When objects can be described by a unitary matrix, they let `Cirq` know by implementing the ``_unitary_`` method.
This method should return a numpy ``ndarray`` matrix and this array should be the unitary matrix corresponding to the object.
The method may also return `NotImplemented`, in which case `cirq.unitary` behaves as if the method is not implemented.


#### `cirq.decompose` and `def _decompose_`

A `cirq.Operation` indicates that it can be broken down into smaller simpler
operations by implementing a `def _decompose_(self):` method.
Code that doesn't understand a particular operation can call
`cirq.decompose_once` or `cirq.decompose` on that operation in order to get
a set of simpler operations that it does understand.

One useful thing about `cirq.decompose` is that it will decompose *recursively*,
until only operations meeting a `keep` predicate remain.
You can also give an `intercepting_decomposer` to `cirq.decompose`, which will
take priority over operations' own decompositions.

For `cirq.Gate`s, the decompose method is slightly different; it takes qubits:
`def _decompose_(self, qubits)`.
Callers who know the qubits that the gate is being applied to will use
`cirq.decompose_once_with_qubits` to trigger this method.

#### `_circuit_diagram_info_(self, args)` and `cirq.circuit_diagram_info(val, [args], [default])`

Circuit diagrams are useful for visualizing the structure of a `Circuit`.
Gates can specify compact representations to use in diagrams by implementing a `_circuit_diagram_info_` method.
For example, this is why SWAP gates are shown as linked '×' characters in diagrams.

The `_circuit_diagram_info_` method takes an `args` parameter of type `cirq.CircuitDiagramInfoArgs` and returns either
a string (typically the gate's name), a sequence of strings (a label to use on each qubit targeted by the gate), or an
instance of `cirq.CircuitDiagramInfo` (which can specify more advanced properties such as exponents and will expand
in the future).

You can query the circuit diagram info of a value by passing it into `cirq.circuit_diagram_info`.

### Xmon gates

Google's Xmon devices support a specific gate set. Gates
in this gate set operate on ``GridQubit``s, which are qubits
arranged on a square grid and which have an ``x`` and ``y``
coordinate.

The native Xmon gates are

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

### Common Gates

Cirq comes with a number of common named gates:

**CNOT** the controlled-X gate

**SWAP** the swap gate

**H** the Hadmard gate

**S** the square root of Z gate

and our error correcting friend **T**

TODO: describe these in more detail.  
