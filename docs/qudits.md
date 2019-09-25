## Qudits

Most of the time in quantum computation, we work with qubits which are 2-level quantum systems.
A qu-*d*-it is a generalization of a qubit to a d-level or d-dimension system.
Qudits with known values for d have specific names.
A qubit has dimension 2, a qutrit has dimension 3, a ququart has dimension 4, and so on.
In Cirq, qudits work exactly like qubits except they have a `dimension` attribute other than 2 and they can only be used with gates specific to that dimension.
Both qubits and qudits are represented by a `Qid` object.

To apply a gate to some qudits, the dimensions of the qudits must match the dimensions it works on.  For example if a gate represents a unitary evolution on three qudits, a qubit, a qutrit, and another qutrit, the gate's "qid shape" is `(2, 3, 3)` and its `on` method will accept exactly 3 `Qid`s with dimension 2, 3, and 3.  This is an example single qutrit gate used in a circuit:

```python
class QutritPlusGate(cirq.SingleQubitGate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]])

    def _circuit_diagram_info_(self, args):
        return '[+1]'

q0 = cirq.LineQid(0, dimension=3)
circuit = cirq.Circuit.from_ops(
    QutritPlusGate().on(q0)
)

print(circuit)
# prints
# 0 (d=3): ───[+1]───
```

### Qids

`Qid` is the type representing qubits and qudits.  By default a qid like `cirq.NamedQubit('a')` is a qubit.  To create a qutrit named 'a', specify the dimension with `cirq.NamedQubit('a').with_dimension(3)`.  In addition, the `LineQid` constructor supports a dimension argument directly `cirq.LineQid(0, dimension=4)`.

### `cirq.qid_shape` and `def _qid_shape_`

Quantum gates, operations, and other types that act on a sequence of qudits can specify the dimension of each qudit they act on by implementing the `_qid_shape_` magic method.
This method returns a tuple of integers corresponding to the required dimension of each qudit it operates on e.g. (2, 3, 3) means an object that acts on a qubit, a qutrit, and another qutrit.

When `Qid`s are used with `Gate`s, `Operation`s, and `Circuit`s, the dimension of each qid must match the corresponding entry in the qid shape.
An error is raised otherwise.

Callers can query the qid shape of an object or a list of `Qid`s by calling `cirq.qid_shape` on it.
By default, `cirq.qid_shape` will return the equivalent qid shape for qubits if `_qid_shape_` is not defined.
For a qubit-only gate the qid shape is a tuple of 2s containing one 2 for each qubit e.g. `(2,) * cirq.num_qubits(gate)`.

### Unitaries, Mixtures, and Channels

The magic methods `_unitary_`, `_apply_unitary_`, `_mixture_`, and `_channel_` used to define unitary operations, mixtures, and channels can be used with qudits with one caveat.
The matrix dimensions for qudits will be larger than for qubits based on the values of the qudit dimensions (the object's qid shape).
The size of the matrix is determined from the product of the qudit dimensions.  For example, a single qubit unitary is a 2x2 matrix whereas a single qutrit unitary is a 3x3 matrix.  A two qutrit unitary is a 9x9 matrix (3 * 3 = 9) and a qubit-ququart unitary is a 8x8 matrix (2 * 4 = 8).  The size of the matrices for mixtures and channels follow the same rule.

### Simulators and Samplers

Simulators like `cirq.Simulator` and `cirq.DensityMatrixSimulator` will return simulation results with larger matrices than the same size qubit circuit when simulating qudit circuits.
The size of the matrix is determined by the product of the dimensions of the qudits being simulated.
The state vector output of `cirq.Simulator` after simulating a circuit on a qubit, a qutrit, and a qutrit will have 2 * 3 * 3 = 18 elements.
Call `cirq.qid_shape(simulation_result)` to check the qudit dimensions.

Measurement results from running a qudit circuit are integers in the range `0` to `qid.dimension-1`.
