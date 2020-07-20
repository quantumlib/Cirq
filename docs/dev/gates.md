# Gate and Operation Guidelines

This developer document explains what is expected of a gate or operation exposed
by Cirq.
In particular, we have a stricter standard than what is required of users of the
library.

For a user of Cirq, specifying either a `_unitary_` method or a `_decompose_`
method is sufficient to get a gate working.
Most other protocols will infer what they need from these two methods.
A gate specified in this way will not be particularly performant, but it will
work.
For internal gates, we also want high performance, and so we require several
other protocol methods to be implemented.

In general, the source of truth for what has to be implemented is enforced
by the `cirq.testing.assert_implements_consistent_protocols` method.
This method verifies the following properties:

1. The class has a `__repr__` method that produces a python expression that
evaluates to an object equal to the original value. The expression assumes that
`cirq`, `sympy`, `numpy as np`, and `pandas as pd` have been imported.

2. If the class is unitary, it specifies a `_has_unitary_` method.

3. The classes various protocols agree with each other.
For example, the decomposition that `_decompose_` produces should have the same
effect as the unitary produced by `_unitary_` or the transformation applied by
`_apply_unitary_`.

If the gate is exposed by `cirq/__init__.py` or another public module, other
tests will notice it and verify that it is serializable.
See the [serialization guidelines](serialization.md).

There are several other informal constraints:

1. Large gates should have a `_decompose_` method that returns
a composition of smaller gates.
This allows optimizers and other tools that cannot understand
the gate to break it into pieces that they do understand.

2. Gates should specify a good `_circuit_diagram_info_` method.
In some cases the default behavior of using `__str__` is sufficient.

3. Gates should have a good `__str__` method.

4. If the `__repr__` is cumbersome, gates should specify a `_repr_pretty_`
method.
This method will be used preferentially by Jupyter notebooks, ipython, etc.

5. Gates should specify an `_apply_unitary_` method.
This is not necessary for single or two qubit gates, but it is a huge
performance difference for larger gates.

6. Gates that take parameters (e.g. a rotation angle) should generally allow for
those parameters to be sympy objects instead of floats, and implement
corresponding `_is_parameterized_` and `_resolve_parameters_` methods.

7. Prefer creating a `Gate` over creating an `Operation`.
In some cases it makes sense to only have an `Operation`, but these cases are
generally surprising to users.
If you have to use an operation, try to have the `.gate` property of the
operation can return something useful instead of `None`.

8. Consider adding interop methods like `_qasm_`.
These methods will fallback to using things like `_decompose_`, but the output
is usually much better when specialized.
