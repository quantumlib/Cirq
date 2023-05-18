# Best Practices

This page describes some of the best practices when using the Cirq library.
Following these guidelines will help you write code that is more performant and
less likely to break from version to version.  Many of these rules apply to
other python libraries as well.

## Use top-level constructs

The Cirq library is designed so that important user-facing classes and objects
are exposed at the package level.  Avoid referencing module names within Cirq.

For instance, use `cirq.X` and **not** `cirq.ops.X`.  The second version will
break if we rename modules or move classes around from version to version.

## Do not use private member variables of classes

Any member of a class that is prefixed with an underscore is, by convention, a
private variable and should not be used outside of this class.  For instance,
`cirq.XPowGate._dimension` should not be accessed or modified, since it is a
private member of that class.  Using or modifying these values could result in
unexpected behavior or in broken code.

## Do not mutate "immutable" classes

While python's flexibility allows developers to modify just about anything, it
is bad practice to modify classes that are designed to be immutable.  Doing so
can violate assumptions made in other parts of the library.

In particular, attributes of `cirq.Gate`, `cirq.Operation`, `cirq.Moment`, and
`cirq.ParamResolver` should not be modified after creation.  If these objects
need to be modified, a new object should be created instead.

Violating this principle could cause problems in other parts of the code.  For
instance, changing the qubits of an `cirq.Operation` could cause a `cirq.Moment`
that contains this Operation to have two Operations with the same qubits (which
is not allowed).

Note that `Circuit` objects can be modified, but `FrozenCircuit` objects cannot.

## Be mindful of exponential scaling

Many algorithms and procedures in quantum computing scale in an exponential
pattern.  Cirq is designed for the noisy intermediate-scale quantum computing
(NISQ) regime.  Creating circuits with hundreds or thousands of qubits may
surpass the capabilities of this library.

Even with smaller numbers of qubits, simulation and other tasks can very quickly
consume resources and time.  The difference between a one second and an hour's
computation can be as few as ten qubits.

## What you see is what you get

Cirq tries to be as true to the specified circuits as possible, especially with
respect to hardware execution.  Cirq highly discourages any hidden automatic
decomposition, compilation, or other modification of a circuit that is unknown
to the user.  Any modification or transformation to the circuit should be
initiated by the user of the library.

This philosophy is important for many use cases.  For instance, certain
benchmarking algorithms rely on the fact that gate sequences will not be optimized,
even if the circuit is nominally inefficient.

Of course, Cirq provides routines and functions for compilation and
transformation of circuits.  Users can and should call these routines.  However,
Cirq and resulting hardware integrations should not modify the circuits without
the user's "permission".

## Other style and performance guidelines

*   Use `cirq.CircuitOperation` to more compactly define large, repeated
    circuits.  This can save space and time for analysis of larger circuits.
*   For hardware execution of multiple circuits, prefer using `run_sweep`  to
    run variants of circuits.  When not possible, try using `run_batch`.  Using
    these methods gives the hardware service the most opportunity to optimize
    the execution of circuits and can result in much faster execution.
    Read more details on the [Parameter Sweeps](../simulate/params.ipynb) page.
*   Consider defining and allocating qubits at the beginning of your code or
    function, then applying gates and circuits to those qubits.  While not
    required, this style can produce cleaner code.
