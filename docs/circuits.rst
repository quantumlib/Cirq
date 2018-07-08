Circuits
========

Conceptual overview
-------------------

There are two primary representations of quantum programs in Cirq, each of
which are represented by a class: :class:`~cirq.Circuit` and
:class:`~cirq.Schedule`. Conceptually a Circuit object is very closely
related to the abstract quantum circuit model, while a Schedule object is a
like the abstract quantum circuit model but includes detailed timing
information.

Conceptually: a :class:`~cirq.Circuit` is a collection of ``Moments``. A
:class:`~cirq.Moment` is a collection of ``Operations`` that all act during
the same abstract time slice. An :class:`~cirq.Operation` is a
:class:`~cirq.Gate` that operates on a specific subset of ``Qubits``.

.. image:: CircuitMomentOperation.png

Let's unpack this.

At the base of this construction is the notion of a qubit.  In
Cirq, qubits are represented by subclasses of the :class:`~cirq.QubitId`
base class. Different subclasses of :class:`~cirq.QubitId` can be used
for different purposes.  For example the qubits that Google's
Xmon devices use are often arranged on the vertices of a
square grid.  For this the class :class:`~cirq.devices.GridQubit`
subclasses :class:`~cirq.QubitId`.   For example, we can create
a 3 by 3 grid of qubits using

.. code-block:: python

    qubits = [cirq.devices.GridQubit(x, y) for x in range(3) for y in range(3)]

    print(qubits[0])
    # prints "(0, 0)"

The next level up conceptually is the notion of a :class:`~cirq.Gate`.
A :class:`~cirq.Gate` represents a physical process that occurs on a
``Qubit``.  The important property of a :class:`~cirq.Gate` is that it
can be applied *on* to one or more qubits.  This can be done
via the :meth:`~cirq.Gate.on` method itself or via ``()`` and doing this
turns the :class:`~cirq.Gate` into an :class:`~cirq.Operation`.

.. code-block:: python

    # This is an Pauli X gate.
    x_gate = cirq.X
    # Applying it to the qubit at location (0, 0) (defined above)
    # turns it into an operation.
    x_op = x_gate(qubits[0])

    print(x_op)
    # prints "X((0, 0))"

A :class:`~cirq.Moment` is quite simply a collection of operations, each of
which operates on a different set of qubits, and which conceptually
represents these operations as occurring during this abstract time
slice. The :class:`~cirq.Moment` structure itself is not required to be
related to the actual scheduling of the operations on a quantum
computer, or via a simulator, though it can be.  For example, here
is a Moment in which Pauli X and a CZ gate operate on three qubits:

.. code-block:: python

    cz = cirq.CZ(qubits[0], qubits[1])
    x = cirq.X(qubits[2])
    moment = cirq.Moment([x, cz])

    print(moment)
    # prints "X((0, 2)) and CZ((0, 0), (0, 1))"

Note that is not the only way to construct moments, nor even the
typical method, but illustrates that a :class:`~cirq.Moment` is just a
collection of operations on disjoint sets of qubits.

Finally at the top level a :class:`~cirq.Circuit` is an ordered series
of :class:`~cirq.Moment`s.  The first :class:`~cirq.Moment` in this series is,
conceptually, contains the first ``Operations`` that will be
applied.  Here, for example, is a simple circuit made up of
two moments

.. code-block:: python

    cz01 = cirq.CZ(qubits[0], qubits[1])
    x2 = cirq.X(qubits[2])
    cz12 = cirq.CZ(qubits[1], qubits[2])
    moment0 = cirq.Moment([cz01, x2])
    moment1 = cirq.Moment([cz12])
    circuit = cirq.Circuit((moment0, moment1))

    print(circuit)
    # prints the text diagram for the circuit:
    # (0, 0): ───@───────
    #            │
    # (0, 1): ───@───@───
    #                │
    # (0, 2): ───X───@───

Again, note that this is only one way to construct a :class:`~cirq.Circuit`
but illustrates the concept that a :class:`~cirq.Circuit` is an iterable
of ``Moments``.

Constructing Circuits
---------------------

Constructing ``Circuits`` as a series of ``Moments`` with each
:class:`~cirq.Moment` being hand-crafted is tedious. Instead we provide a
variety of different manners to create a :class:`~cirq.Circuit`.

One of the most useful ways to construct a :class:`~cirq.Circuit` is by
appending onto the :class:`~cirq.Circuit` with the
:meth:`~cirq.Circuit.append` method.

.. code-block:: python

    from cirq.ops import CZ, H
    q0, q1, q2 = [cirq.devices.GridQubit(i, 0) for i in range(3)]
    circuit = cirq.Circuit()
    circuit.append([CZ(q0, q1), H(q2)])

    print(circuit)
    # prints
    # (0, 0): ───@───
    #            │
    # (1, 0): ───@───
    #
    # (2, 0): ───H───

This appended an entire new moment to the qubit, which we can continue to do,

.. code-block:: python

    circuit.append([H(q0), CZ(q1, q2)])

    print(circuit)
    # prints
    # (0, 0): ───@───H───
    #            │
    # (1, 0): ───@───@───
    #                │
    # (2, 0): ───H───@───

In these two examples, we have appending full moments, what happens when we
append all of these at once?

.. code-block:: python

    circuit = cirq.Circuit()
    circuit.append([CZ(q0, q1), H(q2), H(q0), CZ(q1, q2)])

    print(circuit)
    # prints
    # (0, 0): ───@───H───
    #            │
    # (1, 0): ───@───@───
    #                │
    # (2, 0): ───H───@───

We see that here we have again created two ``Moments``. How did
:class:`~cirq.Circuit` know how to do this? ``Circuit's``
:meth:`~cirq.Circuit.append` method (and its cousin
:meth:`~cirq.Circuit.insert`) both take an argument called the
:class:`~cirq.InsertStrategy`. By default the :class:`~cirq.InsertStrategy`
is :attr:`~cirq.InsertStrategy.NEW_THEN_INLINE`.

InsertStrategies
^^^^^^^^^^^^^^^^

:class:`~cirq.InsertStrategy` defines how ``Operations`` are placed in a
:class:`~cirq.Circuit` when requested to be inserted at a given location.
Here a `location` is identified by the index of the :class:`~cirq.Moment` (in
the :class:`~cirq.Circuit`) where the insertion is requested to be placed at
(in the case of :meth:`~cirq.Circuit.append` this means inserting at the :class:`~cirq.Moment`
at an index one greater than the maximum moment index in the
:class:`~cirq.Circuit`). There are four such strategies:
:attr:`~cirq.InsertStrategy.EARLIEST`, :attr:`~cirq.InsertStrategy.NEW`,
:attr:`~cirq.InsertStrategy.INLINE` and
:attr:`~cirq.InsertStrategy.NEW_THEN_INLINE`.

:attr:`~cirq.InsertStrategy.EARLIEST` is defined as

    :attr:`~cirq.InsertStrategy.EARLIEST`: Scans backward from the insert
    location until a moment with operations touching qubits affected by the
    operation to insert is found. The operation is added into the moment just
    after that location.

For example, if we first create an :class:`~cirq.Operation` in a single moment,
and then use :attr:`~cirq.InsertStrategy.EARLIEST` the :class:`~cirq.Operation` can slide back to this
first :class:`~cirq.Moment` if there is space:

.. code-block:: python

    from cirq.circuits import InsertStrategy
    circuit = cirq.Circuit()
    circuit.append([CZ(q0, q1)])
    circuit.append([H(q0), H(q2)], strategy=InsertStrategy.EARLIEST)

    print(circuit)
    # prints
    # (0, 0): ───@───H───
    #            │
    # (1, 0): ───@───────
    #
    # (2, 0): ───H───────

After creating the first momemnt with a :class:`~cirq.CZ` gate, the second
append usese the :attr:`~cirq.InsertStrategy.EARLIEST` strategy. The
:class:`~cirq.H` on ``q0`` cannot slide back, while the :class:`~cirq.H` on
``q2`` can and so ends up in the first :class:`~cirq.Moment`.

Contrast this with the :attr:`~cirq.InsertStrategy.NEW`
:class:`~cirq.InsertStrategy`:

    :attr:`~cirq.InsertStrategy.NEW`: Every operation that is inserted is
    created in a new moment.

.. code-block:: python

    circuit = cirq.Circuit()
    circuit.append([H(q0), H(q1), H(q2)], strategy=InsertStrategy.NEW)

    print(circuit)
    # prints
    # (0, 0): ───H───────────
    #
    # (1, 0): ───────H───────
    #
    # (2, 0): ───────────H───

Here every operator processed by the append ends up in a new moment.
:attr:`~cirq.InsertStrategy.NEW` is most useful when you are inserting a single operation and
don't want it to interfere with other ``Moments``.

Another strategy is :attr:`~cirq.InsertStrategy.INLINE`:

    :attr:`~cirq.InsertStrategy.INLINE`: Attempts to add the operation to
    insert into the moment just before the desired insert location. But, if
    there's already an existing operation affecting any of the qubits touched
    by the operation to insert, a new moment is created instead.

.. code-block:: python

    circuit = cirq.Circuit()
    circuit.append([CZ(q1, q2)])
    circuit.append([CZ(q0,q1), H(q2), H(q0)], strategy=InsertStrategy.INLINE)

    print(circuit)
    # prints
    # (0, 0): ───────@───H───
    #                │
    # (1, 0): ───@───@───────
    #            │
    # (2, 0): ───@───H───────

After an initial :class:`~cirq.CZ` between the second and third qubit, we try
to insert 3 ``Operations``. We see that the :class:`~cirq.CZ` on the first
two qubits and the :class:`~cirq.H` on the third qubit are inserted into the
new :class:`~cirq.Moment`, but then the insert of :class:`~cirq.H` on the
first qubit cannot be insert into this :class:`~cirq.Moment`, so a new
:class:`~cirq.Moment` is created.

Finally we turn to the default strategy:

    :attr:`~cirq.InsertStrategy.NEW_THEN_INLINE`: Creates a new moment at the
    desired insert location for the first operation, but then switches to
    inserting operations according to :attr:`~cirq.InsertStrategy.INLINE`.

.. code-block:: python

    circuit = cirq.Circuit()
    circuit.append([H(q0)])
    circuit.append([CZ(q1,q2), H(q0)], strategy=InsertStrategy.NEW_THEN_INLINE)

    print(circuit)
    # prints
    # (0, 0): ───H───H───
    #
    # (1, 0): ───────@───
    #                │
    # (2, 0): ───────@───

The first append creates a single moment with a :class:`~cirq.H` on the first
qubit. Then the append with the :attr:`~cirq.InsertStrategy.NEW_THEN_INLINE`
strategy begins by inserting the :class:`~cirq.CZ` in a new
:class:`~cirq.Moment` (the :attr:`~cirq.InsertStrategy.NEW` in
:attr:`~cirq.InsertStrategy.NEW_THEN_INLINE`). Subsequent appending is done
:attr:`~cirq.InsertStrategy.INLINE` so the next :class:`~cirq.H` on the first
qubit is appending in the just created :class:`~cirq.Moment`.

Here is a helpful diagram for the different ``InsertStrategies``

TODO(dabacon): diagram.


Patterns for Arguments to Append and Insert
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Above we have used a series of :meth:`~cirq.Circuit.append` calls with a list
of different ``Operations`` we are adding to the circuit. But the argument
where we have supplied a list can also take more than just :class:`list`
values.

Example:

.. code-block:: python

    def my_layer():
        yield CZ(q0, q1)
        yield [H(q) for q in (q0, q1, q2)]
        yield [CZ(q1, q2)]
        yield [H(q0), [CZ(q1, q2)]]

    circuit = cirq.Circuit()
    circuit.append(my_layer())

    for x in my_layer():
        print(x)
    # prints
    # CZ((0, 0), (1, 0))
    # [Operation(H, (GridQubit(0, 0),)), Operation(H, (GridQubit(1, 0),)), Operation(H, (GridQubit(2, 0),))]
    # [Operation(CZ, (GridQubit(1, 0), GridQubit(2, 0)))]
    # [Operation(H, (GridQubit(0, 0),)), [Operation(CZ, (GridQubit(1, 0), GridQubit(2, 0)))]]

    print(circuit)
    # prints
    # (0, 0): ───@───H───H───────
    #            │
    # (1, 0): ───@───H───@───@───
    #                    │   │
    # (2, 0): ───────H───@───@───

Recall that in Python functions that have a ``yield`` are *generators*.
Generators are functions that act as *iterators*. Above we see that we can
iterate over ``my_layer()``. We see that when we do this each of the
``yields`` produces what was yielded, and here these are ``Operations``,
lists of ``Operations`` or lists of ``Operations`` mixed with lists of
``Operations``. But when we pass this iterator to the append method,
something magical happens. :class:`~cirq.Circuit` is able to flatten all of
these an pass them as one giant list to :meth:`~cirq.Circuit.append` (this
also works for :meth:`~cirq.Circuit.insert`).

.. note::

    The above idea uses a concept we call an ``OP_TREE``. An ``OP_TREE`` is
    not a class, but a contract. The basic idea is that, if the input can be
    iteratively flattened into a list of operations, then the input is an
    ``OP_TREE``.

A very nice pattern emerges from this structure: define
*generators* for sub-circuits, which can vary by size
or :class:`~cirq.Operation` parameters.

Another useful method is to construct a :class:`~cirq.Circuit` fully formed
from an ``OP_TREE`` via the static method :meth:`~cirq.Circuit.from_ops`
(which takes an insertion strategy as a parameter):

.. code-block:: python

    circuit = cirq.Circuit.from_ops(H(q0), H(q1))
    print(circuit)
    # prints
    # (0, 0): ───H───
    #
    # (1, 0): ───H───


Slicing and Iterating over Circuits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Circuits`` can be iterated over and sliced. When they are iterated
over each item in the iteration is a moment:

.. code-block:: python

    circuit = cirq.Circuit.from_ops(H(q0), CZ(q0, q1))
    for moment in circuit:
        print(moment)
    # prints
    # H((0, 0))
    # CZ((0, 0), (1, 0))

Slicing a :class:`~cirq.Circuit` on the other hand, produces a new
:class:`~cirq.Circuit` with only the moments corresponding to the slice:

.. code-block:: python

    circuit = cirq.Circuit.from_ops(H(q0), CZ(q0, q1), H(q1), CZ(q0, q1))
    print(circuit[1:3])
    # prints
    # (0, 0): ───@───────
    #            │
    # (1, 0): ───@───H───

Especially useful is dropping the last moment (which is often just
measurements): ``circuit[:-1]``, or reversing a circuit:
``circuit[::-1]``.
