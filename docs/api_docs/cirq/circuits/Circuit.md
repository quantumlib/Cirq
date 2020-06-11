<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.circuits.Circuit" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__pow__"/>
<meta itemprop="property" content="__radd__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="all_measurement_keys"/>
<meta itemprop="property" content="all_operations"/>
<meta itemprop="property" content="all_qubits"/>
<meta itemprop="property" content="append"/>
<meta itemprop="property" content="are_all_matches_terminal"/>
<meta itemprop="property" content="are_all_measurements_terminal"/>
<meta itemprop="property" content="batch_insert"/>
<meta itemprop="property" content="batch_insert_into"/>
<meta itemprop="property" content="batch_remove"/>
<meta itemprop="property" content="batch_replace"/>
<meta itemprop="property" content="clear_operations_touching"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="final_state_vector"/>
<meta itemprop="property" content="final_wavefunction"/>
<meta itemprop="property" content="findall_operations"/>
<meta itemprop="property" content="findall_operations_between"/>
<meta itemprop="property" content="findall_operations_until_blocked"/>
<meta itemprop="property" content="findall_operations_with_gate_type"/>
<meta itemprop="property" content="has_measurements"/>
<meta itemprop="property" content="insert"/>
<meta itemprop="property" content="insert_at_frontier"/>
<meta itemprop="property" content="insert_into_range"/>
<meta itemprop="property" content="next_moment_operating_on"/>
<meta itemprop="property" content="next_moments_operating_on"/>
<meta itemprop="property" content="operation_at"/>
<meta itemprop="property" content="prev_moment_operating_on"/>
<meta itemprop="property" content="qid_shape"/>
<meta itemprop="property" content="reachable_frontier_from"/>
<meta itemprop="property" content="save_qasm"/>
<meta itemprop="property" content="to_qasm"/>
<meta itemprop="property" content="to_quil"/>
<meta itemprop="property" content="to_text_diagram"/>
<meta itemprop="property" content="to_text_diagram_drawer"/>
<meta itemprop="property" content="transform_qubits"/>
<meta itemprop="property" content="unitary"/>
<meta itemprop="property" content="with_device"/>
<meta itemprop="property" content="with_noise"/>
<meta itemprop="property" content="zip"/>
</div>

# cirq.circuits.Circuit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A mutable list of groups of operations to apply to some qubits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.Circuit`, `cirq.circuits.circuit.Circuit`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.circuits.Circuit(
    strategy: "cirq.InsertStrategy" = cirq.circuits.InsertStrategy.EARLIEST,
    *contents,
    device: "cirq.Device" = cirq.devices.UNCONSTRAINED_DEVICE
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Methods returning information about the circuit:
    next_moment_operating_on
    prev_moment_operating_on
    next_moments_operating_on
    operation_at
    all_qubits
    all_operations
    findall_operations
    findall_operations_between
    findall_operations_until_blocked
    findall_operations_with_gate_type
    reachable_frontier_from
    has_measurements
    are_all_matches_terminal
    are_all_measurements_terminal
    unitary
    final_state_vector
    to_text_diagram
    to_text_diagram_drawer

#### Methods for mutation:

insert
append
insert_into_range
clear_operations_touching
batch_insert
batch_remove
batch_insert_into
insert_at_frontier


Circuits can also be iterated over,
    for moment in circuit:
        ...
and sliced,
    circuit[1:3] is a new Circuit made up of two moments, the first being
        circuit[1] and the second being circuit[2];
    circuit[:, qubit] is a new Circuit with the same moments, but with only
        those operations which act on the given Qubit;
    circuit[:, qubits], where 'qubits' is list of Qubits, is a new Circuit
        with the same moments, but only with those operations which touch
        any of the given qubits;
    circuit[1:3, qubit] is equivalent to circuit[1:3][:, qubit];
    circuit[1:3, qubits] is equivalent to circuit[1:3][:, qubits];
and concatenated,
    circuit1 + circuit2 is a new Circuit made up of the moments in circuit1
        followed by the moments in circuit2;
and multiplied by an integer,
    circuit * k is a new Circuit made up of the moments in circuit repeated
        k times.
and mutated,
    circuit[1:7] = [Moment(...)]

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`contents`
</td>
<td>
The initial list of moments and operations defining the
circuit. You can also pass in operations, lists of operations,
or generally anything meeting the <a href="../../cirq/ops/OP_TREE.md"><code>cirq.OP_TREE</code></a> contract.
Non-moment entries will be inserted according to the specified
insertion strategy.
</td>
</tr><tr>
<td>
`strategy`
</td>
<td>
When initializing the circuit with operations and moments
from `contents`, this determines how the operations are packed
together. This option does not affect later insertions into the
circuit.
</td>
</tr><tr>
<td>
`device`
</td>
<td>
Hardware that the circuit should be able to run on.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`device`
</td>
<td>

</td>
</tr><tr>
<td>
`moments`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="all_measurement_keys"><code>all_measurement_keys</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>all_measurement_keys() -> Tuple[str, ...]
</code></pre>




<h3 id="all_operations"><code>all_operations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>all_operations() -> Iterator[<a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>]
</code></pre>

Iterates over the operations applied by this circuit.

Operations from earlier moments will be iterated over first. Operations
within a moment are iterated in the order they were given to the
moment's constructor.

<h3 id="all_qubits"><code>all_qubits</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>all_qubits() -> FrozenSet['cirq.Qid']
</code></pre>

Returns the qubits acted upon by Operations in this circuit.


<h3 id="append"><code>append</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>append(
    moment_or_operation_tree: Union['cirq.Moment', 'cirq.OP_TREE'],
    strategy: "cirq.InsertStrategy" = cirq.circuits.InsertStrategy.EARLIEST
)
</code></pre>

Appends operations onto the end of the circuit.

Moments within the operation tree are appended intact.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`moment_or_operation_tree`
</td>
<td>
The moment or operation tree to append.
</td>
</tr><tr>
<td>
`strategy`
</td>
<td>
How to pick/create the moment to put operations into.
</td>
</tr>
</table>



<h3 id="are_all_matches_terminal"><code>are_all_matches_terminal</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>are_all_matches_terminal(
    predicate: Callable[['cirq.Operation'], bool]
)
</code></pre>

Check whether all of the ops that satisfy a predicate are terminal.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`predicate`
</td>
<td>
A predicate on ops.Operations which is being checked.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Whether or not all `Operation` s in a circuit that satisfy the
given predicate are terminal.
</td>
</tr>

</table>



<h3 id="are_all_measurements_terminal"><code>are_all_measurements_terminal</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>are_all_measurements_terminal()
</code></pre>

Whether all measurement gates are at the end of the circuit.


<h3 id="batch_insert"><code>batch_insert</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_insert(
    insertions: Iterable[Tuple[int, 'cirq.OP_TREE']]
) -> None
</code></pre>

Applies a batched insert operation to the circuit.

Transparently handles the fact that earlier insertions may shift
the index that later insertions should occur at. For example, if you
insert an operation at index 2 and at index 4, but the insert at index 2
causes a new moment to be created, then the insert at "4" will actually
occur at index 5 to account for the shift from the new moment.

All insertions are done with the strategy 'EARLIEST'.

When multiple inserts occur at the same index, the gates from the later
inserts end up before the gates from the earlier inserts (exactly as if
you'd called list.insert several times with the same index: the later
inserts shift the earliest inserts forward).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`insertions`
</td>
<td>
A sequence of (insert_index, operations) pairs
indicating operations to add into the circuit at specific
places.
</td>
</tr>
</table>



<h3 id="batch_insert_into"><code>batch_insert_into</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_insert_into(
    insert_intos: Iterable[Tuple[int, <a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>]]
) -> None
</code></pre>

Inserts operations into empty spaces in existing moments.

If any of the insertions fails (due to colliding with an existing
operation), this method fails without making any changes to the circuit.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`insert_intos`
</td>
<td>
A sequence of (moment_index, new_operation)
pairs indicating a moment to add a new operation into.
</td>
</tr>
</table>



#### ValueError:

One of the insertions collided with an existing operation.



#### IndexError:

Inserted into a moment index that doesn't exist.


<h3 id="batch_remove"><code>batch_remove</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_remove(
    removals: Iterable[Tuple[int, 'cirq.Operation']]
) -> None
</code></pre>

Removes several operations from a circuit.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`removals`
</td>
<td>
A sequence of (moment_index, operation) tuples indicating
operations to delete from the moments that are present. All
listed operations must actually be present or the edit will
fail (without making any changes to the circuit).
</td>
</tr>
</table>



#### ValueError:

One of the operations to delete wasn't present to start with.



#### IndexError:

Deleted from a moment that doesn't exist.


<h3 id="batch_replace"><code>batch_replace</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_replace(
    replacements: Iterable[Tuple[int, 'cirq.Operation', 'cirq.Operation']]
) -> None
</code></pre>

Replaces several operations in a circuit with new operations.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`replacements`
</td>
<td>
A sequence of (moment_index, old_op, new_op) tuples
indicating operations to be replaced in this circuit. All "old"
operations must actually be present or the edit will fail
(without making any changes to the circuit).
</td>
</tr>
</table>



#### ValueError:

One of the operations to replace wasn't present to start with.



#### IndexError:

Replaced in a moment that doesn't exist.


<h3 id="clear_operations_touching"><code>clear_operations_touching</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>clear_operations_touching(
    qubits: Iterable['cirq.Qid'],
    moment_indices: Iterable[int]
)
</code></pre>

Clears operations that are touching given qubits at given moments.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The qubits to check for operations on.
</td>
</tr><tr>
<td>
`moment_indices`
</td>
<td>
The indices of moments to check for operations
within.
</td>
</tr>
</table>



<h3 id="copy"><code>copy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy() -> "Circuit"
</code></pre>




<h3 id="final_state_vector"><code>final_state_vector</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>final_state_vector(
    initial_state: "cirq.STATE_VECTOR_LIKE" = 0,
    qubit_order: "cirq.QubitOrderOrList" = cirq.ops.QubitOrder.DEFAULT,
    qubits_that_should_be_present: Iterable['cirq.Qid'] = (),
    ignore_terminal_measurements: bool = True,
    dtype: Type[np.number] = np.complex128
) -> np.ndarray
</code></pre>

Left-multiplies a state vector by the circuit's unitary effect.

A circuit's "unitary effect" is the unitary matrix produced by
multiplying together all of its gates' unitary matrices. A circuit
with non-unitary gates (such as measurement or parameterized gates) does
not have a well-defined unitary effect, and the method will fail if such
operations are present.

For convenience, terminal measurements are automatically ignored
instead of causing a failure. Set the `ignore_terminal_measurements`
argument to False to disable this behavior.

This method is equivalent to left-multiplying the input state by
<a href="../../cirq/protocols/unitary.md"><code>cirq.unitary(circuit)</code></a> but it's computed in a more efficient
way.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`initial_state`
</td>
<td>
The input state for the circuit. This can be a list
of qudit values, a big endian int encoding the qudit values,
a vector of amplitudes, or a tensor of amplitudes.

When this is an int, it refers to a computational
basis state (e.g. 5 means initialize to ``|5⟩ = |...000101⟩``).

If this is a vector of amplitudes (a flat numpy array of the
correct length for the system) or a tensor of amplitudes (a
numpy array whose shape equals this circuit's `qid_shape`), it
directly specifies the initial state's amplitudes. The vector
type must be convertible to the given `dtype` argument.
</td>
</tr><tr>
<td>
`qubit_order`
</td>
<td>
Determines how qubits are ordered when passing matrices
into np.kron.
</td>
</tr><tr>
<td>
`qubits_that_should_be_present`
</td>
<td>
Qubits that may or may not appear
in operations within the circuit, but that should be included
regardless when generating the matrix.
</td>
</tr><tr>
<td>
`ignore_terminal_measurements`
</td>
<td>
When set, measurements at the end of
the circuit are ignored instead of causing the method to
fail.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The numpy dtype for the returned unitary. Defaults to
np.complex128. Specifying np.complex64 will run faster at the
cost of precision. `dtype` must be a complex np.dtype, unless
all operations in the circuit have unitary matrices with
exclusively real coefficients (e.g. an H + TOFFOLI circuit).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A (possibly gigantic) numpy array storing the superposition that
came out of the circuit for the given input state.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
The circuit contains measurement gates that are not
ignored.
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
The circuit contains gates that don't have a known
unitary matrix, e.g. gates parameterized by a Symbol.
</td>
</tr>
</table>



<h3 id="final_wavefunction"><code>final_wavefunction</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>final_wavefunction(
    initial_state: "cirq.STATE_VECTOR_LIKE" = 0,
    qubit_order: "cirq.QubitOrderOrList" = cirq.ops.QubitOrder.DEFAULT,
    qubits_that_should_be_present: Iterable['cirq.Qid'] = (),
    ignore_terminal_measurements: bool = True,
    dtype: Type[np.number] = np.complex128
) -> np.ndarray
</code></pre>

THIS FUNCTION IS DEPRECATED.

IT WILL BE REMOVED IN `cirq v0.10.0`.

Use final_state_vector instead.

Deprecated. Please use `final_state_vector`.

<h3 id="findall_operations"><code>findall_operations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>findall_operations(
    predicate: Callable[['cirq.Operation'], bool]
) -> Iterable[Tuple[int, 'cirq.Operation']]
</code></pre>

Find the locations of all operations that satisfy a given condition.

This returns an iterator of (index, operation) tuples where each
operation satisfies op_cond(operation) is truthy. The indices are
in order of the moments and then order of the ops within that moment.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`predicate`
</td>
<td>
A method that takes an Operation and returns a Truthy
value indicating the operation meets the find condition.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An iterator (index, operation)'s that satisfy the op_condition.
</td>
</tr>

</table>



<h3 id="findall_operations_between"><code>findall_operations_between</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>findall_operations_between(
    start_frontier: Dict['cirq.Qid', int],
    end_frontier: Dict['cirq.Qid', int],
    omit_crossing_operations: bool = False
) -> List[Tuple[int, 'cirq.Operation']]
</code></pre>

Finds operations between the two given frontiers.

If a qubit is in `start_frontier` but not `end_frontier`, its end index
defaults to the end of the circuit. If a qubit is in `end_frontier` but
not `start_frontier`, its start index defaults to the start of the
circuit. Operations on qubits not mentioned in either frontier are not
included in the results.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`start_frontier`
</td>
<td>
Just before where to start searching for operations,
for each qubit of interest. Start frontier indices are
inclusive.
</td>
</tr><tr>
<td>
`end_frontier`
</td>
<td>
Just before where to stop searching for operations,
for each qubit of interest. End frontier indices are exclusive.
</td>
</tr><tr>
<td>
`omit_crossing_operations`
</td>
<td>
Determines whether or not operations that
cross from a location between the two frontiers to a location
outside the two frontiers are included or excluded. (Operations
completely inside are always included, and operations completely
outside are always excluded.)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of tuples. Each tuple describes an operation found between
the two frontiers. The first item of each tuple is the index of the
moment containing the operation, and the second item is the
operation itself. The list is sorted so that the moment index
increases monotonically.
</td>
</tr>

</table>



<h3 id="findall_operations_until_blocked"><code>findall_operations_until_blocked</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>findall_operations_until_blocked(
    start_frontier: Dict['cirq.Qid', int],
    is_blocker: Callable[['cirq.Operation'], bool] = (lambda op: False)
) -> List[Tuple[int, <a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>]]
</code></pre>

Finds all operations until a blocking operation is hit.

An operation is considered blocking if

a) It is in the 'light cone' of start_frontier.

AND

(

    1) is_blocker returns a truthy value.

    OR

    2) It acts on a blocked qubit.
)

Every qubit acted on by a blocking operation is thereafter itself
blocked.


The notion of reachability here differs from that in
reachable_frontier_from in two respects:

1) An operation is not considered blocking only because it is in a
    moment before the start_frontier of one of the qubits on which it
    acts.
2) Operations that act on qubits not in start_frontier are not
    automatically blocking.

For every (moment_index, operation) returned:

1) moment_index >= min((start_frontier[q] for q in operation.qubits
    if q in start_frontier), default=0)
2) set(operation.qubits).intersection(start_frontier)

Below are some examples, where on the left the opening parentheses show
`start_frontier` and on the right are the operations included (with
their moment indices) in the output. `F` and `T` indicate that
`is_blocker` return `False` or `True`, respectively, when applied to
the gates; `M` indicates that it doesn't matter.


─(─F───F───────    ┄(─F───F─)┄┄┄┄┄
   │   │              │   │
─(─F───F───T─── => ┄(─F───F─)┄┄┄┄┄
           │                  ┊
───────────T───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


───M─────(─F───    ┄┄┄┄┄┄┄┄┄(─F─)┄┄
   │       │          ┊       │
───M───M─(─F───    ┄┄┄┄┄┄┄┄┄(─F─)┄┄
       │        =>        ┊
───────M───M───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
           │                  ┊
───────────M───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


───M─(─────M───     ┄┄┄┄┄()┄┄┄┄┄┄┄┄
   │       │           ┊       ┊
───M─(─T───M───     ┄┄┄┄┄()┄┄┄┄┄┄┄┄
       │        =>         ┊
───────T───M───     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
           │                   ┊
───────────M───     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


─(─F───F───    ┄(─F───F─)┄
   │   │    =>    │   │
───F─(─F───    ┄(─F───F─)┄


─(─F───────────    ┄(─F─)┄┄┄┄┄┄┄┄┄
   │                  │
───F───F───────    ┄(─F─)┄┄┄┄┄┄┄┄┄
       │        =>        ┊
───────F───F───    ┄┄┄┄┄┄┄┄┄(─F─)┄
           │                  │
─(─────────F───    ┄┄┄┄┄┄┄┄┄(─F─)┄

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`start_frontier`
</td>
<td>
A starting set of reachable locations.
</td>
</tr><tr>
<td>
`is_blocker`
</td>
<td>
A predicate that determines if operations block
reachability. Any location covered by an operation that causes
`is_blocker` to return True is considered to be an unreachable
location.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of tuples. Each tuple describes an operation found between
the start frontier and a blocking operation. The first item of
each tuple is the index of the moment containing the operation,
and the second item is the operation itself.
</td>
</tr>

</table>



<h3 id="findall_operations_with_gate_type"><code>findall_operations_with_gate_type</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>findall_operations_with_gate_type(
    gate_type: Type[<a href="../../cirq/circuits/circuit/T_DESIRED_GATE_TYPE.md"><code>cirq.circuits.circuit.T_DESIRED_GATE_TYPE</code></a>]
) -> Iterable[Tuple[int, <a href="../../cirq/ops/GateOperation.md"><code>cirq.ops.GateOperation</code></a>, <a href="../../cirq/circuits/circuit/T_DESIRED_GATE_TYPE.md"><code>cirq.circuits.circuit.T_DESIRED_GATE_TYPE</code></a>]]
</code></pre>

Find the locations of all gate operations of a given type.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`gate_type`
</td>
<td>
The type of gate to find, e.g. XPowGate or
MeasurementGate.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An iterator (index, operation, gate)'s for operations with the given
gate type.
</td>
</tr>

</table>



<h3 id="has_measurements"><code>has_measurements</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>has_measurements()
</code></pre>




<h3 id="insert"><code>insert</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>insert(
    index: int,
    moment_or_operation_tree: Union['cirq.Operation', 'cirq.OP_TREE'],
    strategy: "cirq.InsertStrategy" = cirq.circuits.InsertStrategy.EARLIEST
) -> int
</code></pre>

Inserts operations into the circuit.
    Operations are inserted into the moment specified by the index and
    'InsertStrategy'.
    Moments within the operation tree are inserted intact.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`index`
</td>
<td>
The index to insert all of the operations at.
</td>
</tr><tr>
<td>
`moment_or_operation_tree`
</td>
<td>
The moment or operation tree to insert.
</td>
</tr><tr>
<td>
`strategy`
</td>
<td>
How to pick/create the moment to put operations into.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The insertion index that will place operations just after the
operations that were inserted by this method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
Bad insertion strategy.
</td>
</tr>
</table>



<h3 id="insert_at_frontier"><code>insert_at_frontier</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>insert_at_frontier(
    operations: "cirq.OP_TREE",
    start: int,
    frontier: Dict['cirq.Qid', int] = None
) -> Dict['cirq.Qid', int]
</code></pre>

Inserts operations inline at frontier.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`operations`
</td>
<td>
the operations to insert
</td>
</tr><tr>
<td>
`start`
</td>
<td>
the moment at which to start inserting the operations
</td>
</tr><tr>
<td>
`frontier`
</td>
<td>
frontier[q] is the earliest moment in which an operation
acting on qubit q can be placed.
</td>
</tr>
</table>



<h3 id="insert_into_range"><code>insert_into_range</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>insert_into_range(
    operations: "cirq.OP_TREE",
    start: int,
    end: int
) -> int
</code></pre>

Writes operations inline into an area of the circuit.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`start`
</td>
<td>
The start of the range (inclusive) to write the
given operations into.
</td>
</tr><tr>
<td>
`end`
</td>
<td>
The end of the range (exclusive) to write the given
operations into. If there are still operations remaining,
new moments are created to fit them.
</td>
</tr><tr>
<td>
`operations`
</td>
<td>
An operation or tree of operations to insert.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An insertion index that will place operations after the operations
that were inserted by this method.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`IndexError`
</td>
<td>
Bad inline_start and/or inline_end.
</td>
</tr>
</table>



<h3 id="next_moment_operating_on"><code>next_moment_operating_on</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_moment_operating_on(
    qubits: Iterable['cirq.Qid'],
    start_moment_index: int = 0,
    max_distance: int = None
) -> Optional[int]
</code></pre>

Finds the index of the next moment that touches the given qubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
We're looking for operations affecting any of these qubits.
</td>
</tr><tr>
<td>
`start_moment_index`
</td>
<td>
The starting point of the search.
</td>
</tr><tr>
<td>
`max_distance`
</td>
<td>
The number of moments (starting from the start index
and moving forward) to check. Defaults to no limit.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None if there is no matching moment, otherwise the index of the
earliest matching moment.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
negative max_distance.
</td>
</tr>
</table>



<h3 id="next_moments_operating_on"><code>next_moments_operating_on</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_moments_operating_on(
    qubits: Iterable['cirq.Qid'],
    start_moment_index: int = 0
) -> Dict['cirq.Qid', int]
</code></pre>

Finds the index of the next moment that touches each qubit.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The qubits to find the next moments acting on.
</td>
</tr><tr>
<td>
`start_moment_index`
</td>
<td>
The starting point of the search.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The index of the next moment that touches each qubit. If there
is no such moment, the next moment is specified as the number of
moments in the circuit. Equivalently, can be characterized as one
plus the index of the last moment after start_moment_index
(inclusive) that does *not* act on a given qubit.
</td>
</tr>

</table>



<h3 id="operation_at"><code>operation_at</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>operation_at(
    qubit: "cirq.Qid",
    moment_index: int
) -> Optional['cirq.Operation']
</code></pre>

Finds the operation on a qubit within a moment, if any.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubit`
</td>
<td>
The qubit to check for an operation on.
</td>
</tr><tr>
<td>
`moment_index`
</td>
<td>
The index of the moment to check for an operation
within. Allowed to be beyond the end of the circuit.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None if there is no operation on the qubit at the given moment, or
else the operation.
</td>
</tr>

</table>



<h3 id="prev_moment_operating_on"><code>prev_moment_operating_on</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>prev_moment_operating_on(
    qubits: Sequence['cirq.Qid'],
    end_moment_index: Optional[int] = None,
    max_distance: Optional[int] = None
) -> Optional[int]
</code></pre>

Finds the index of the next moment that touches the given qubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
We're looking for operations affecting any of these qubits.
</td>
</tr><tr>
<td>
`end_moment_index`
</td>
<td>
The moment index just after the starting point of
the reverse search. Defaults to the length of the list of
moments.
</td>
</tr><tr>
<td>
`max_distance`
</td>
<td>
The number of moments (starting just before from the
end index and moving backward) to check. Defaults to no limit.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None if there is no matching moment, otherwise the index of the
latest matching moment.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
negative max_distance.
</td>
</tr>
</table>



<h3 id="qid_shape"><code>qid_shape</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qid_shape(
    qubit_order: "cirq.QubitOrderOrList" = cirq.ops.QubitOrder.DEFAULT
) -> Tuple[int, ...]
</code></pre>




<h3 id="reachable_frontier_from"><code>reachable_frontier_from</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reachable_frontier_from(
    start_frontier: Dict['cirq.Qid', int],
    *,
    is_blocker: Callable[['cirq.Operation'], bool] = (lambda op: False)
) -> Dict['cirq.Qid', int]
</code></pre>

Determines how far can be reached into a circuit under certain rules.

The location L = (qubit, moment_index) is *reachable* if and only if:

    a) There is not a blocking operation covering L.

    AND

    [
        b1) qubit is in start frontier and moment_index =
            max(start_frontier[qubit], 0).

        OR

        b2) There is no operation at L and prev(L) = (qubit,
            moment_index-1) is reachable.

        OR

        b3) There is an (non-blocking) operation P covering L such that
            (q', moment_index - 1) is reachable for every q' on which P
            acts.
    ]

An operation in moment moment_index is blocking if

    a) `is_blocker` returns a truthy value.

    OR

    b) The operation acts on a qubit not in start_frontier.

    OR

    c) The operation acts on a qubit q such that start_frontier[q] >
        moment_index.

In other words, the reachable region extends forward through time along
each qubit in start_frontier until it hits a blocking operation. Any
location involving a qubit not in start_frontier is unreachable.

For each qubit q in `start_frontier`, the reachable locations will
correspond to a contiguous range starting at start_frontier[q] and
ending just before some index end_q. The result of this method is a
dictionary, and that dictionary maps each qubit q to its end_q.

#### Examples:


If start_frontier is {
    cirq.LineQubit(0): 6,
    cirq.LineQubit(1): 2,
    cirq.LineQubit(2): 2,
} then the reachable wire locations in the following circuit are
highlighted with '█' characters:

    0   1   2   3   4   5   6   7   8   9   10  11  12  13

* <b>`0`</b>: ───H───@─────────────────█████████████████████─@───H───
          │                                       │
* <b>`1`</b>: ───────@─██H███@██████████████████████─@───H───@───────
                  │                       │
* <b>`2`</b>: ─────────██████@███H██─@───────@───H───@───────────────
                          │       │
* <b>`3`</b>: ───────────────────────@───H───@───────────────────────

And the computed end_frontier is {
    cirq.LineQubit(0): 11,
    cirq.LineQubit(1): 9,
    cirq.LineQubit(2): 6,
}

Note that the frontier indices (shown above the circuit) are
best thought of (and shown) as happening *between* moment indices.

If we specify a blocker as follows:

    is_blocker=lambda: op == cirq.CZ(cirq.LineQubit(1),
                                     cirq.LineQubit(2))

and use this start_frontier:

    {
        cirq.LineQubit(0): 0,
        cirq.LineQubit(1): 0,
        cirq.LineQubit(2): 0,
        cirq.LineQubit(3): 0,
    }

Then this is the reachable area:

    0   1   2   3   4   5   6   7   8   9   10  11  12  13
* <b>`0`</b>: ─██H███@██████████████████████████████████████─@───H───
          │                                       │
* <b>`1`</b>: ─██████@███H██─@───────────────────────@───H───@───────
                  │                       │
* <b>`2`</b>: ─█████████████─@───H───@───────@───H───@───────────────
                          │       │
* <b>`3`</b>: ─█████████████████████─@───H───@───────────────────────

and the computed end_frontier is:

    {
        cirq.LineQubit(0): 11,
        cirq.LineQubit(1): 3,
        cirq.LineQubit(2): 3,
        cirq.LineQubit(3): 5,
    }


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`start_frontier`
</td>
<td>
A starting set of reachable locations.
</td>
</tr><tr>
<td>
`is_blocker`
</td>
<td>
A predicate that determines if operations block
reachability. Any location covered by an operation that causes
`is_blocker` to return True is considered to be an unreachable
location.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An end_frontier dictionary, containing an end index for each qubit q
mapped to a start index by the given `start_frontier` dictionary.

To determine if a location (q, i) was reachable, you can use
this expression:

q in start_frontier and start_frontier[q] <= i < end_frontier[q]

where i is the moment index, q is the qubit, and end_frontier is the
result of this method.
</td>
</tr>

</table>



<h3 id="save_qasm"><code>save_qasm</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_qasm(
    file_path: Union[str, bytes, int],
    header: Optional[str] = None,
    precision: int = 10,
    qubit_order: "cirq.QubitOrderOrList" = cirq.ops.QubitOrder.DEFAULT
) -> None
</code></pre>

Save a QASM file equivalent to the circuit.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`file_path`
</td>
<td>
The location of the file where the qasm will be written.
</td>
</tr><tr>
<td>
`header`
</td>
<td>
A multi-line string that is placed in a comment at the top
of the QASM. Defaults to a cirq version specifier.
</td>
</tr><tr>
<td>
`precision`
</td>
<td>
Number of digits to use when representing numbers.
</td>
</tr><tr>
<td>
`qubit_order`
</td>
<td>
Determines how qubits are ordered in the QASM
register.
</td>
</tr>
</table>



<h3 id="to_qasm"><code>to_qasm</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_qasm(
    header: Optional[str] = None,
    precision: int = 10,
    qubit_order: "cirq.QubitOrderOrList" = cirq.ops.QubitOrder.DEFAULT
) -> str
</code></pre>

Returns QASM equivalent to the circuit.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`header`
</td>
<td>
A multi-line string that is placed in a comment at the top
of the QASM. Defaults to a cirq version specifier.
</td>
</tr><tr>
<td>
`precision`
</td>
<td>
Number of digits to use when representing numbers.
</td>
</tr><tr>
<td>
`qubit_order`
</td>
<td>
Determines how qubits are ordered in the QASM
register.
</td>
</tr>
</table>



<h3 id="to_quil"><code>to_quil</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_quil(
    qubit_order: "cirq.QubitOrderOrList" = cirq.ops.QubitOrder.DEFAULT
) -> str
</code></pre>




<h3 id="to_text_diagram"><code>to_text_diagram</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_text_diagram(
    *,
    use_unicode_characters: bool = True,
    transpose: bool = False,
    include_tags: bool = True,
    precision: Optional[int] = 3,
    qubit_order: "cirq.QubitOrderOrList" = cirq.ops.QubitOrder.DEFAULT
) -> str
</code></pre>

Returns text containing a diagram describing the circuit.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`use_unicode_characters`
</td>
<td>
Determines if unicode characters are
allowed (as opposed to ascii-only diagrams).
</td>
</tr><tr>
<td>
`transpose`
</td>
<td>
Arranges qubit wires vertically instead of horizontally.
</td>
</tr><tr>
<td>
`include_tags`
</td>
<td>
Whether tags on TaggedOperations should be printed
</td>
</tr><tr>
<td>
`precision`
</td>
<td>
Number of digits to display in text diagram
</td>
</tr><tr>
<td>
`qubit_order`
</td>
<td>
Determines how qubits are ordered in the diagram.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The text diagram.
</td>
</tr>

</table>



<h3 id="to_text_diagram_drawer"><code>to_text_diagram_drawer</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_text_diagram_drawer(
    *,
    use_unicode_characters: bool = True,
    qubit_namer: Optional[Callable[['cirq.Qid'], str]] = None,
    transpose: bool = False,
    include_tags: bool = True,
    precision: Optional[int] = 3,
    qubit_order: "cirq.QubitOrderOrList" = cirq.ops.QubitOrder.DEFAULT,
    get_circuit_diagram_info: Optional[Callable[['cirq.Operation', 'cirq.CircuitDiagramInfoArgs'],
        'cirq.CircuitDiagramInfo']] = None
) -> <a href="../../cirq/circuits/TextDiagramDrawer.md"><code>cirq.circuits.TextDiagramDrawer</code></a>
</code></pre>

Returns a TextDiagramDrawer with the circuit drawn into it.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`use_unicode_characters`
</td>
<td>
Determines if unicode characters are
allowed (as opposed to ascii-only diagrams).
</td>
</tr><tr>
<td>
`qubit_namer`
</td>
<td>
Names qubits in diagram. Defaults to str.
</td>
</tr><tr>
<td>
`transpose`
</td>
<td>
Arranges qubit wires vertically instead of horizontally.
</td>
</tr><tr>
<td>
`precision`
</td>
<td>
Number of digits to use when representing numbers.
</td>
</tr><tr>
<td>
`qubit_order`
</td>
<td>
Determines how qubits are ordered in the diagram.
</td>
</tr><tr>
<td>
`get_circuit_diagram_info`
</td>
<td>
Gets circuit diagram info. Defaults to
protocol with fallback.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The TextDiagramDrawer instance.
</td>
</tr>

</table>



<h3 id="transform_qubits"><code>transform_qubits</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>transform_qubits(
    func: Callable[['cirq.Qid'], 'cirq.Qid'],
    *,
    new_device: "cirq.Device" = None
) -> "cirq.Circuit"
</code></pre>

Returns the same circuit, but with different qubits.

Note that this method does essentially the same thing as
<a href="../../cirq/circuits/Circuit.md#with_device"><code>cirq.Circuit.with_device</code></a>. It is included regardless because there are
also `transform_qubits` methods on <a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a> and <a href="../../cirq/ops/Moment.md"><code>cirq.Moment</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`func`
</td>
<td>
The function to use to turn each current qubit into a desired
new qubit.
</td>
</tr><tr>
<td>
`new_device`
</td>
<td>
The device to use for the new circuit, if different.
If this is not set, the new device defaults to the current
device.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The receiving circuit but with qubits transformed by the given
function, and with an updated device (if specified).
</td>
</tr>

</table>



<h3 id="unitary"><code>unitary</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unitary(
    qubit_order: "cirq.QubitOrderOrList" = cirq.ops.QubitOrder.DEFAULT,
    qubits_that_should_be_present: Iterable['cirq.Qid'] = (),
    ignore_terminal_measurements: bool = True,
    dtype: Type[np.number] = np.complex128
) -> np.ndarray
</code></pre>

Converts the circuit into a unitary matrix, if possible.

Returns the same result as <a href="../../cirq/protocols/unitary.md"><code>cirq.unitary</code></a>, but provides more options.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubit_order`
</td>
<td>
Determines how qubits are ordered when passing matrices
into np.kron.
</td>
</tr><tr>
<td>
`qubits_that_should_be_present`
</td>
<td>
Qubits that may or may not appear
in operations within the circuit, but that should be included
regardless when generating the matrix.
</td>
</tr><tr>
<td>
`ignore_terminal_measurements`
</td>
<td>
When set, measurements at the end of
the circuit are ignored instead of causing the method to
fail.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The numpy dtype for the returned unitary. Defaults to
np.complex128. Specifying np.complex64 will run faster at the
cost of precision. `dtype` must be a complex np.dtype, unless
all operations in the circuit have unitary matrices with
exclusively real coefficients (e.g. an H + TOFFOLI circuit).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A (possibly gigantic) 2d numpy array corresponding to a matrix
equivalent to the circuit's effect on a quantum state.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
The circuit contains measurement gates that are not
ignored.
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
The circuit contains gates that don't have a known
unitary matrix, e.g. gates parameterized by a Symbol.
</td>
</tr>
</table>



<h3 id="with_device"><code>with_device</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_device(
    new_device: "cirq.Device",
    qubit_mapping: Callable[['cirq.Qid'], 'cirq.Qid'] = (lambda e: e)
) -> "Circuit"
</code></pre>

Maps the current circuit onto a new device, and validates.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`new_device`
</td>
<td>
The new device that the circuit should be on.
</td>
</tr><tr>
<td>
`qubit_mapping`
</td>
<td>
How to translate qubits from the old device into
qubits on the new device.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The translated circuit.
</td>
</tr>

</table>



<h3 id="with_noise"><code>with_noise</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_noise(
    noise: "cirq.NOISE_MODEL_LIKE"
) -> "cirq.Circuit"
</code></pre>

Make a noisy version of the circuit.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`noise`
</td>
<td>
The noise model to use.  This describes the kind of noise to
add to the circuit.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A new circuit with the same moment structure but with new moments
inserted where needed when more than one noisy operation is
generated for an input operation.  Emptied moments are removed.
</td>
</tr>

</table>



<h3 id="zip"><code>zip</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>zip(
    *circuits
)
</code></pre>

Combines operations from circuits in a moment-by-moment fashion.

Moment k of the resulting circuit will have all operations from moment
k of each of the given circuits.

When the given circuits have different lengths, the shorter circuits are
implicitly padded with empty moments. This differs from the behavior of
python's built-in zip function, which would instead truncate the longer
circuits.

The zipped circuits can't have overlapping operations occurring at the
same moment index.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`circuits`
</td>
<td>
The circuits to merge together.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The merged circuit.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
The zipped circuits have overlapping operations occurring at the
same moment index.
</td>
</tr>
</table>



#### Examples:

>>> import cirq
>>> a, b, c, d = cirq.LineQubit.range(4)
>>> circuit1 = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b))
>>> circuit2 = cirq.Circuit(cirq.X(c), cirq.Y(c), cirq.Z(c))
>>> circuit3 = cirq.Circuit(cirq.Moment(), cirq.Moment(cirq.S(d)))
>>> print(circuit1.zip(circuit2))

* <b>`0`</b>: ───H───@───────
          │
* <b>`1`</b>: ───────X───────
<BLANKLINE>
* <b>`2`</b>: ───X───Y───Z───
>>> print(circuit1.zip(circuit2, circuit3))
* <b>`0`</b>: ───H───@───────
          │
* <b>`1`</b>: ───────X───────
<BLANKLINE>
* <b>`2`</b>: ───X───Y───Z───
<BLANKLINE>
* <b>`3`</b>: ───────S───────
>>> print(cirq.Circuit.zip(circuit3, circuit2, circuit1))
* <b>`0`</b>: ───H───@───────
          │
* <b>`1`</b>: ───────X───────
<BLANKLINE>
* <b>`2`</b>: ───X───Y───Z───
<BLANKLINE>
* <b>`3`</b>: ───────S───────


<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other
)
</code></pre>




<h3 id="__bool__"><code>__bool__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__()
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    key
)
</code></pre>




<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__() -> Iterator['cirq.Moment']
</code></pre>




<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__() -> int
</code></pre>




<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    repetitions: int
)
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
) -> bool
</code></pre>

Return self!=value.


<h3 id="__pow__"><code>__pow__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    exponent: int
) -> "Circuit"
</code></pre>

A circuit raised to a power, only valid for exponent -1, the inverse.

This will fail if anything other than -1 is passed to the Circuit by
returning NotImplemented.  Otherwise this will return the inverse
circuit, which is the circuit with its moment order reversed and for
every moment all the moment's operations are replaced by its inverse.
If any of the operations do not support inverse, NotImplemented will be
returned.

<h3 id="__radd__"><code>__radd__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__radd__(
    other
)
</code></pre>




<h3 id="__rmul__"><code>__rmul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    repetitions: int
)
</code></pre>






