<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.ApplyUnitaryArgs" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="default"/>
<meta itemprop="property" content="subspace_index"/>
<meta itemprop="property" content="with_axes_transposed_to_start"/>
</div>

# cirq.protocols.ApplyUnitaryArgs

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/apply_unitary_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Arguments for performing an efficient left-multiplication by a unitary.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ApplyUnitaryArgs`, `cirq.protocols.apply_unitary_protocol.ApplyUnitaryArgs`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.ApplyUnitaryArgs(
    target_tensor: np.ndarray,
    available_buffer: np.ndarray,
    axes: Iterable[int]
)
</code></pre>



<!-- Placeholder for "Used in" -->

The receiving object is expected to mutate `target_tensor` so that it
contains the state after multiplication, and then return `target_tensor`.
Alternatively, if workspace is required, the receiving object can overwrite
`available_buffer` with the results and return `available_buffer`. Or, if
the receiving object is attempting to be simple instead of fast, it can
create an entirely new array and return that.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`target_tensor`
</td>
<td>
The input tensor that needs to be left-multiplied by
the unitary effect of the receiving object. The tensor will
have the shape (2, 2, 2, ..., 2). It usually corresponds to
a multi-qubit superposition, but it could also be a multi-qubit
unitary transformation or some other concept.
</td>
</tr><tr>
<td>
`available_buffer`
</td>
<td>
Pre-allocated workspace with the same shape and
dtype as the target tensor.
</td>
</tr><tr>
<td>
`axes`
</td>
<td>
Which axes the unitary effect is being applied to (e.g. the
qubits that the gate is operating on).
</td>
</tr>
</table>



## Methods

<h3 id="default"><code>default</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/apply_unitary_protocol.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>default(
    num_qubits: Optional[int] = None,
    *,
    qid_shape: Optional[Tuple[int, ...]] = None
) -> "ApplyUnitaryArgs"
</code></pre>

A default instance starting in state |0⟩.

Specify exactly one argument.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`num_qubits`
</td>
<td>
The number of qubits to make space for in the state.
</td>
</tr><tr>
<td>
`qid_shape`
</td>
<td>
The shape of the state, specifying the dimension of each
qid.
</td>
</tr>
</table>



<h3 id="subspace_index"><code>subspace_index</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/apply_unitary_protocol.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>subspace_index(
    little_endian_bits_int: int = 0,
    *,
    big_endian_bits_int: int = 0
) -> Tuple[Union[slice, int, 'ellipsis'], ...]
</code></pre>

An index for the subspace where the target axes equal a value.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`little_endian_bits_int`
</td>
<td>
The desired value of the qubits at the
targeted `axes`, packed into an integer. The least significant
bit of the integer is the desired bit for the first axis, and
so forth in increasing order. Can't be specified at the same
time as `big_endian_bits_int`.
</td>
</tr><tr>
<td>
`big_endian_bits_int`
</td>
<td>
The desired value of the qubits at the
targeted `axes`, packed into an integer. The most significant
bit of the integer is the desired bit for the first axis, and
so forth in decreasing order. Can't be specified at the same
time as `little_endian_bits_int`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A value that can be used to index into `target_tensor` and
`available_buffer`, and manipulate only the part of Hilbert space
corresponding to a given bit assignment.
</td>
</tr>

</table>



#### Example:

If `target_tensor` is a 4 qubit tensor and `axes` is `[1, 3]` and
then this method will return the following when given
`little_endian_bits=0b01`:

    `(slice(None), 0, slice(None), 1, Ellipsis)`

Therefore the following two lines would be equivalent:

    args.target_tensor[args.subspace_index(0b01)] += 1

    args.target_tensor[:, 0, :, 1] += 1


<h3 id="with_axes_transposed_to_start"><code>with_axes_transposed_to_start</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/apply_unitary_protocol.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_axes_transposed_to_start() -> "ApplyUnitaryArgs"
</code></pre>

Returns a transposed view of the same arguments.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A view over the same target tensor and available workspace, but
with the numpy arrays transposed such that the axes field is
guaranteed to equal `range(len(result.axes))`. This allows one to
say e.g. `result.target_tensor[0, 1, 0, ...]` instead of
`result.target_tensor[result.subspace_index(0b010)]`.
</td>
</tr>

</table>





