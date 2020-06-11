<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.ActOnStateVectorArgs" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="record_measurement_result"/>
<meta itemprop="property" content="subspace_index"/>
<meta itemprop="property" content="swap_target_tensor_for"/>
</div>

# cirq.sim.ActOnStateVectorArgs

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/act_on_state_vector_args.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



State and context for an operation acting on a state vector.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ActOnStateVectorArgs`, `cirq.sim.act_on_state_vector_args.ActOnStateVectorArgs`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.ActOnStateVectorArgs(
    target_tensor: np.ndarray,
    available_buffer: np.ndarray,
    axes: Iterable[int],
    prng: np.random.RandomState,
    log_of_measurement_results: Dict[str, Any]
)
</code></pre>



<!-- Placeholder for "Used in" -->

There are three common ways to act on this object:

1. Directly edit the `target_tensor` property, which is storing the state
    vector of the quantum system as a numpy array with one axis per qudit.
2. Overwrite the `available_buffer` property with the new state vector, and
    then pass `available_buffer` into `swap_target_tensor_for`.
3. Call `record_measurement_result(key, val)` to log a measurement result.

## Methods

<h3 id="record_measurement_result"><code>record_measurement_result</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/act_on_state_vector_args.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>record_measurement_result(
    key: str,
    value: Any
)
</code></pre>

Adds a measurement result to the log.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`key`
</td>
<td>
The key the measurement result should be logged under. Note
that operations should only store results under keys they have
declared in a `_measurement_keys_` method.
</td>
</tr><tr>
<td>
`value`
</td>
<td>
The value to log for the measurement.
</td>
</tr>
</table>



<h3 id="subspace_index"><code>subspace_index</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/act_on_state_vector_args.py">View source</a>

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

When operating on qudits instead of qubits, the same basic logic
applies but in a different basis. For example, if the target
axes have dimension [a:2, b:3, c:2] then the integer 10
decomposes into [a=0, b=2, c=1] via 7 = 1*(3*2) +  2*(2) + 0.
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

When operating on qudits instead of qubits, the same basic logic
applies but in a different basis. For example, if the target
axes have dimension [a:2, b:3, c:2] then the integer 10
decomposes into [a=1, b=2, c=0] via 7 = 1*(3*2) +  2*(2) + 0.
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


<h3 id="swap_target_tensor_for"><code>swap_target_tensor_for</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/act_on_state_vector_args.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>swap_target_tensor_for(
    new_target_tensor: np.ndarray
)
</code></pre>

Gives a new state vector for the system.

Typically, the new state vector should be `args.available_buffer` where
`args` is this <a href="../../cirq/sim/ActOnStateVectorArgs.md"><code>cirq.ActOnStateVectorArgs</code></a> instance.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`new_target_tensor`
</td>
<td>
The new system state. Must have the same shape
and dtype as the old system state.
</td>
</tr>
</table>





