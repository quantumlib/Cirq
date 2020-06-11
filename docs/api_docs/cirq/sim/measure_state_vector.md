<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.measure_state_vector" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.sim.measure_state_vector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/state_vector.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Performs a measurement of the state in the computational basis.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.measure_state_vector`, `cirq.sim.state_vector.measure_state_vector`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.measure_state_vector(
    state_vector: np.ndarray,
    indices: Sequence[int],
    *,
    qid_shape: Optional[Tuple[int, ...]] = None,
    out: np.ndarray = None,
    seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None
) -> Tuple[List[int], np.ndarray]
</code></pre>



<!-- Placeholder for "Used in" -->

This does not modify `state` unless the optional `out` is `state`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`state_vector`
</td>
<td>
The state to be measured. This state vector is assumed to
be normalized. The state vector must be of size 2 ** integer.  The
state vector can be of shape (2 ** integer) or (2, 2, ..., 2).
</td>
</tr><tr>
<td>
`indices`
</td>
<td>
Which qubits are measured. The `state_vector` is assumed to be
supplied in big endian order. That is the xth index of v, when
expressed as a bitstring, has the largest values in the 0th index.
</td>
</tr><tr>
<td>
`qid_shape`
</td>
<td>
The qid shape of the `state_vector`.  Specify this argument
when using qudits.
</td>
</tr><tr>
<td>
`out`
</td>
<td>
An optional place to store the result. If `out` is the same as
the `state_vector` parameter, then `state_vector` will be modified
inline. If `out` is not None, then the result is put into `out`.
If `out` is None a new value will be allocated. In all of these
case out will be the same as the returned ndarray of the method.
The shape and dtype of `out` will match that of `state_vector` if
`out` is None, otherwise it will match the shape and dtype of `out`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A seed for the pseudorandom number generator.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple of a list and an numpy array. The list is an array of booleans
corresponding to the measurement values (ordered by the indices). The
numpy array is the post measurement state vector. This state vector has
the same shape and dtype as the input `state_vector`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
ValueError if the size of state is not a power of 2.
IndexError if the indices are out of range for the number of qubits
corresponding to the state.
</td>
</tr>

</table>

