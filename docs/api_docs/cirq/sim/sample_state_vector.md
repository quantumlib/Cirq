<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.sample_state_vector" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.sim.sample_state_vector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/state_vector.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Samples repeatedly from measurements in the computational basis.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.sample_state_vector`, `cirq.sim.state_vector.sample_state_vector`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.sample_state_vector(
    state_vector: np.ndarray,
    indices: List[int],
    *,
    qid_shape: Optional[Tuple[int, ...]] = None,
    repetitions: int = 1,
    seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

Note that this does not modify the passed in state.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`state_vector`
</td>
<td>
The multi-qubit state vector to be sampled. This is an
array of 2 to the power of the number of qubit complex numbers, and
so state must be of size ``2**integer``.  The `state_vector` can be
a vector of size ``2**integer`` or a tensor of shape
``(2, 2, ..., 2)``.
</td>
</tr><tr>
<td>
`indices`
</td>
<td>
Which qubits are measured. The `state_vector` is assumed to be
supplied in big endian order. That is the xth index of v, when
expressed as a bitstring, has its largest values in the 0th index.
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
`repetitions`
</td>
<td>
The number of times to sample.
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
Measurement results with True corresponding to the ``|1⟩`` state.
The outer list is for repetitions, and the inner corresponds to
measurements ordered by the supplied qubits. These lists
are wrapped as an numpy ndarray.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
``repetitions`` is less than one or size of `state_vector`
is not a power of 2.
</td>
</tr><tr>
<td>
`IndexError`
</td>
<td>
An index from ``indices`` is out of range, given the number
of qubits corresponding to the state.
</td>
</tr>
</table>

