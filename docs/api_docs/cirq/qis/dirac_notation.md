<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.qis.dirac_notation" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.qis.dirac_notation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/qis/states.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the state vector as a string in Dirac notation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.dirac_notation`, `cirq.qis.states.dirac_notation`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.qis.dirac_notation(
    state_vector: Sequence,
    decimals: int = 2,
    qid_shape: Optional[Tuple[int, ...]] = None
) -> str
</code></pre>



<!-- Placeholder for "Used in" -->


#### For example:


state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2)],
                        dtype=np.complex64)
print(dirac_notation(state_vector)) -> 0.71|0⟩ + 0.71|1⟩



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`state_vector`
</td>
<td>
A sequence representing a state vector in which
the ordering mapping to qubits follows the standard Kronecker
convention of numpy.kron (big-endian).
</td>
</tr><tr>
<td>
`decimals`
</td>
<td>
How many decimals to include in the pretty print.
</td>
</tr><tr>
<td>
`qid_shape`
</td>
<td>
specifies the dimensions of the qudits for the input
`state_vector`.  If not specified, qubits are assumed and the
`state_vector` must have a dimension a power of two.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A pretty string consisting of a sum of computational basis kets
and non-zero floats of the specified accuracy.
</td>
</tr>

</table>

