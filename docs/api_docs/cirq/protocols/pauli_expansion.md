<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.pauli_expansion" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.pauli_expansion

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/pauli_expansion_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns coefficients of the expansion of val in the Pauli basis.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.pauli_expansion`, `cirq.protocols.pauli_expansion_protocol.pauli_expansion`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.pauli_expansion(
    val: Any,
    *,
    default: Union[value.LinearDict[str], TDefault] = cirq.protocols.pauli_expansion_protocol.RaiseTypeErrorIfNotProvided,
    atol: float = 1e-09
) -> Union[value.LinearDict[str], TDefault]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value whose Pauli expansion is to returned.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
Determines what happens when `val` does not have methods that
allow Pauli expansion to be obtained (see below). If set, the value
is returned in that case. Otherwise, TypeError is raised.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Ignore coefficients whose absolute value is smaller than this.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `val` has a _pauli_expansion_ method, then its result is returned.
Otherwise, if `val` has a small unitary then that unitary is expanded
in the Pauli basis and coefficients are returned. Otherwise, if default
is set to None or other value then default is returned. Otherwise,
TypeError is raised.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
TypeError if `val` has none of the methods necessary to obtain its Pauli
expansion and no default value has been provided.
</td>
</tr>

</table>

