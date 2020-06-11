<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.unitary" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.unitary

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/unitary_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a unitary matrix describing the given value.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.protocols.unitary_protocol.unitary`, `cirq.unitary`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.unitary(
    val: Any,
    default: <a href="../../cirq/protocols/unitary_protocol/TDefault.md"><code>cirq.protocols.unitary_protocol.TDefault</code></a> = cirq.protocols.unitary_protocol.RaiseTypeErrorIfNotProvided
) -> Union[np.ndarray, <a href="../../cirq/protocols/unitary_protocol/TDefault.md"><code>cirq.protocols.unitary_protocol.TDefault</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

The matrix is determined by any one of the following techniques:

- The value has a `_unitary_` method that returns something besides None or
    NotImplemented. The matrix is whatever the method returned.
- The value has a `_decompose_` method that returns a list of operations,
    and each operation in the list has a unitary effect. The matrix is
    created by aggregating the sub-operations' unitary effects.
- The value has an `_apply_unitary_` method, and it returns something
    besides None or NotImplemented. The matrix is created by applying
    `_apply_unitary_` to an identity matrix.

If none of these techniques succeeds, it is assumed that `val` doesn't have
a unitary effect. The order in which techniques are attempted is
unspecified.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value to describe with a unitary matrix.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
Determines the fallback behavior when `val` doesn't have
a unitary effect. If `default` is not set, a TypeError is raised. If
`default` is set to a value, that value is returned.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `val` has a unitary effect, the corresponding unitary matrix.
Otherwise, if `default` is specified, it is returned.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
`val` doesn't have a unitary effect and no default value was
specified.
</td>
</tr>
</table>

