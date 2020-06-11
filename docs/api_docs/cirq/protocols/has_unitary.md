<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.has_unitary" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.has_unitary

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/has_unitary_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Determines whether the value has a unitary effect.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.has_unitary`, `cirq.protocols.has_unitary_protocol.has_unitary`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.has_unitary(
    val: Any,
    *,
    allow_decompose: bool = True
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

Determines whether `val` has a unitary effect by attempting the following
strategies:

1. Try to use `val.has_unitary()`.
    Case a) Method not present or returns `NotImplemented`.
        Inconclusive.
    Case b) Method returns `True`.
        Unitary.
    Case c) Method returns `False`.
        Not unitary.

2. Try to use `val._decompose_()`.
    Case a) Method not present or returns `NotImplemented` or `None`.
        Inconclusive.
    Case b) Method returns an OP_TREE containing only unitary operations.
        Unitary.
    Case c) Method returns an OP_TREE containing non-unitary operations.
        Not Unitary.

3. Try to use `val._apply_unitary_(args)`.
    Case a) Method not present or returns `NotImplemented`.
        Inconclusive.
    Case b) Method returns a numpy array.
        Unitary.
    Case c) Method returns `None`.
        Not unitary.

4. Try to use `val._unitary_()`.
    Case a) Method not present or returns `NotImplemented`.
        Continue to next strategy.
    Case b) Method returns a numpy array.
        Unitary.
    Case c) Method returns `None`.
        Not unitary.

It is assumed that, when multiple of these strategies give a conclusive
result, that these results will all be consistent with each other. If all
strategies are inconclusive, the value is classified as non-unitary.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr class="alt">
<td colspan="2">
The value that may or may not have a unitary effect.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Whether or not `val` has a unitary effect.
</td>
</tr>

</table>

