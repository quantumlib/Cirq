<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.definitely_commutes" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.definitely_commutes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/commutes_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Determines whether two values definitely commute.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.definitely_commutes`, `cirq.protocols.commutes_protocol.definitely_commutes`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.definitely_commutes(
    v1: Any,
    v2: Any,
    *,
    atol: Union[int, float] = 1e-08
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`True`
</td>
<td>
The two values definitely commute.
</td>
</tr><tr>
<td>
`False`
</td>
<td>
The two values may or may not commute.
</td>
</tr>
</table>

