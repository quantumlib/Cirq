<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.has_stabilizer_effect" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.has_stabilizer_effect

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/has_stabilizer_effect_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns whether the input has a stabilizer effect.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.has_stabilizer_effect`, `cirq.protocols.has_stabilizer_effect_protocol.has_stabilizer_effect`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.has_stabilizer_effect(
    val: Any
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->
For 1-qubit gates always returns correct result. For other operations relies
on the operation to define whether it has stabilizer effect.