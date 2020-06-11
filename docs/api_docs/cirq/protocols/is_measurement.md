<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.is_measurement" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.is_measurement

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/measurement_key_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Determines whether or not the given value is a measurement.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.is_measurement`, `cirq.protocols.measurement_key_protocol.is_measurement`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.is_measurement(
    val: Any
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

Measurements are identified by the fact that <a href="../../cirq/protocols/measurement_keys.md"><code>cirq.measurement_keys</code></a> returns
a non-empty result for them.