<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.quirk.circuit_to_quirk_url" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.quirk.circuit_to_quirk_url

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/quirk/export_to_quirk.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a Quirk URL for the given circuit.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.quirk.export_to_quirk.circuit_to_quirk_url`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.quirk.circuit_to_quirk_url(
    circuit: <a href="../../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    prefer_unknown_gate_to_failure: bool = False,
    escape_url=True
) -> str
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The circuit to open in Quirk.
</td>
</tr><tr>
<td>
`prefer_unknown_gate_to_failure`
</td>
<td>
If not set, gates that fail to convert
will cause this function to raise an error. If set, a URL
containing bad gates will be generated. (Quirk will open the URL,
and replace the bad gates with parse errors, but still get the rest
of the circuit.)
</td>
</tr><tr>
<td>
`escape_url`
</td>
<td>
If set, the generated URL will have special characters such
as quotes escaped using %. This makes it possible to paste the URL
into forums and the command line and etc and have it properly
parse. If not set, the generated URL will be more compact and human
readable (and can still be pasted directly into a browser's address
bar).
</td>
</tr>
</table>


Returns: