<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.circuits.QuilOutput" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="rename_defgates"/>
<meta itemprop="property" content="save_to_file"/>
</div>

# cirq.circuits.QuilOutput

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/quil_output.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An object for passing operations and qubits then outputting them to

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.QuilOutput`, `cirq.circuits.quil_output.QuilOutput`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.circuits.QuilOutput(
    operations: "cirq.OP_TREE",
    qubits: Tuple['cirq.Qid', ...]
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
QUIL format. The string representation returns the QUIL output for the
circuit.

## Methods

<h3 id="rename_defgates"><code>rename_defgates</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/quil_output.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>rename_defgates(
    output: str
) -> str
</code></pre>

A function for renaming the DEFGATEs within the QUIL output. This
utilizes a second pass to find each DEFGATE and rename it based on
a counter.

<h3 id="save_to_file"><code>save_to_file</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/quil_output.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_to_file(
    path: Union[str, bytes, int]
) -> None
</code></pre>

Write QUIL output to a file specified by path.




