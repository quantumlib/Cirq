<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.InterchangeableQubitsGate" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="qubit_index_to_equivalence_group_key"/>
</div>

# cirq.ops.InterchangeableQubitsGate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/gate_features.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Indicates operations should be equal under some qubit permutations.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.InterchangeableQubitsGate`, `cirq.ops.gate_features.InterchangeableQubitsGate`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->


## Methods

<h3 id="qubit_index_to_equivalence_group_key"><code>qubit_index_to_equivalence_group_key</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/gate_features.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qubit_index_to_equivalence_group_key(
    index: int
) -> int
</code></pre>

Returns a key that differs between non-interchangeable qubits.




