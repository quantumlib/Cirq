<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ion.ConvertToIonGates" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="convert_circuit"/>
<meta itemprop="property" content="convert_one"/>
</div>

# cirq.ion.ConvertToIonGates

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/convert_to_ion_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Attempts to convert non-native gates into IonGates.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ConvertToIonGates`, `cirq.ion.convert_to_ion_gates.ConvertToIonGates`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ion.ConvertToIonGates(
    ignore_failures: bool = False
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
    

## Methods

<h3 id="convert_circuit"><code>convert_circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/convert_to_ion_gates.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convert_circuit(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
) -> <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
</code></pre>




<h3 id="convert_one"><code>convert_one</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ion/convert_to_ion_gates.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convert_one(
    op: <a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> <a href="../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
</code></pre>

Convert a single (one- or two-qubit) operation

into ion trap native gates
Args:
    op: gate operation to be converted

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
the desired operation implemented with ion trap gates
</td>
</tr>

</table>





