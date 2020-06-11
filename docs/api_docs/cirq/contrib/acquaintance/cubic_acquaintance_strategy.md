<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.cubic_acquaintance_strategy" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.acquaintance.cubic_acquaintance_strategy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/strategies/cubic.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Acquaints every triple of qubits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.acquaintance.strategies.cubic.cubic_acquaintance_strategy`, `cirq.contrib.acquaintance.strategies.cubic_acquaintance_strategy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.cubic_acquaintance_strategy(
    qubits: Iterable['cirq.Qid'],
    swap_gate: "cirq.Gate" = cirq.ops.SWAP
) -> "cirq.Circuit"
</code></pre>



<!-- Placeholder for "Used in" -->

Exploits the fact that in a simple linear swap network every pair of
logical qubits that starts at distance two remains so (except temporarily
near the edge), and that every third one `goes through` the pair at some
point in the network. The strategy then iterates through a series of
mappings in which qubits i and i + k are placed at distance two, for k = 1
through n / 2. Linear swap networks are used in between to effect the
permutation.