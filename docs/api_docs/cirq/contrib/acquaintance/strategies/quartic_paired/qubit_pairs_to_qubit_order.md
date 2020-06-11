<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.strategies.quartic_paired.qubit_pairs_to_qubit_order" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.acquaintance.strategies.quartic_paired.qubit_pairs_to_qubit_order

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/strategies/quartic_paired.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Takes a sequence of qubit pairs and returns a sequence in which every

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.strategies.quartic_paired.qubit_pairs_to_qubit_order(
    qubit_pairs: Sequence[Sequence['cirq.Qid']]
) -> List['cirq.Qid']
</code></pre>



<!-- Placeholder for "Used in" -->
pair is at distance two.

Specifically, given pairs (1a, 1b), (2a, 2b), etc. returns
(1a, 2a, 1b, 2b, 3a, 4a, 3b, 4b, ...).