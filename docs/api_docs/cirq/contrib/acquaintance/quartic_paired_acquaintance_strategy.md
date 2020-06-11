<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.quartic_paired_acquaintance_strategy" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.acquaintance.quartic_paired_acquaintance_strategy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/strategies/quartic_paired.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Acquaintance strategy for pairs of pairs.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.acquaintance.strategies.quartic_paired.quartic_paired_acquaintance_strategy`, `cirq.contrib.acquaintance.strategies.quartic_paired_acquaintance_strategy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.quartic_paired_acquaintance_strategy(
    qubit_pairs: Iterable[Tuple['cirq.Qid', <a href="../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]]
) -> Tuple['cirq.Circuit', Sequence['cirq.Qid']]
</code></pre>



<!-- Placeholder for "Used in" -->

Implements UpCCGSD ansatz from arXiv:1810.02327.