<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.depolarize" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.depolarize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/common_channels.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a DepolarizingChannel with given probability of error.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.depolarize`, `cirq.ops.common_channels.depolarize`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.depolarize(
    p: float
) -> <a href="../../cirq/ops/DepolarizingChannel.md"><code>cirq.ops.DepolarizingChannel</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This channel applies one of four disjoint possibilities: nothing (the
identity channel) or one of the three pauli gates. The disjoint
probabilities of the three gates are all the same, p / 3, and the
identity is done with probability 1 - p. The supplied probability
must be a valid probability or else this constructor will raise a
ValueError.

This channel evolves a density matrix via

    $$
    \rho \rightarrow (1 - p) \rho
            + (p / 3) X \rho X + (p / 3) Y \rho Y + (p / 3) Z \rho Z
    $$

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`p`
</td>
<td>
The probability that one of the Pauli gates is applied. Each of
the Pauli gates is applied independently with probability p / 3.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if p is not a valid probability.
</td>
</tr>
</table>

