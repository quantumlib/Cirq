<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.asymmetric_depolarize" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.asymmetric_depolarize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/common_channels.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a AsymmetricDepolarizingChannel with given parameter.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.asymmetric_depolarize`, `cirq.ops.common_channels.asymmetric_depolarize`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.asymmetric_depolarize(
    p_x: float,
    p_y: float,
    p_z: float
) -> <a href="../../cirq/ops/AsymmetricDepolarizingChannel.md"><code>cirq.ops.AsymmetricDepolarizingChannel</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This channel evolves a density matrix via

    $$
    \rho \rightarrow (1 - p_x - p_y - p_z) \rho
            + p_x X \rho X + p_y Y \rho Y + p_z Z \rho Z
    $$

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`p_x`
</td>
<td>
The probability that a Pauli X and no other gate occurs.
</td>
</tr><tr>
<td>
`p_y`
</td>
<td>
The probability that a Pauli Y and no other gate occurs.
</td>
</tr><tr>
<td>
`p_z`
</td>
<td>
The probability that a Pauli Z and no other gate occurs.
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
if the args or the sum of the args are not probabilities.
</td>
</tr>
</table>

