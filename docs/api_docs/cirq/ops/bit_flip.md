<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.bit_flip" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.bit_flip

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/common_channels.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Construct a BitFlipChannel that flips a qubit state

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.bit_flip`, `cirq.ops.common_channels.bit_flip`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.bit_flip(
    p: Optional[float] = None
) -> Union[<a href="../../cirq/ops/XPowGate.md"><code>cirq.ops.XPowGate</code></a>, <a href="../../cirq/ops/BitFlipChannel.md"><code>cirq.ops.BitFlipChannel</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->
with probability of a flip given by p. If p is None, return
a guaranteed flip in the form of an X operation.

This channel evolves a density matrix via

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

With

    $$
    \begin{aligned}
    M_0 =& \sqrt{p} \begin{bmatrix}
                        1 & 0 \\
                        0 & 1
                   \end{bmatrix}
    \\
    M_1 =& \sqrt{1-p} \begin{bmatrix}
                        0 & 1 \\
                        1 & -0
                     \end{bmatrix}
    \end{aligned}
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
the probability of a bit flip.
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

