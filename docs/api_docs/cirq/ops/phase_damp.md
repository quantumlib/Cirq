<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.phase_damp" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.phase_damp

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/common_channels.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Creates a PhaseDampingChannel with damping constant gamma.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ops.common_channels.phase_damp`, `cirq.phase_damp`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.phase_damp(
    gamma: float
) -> <a href="../../cirq/ops/PhaseDampingChannel.md"><code>cirq.ops.PhaseDampingChannel</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This channel evolves a density matrix via:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

#### With:


$$
\begin{aligned}
M_0 =& \begin{bmatrix}
        1 & 0  \\
        0 & \sqrt{1 - \gamma}
      \end{bmatrix}
\\
M_1 =& \begin{bmatrix}
        0 & 0 \\
        0 & \sqrt{\gamma}
      \end{bmatrix}
\end{aligned}
$$



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`gamma`
</td>
<td>
The damping constant.
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
is gamma is not a valid probability.
</td>
</tr>
</table>

