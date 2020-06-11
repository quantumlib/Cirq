<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.generalized_amplitude_damp" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.generalized_amplitude_damp

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/common_channels.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a GeneralizedAmplitudeDampingChannel with the given

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.generalized_amplitude_damp`, `cirq.ops.common_channels.generalized_amplitude_damp`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.generalized_amplitude_damp(
    p: float,
    gamma: float
) -> <a href="../../cirq/ops/GeneralizedAmplitudeDampingChannel.md"><code>cirq.ops.GeneralizedAmplitudeDampingChannel</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->
probabilities gamma and p.

This channel evolves a density matrix via:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
          + M_2 \rho M_2^\dagger + M_3 \rho M_3^\dagger
    $$

#### With:


$$
\begin{aligned}
M_0 =& \sqrt{p} \begin{bmatrix}
                    1 & 0  \\
                    0 & \sqrt{1 - \gamma}
               \end{bmatrix}
\\
M_1 =& \sqrt{p} \begin{bmatrix}
                    0 & \sqrt{\gamma} \\
                    0 & 0
               \end{bmatrix}
\\
M_2 =& \sqrt{1-p} \begin{bmatrix}
                    \sqrt{1-\gamma} & 0 \\
                     0 & 1
                  \end{bmatrix}
\\
M_3 =& \sqrt{1-p} \begin{bmatrix}
                     0 & 0 \\
                     \sqrt{\gamma} & 0
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
the probability of the interaction being dissipative.
</td>
</tr><tr>
<td>
`p`
</td>
<td>
the probability of the qubit and environment exchanging energy.
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
gamma or p is not a valid probability.
</td>
</tr>
</table>

