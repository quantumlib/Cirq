<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.qis.fidelity" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.qis.fidelity

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/qis/measures.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Fidelity of two quantum states.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.fidelity`, `cirq.qis.measures.fidelity`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.qis.fidelity(
    state1: np.ndarray,
    state2: np.ndarray
) -> float
</code></pre>



<!-- Placeholder for "Used in" -->

The fidelity of two density matrices ρ and σ is defined as

    trace(sqrt(sqrt(ρ) σ sqrt(ρ)))^2.

The given states can be state vectors or density matrices.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`state1`
</td>
<td>
The first state.
</td>
</tr><tr>
<td>
`state2`
</td>
<td>
The second state.
</td>
</tr>
</table>

