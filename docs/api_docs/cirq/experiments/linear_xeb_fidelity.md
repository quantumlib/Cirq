<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.linear_xeb_fidelity" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.linear_xeb_fidelity

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/fidelity_estimation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Estimates XEB fidelity from one circuit using linear estimator.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.fidelity_estimation.linear_xeb_fidelity`, `cirq.linear_xeb_fidelity`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.linear_xeb_fidelity(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    bitstrings: Sequence[int],
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT,
    amplitudes: Optional[Mapping[int, complex]] = None
) -> float
</code></pre>



<!-- Placeholder for "Used in" -->
