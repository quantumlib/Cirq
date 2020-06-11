<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.linear_xeb_fidelity_from_probabilities" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.linear_xeb_fidelity_from_probabilities

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/fidelity_estimation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Linear XEB fidelity estimator.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.fidelity_estimation.linear_xeb_fidelity_from_probabilities`, `cirq.linear_xeb_fidelity_from_probabilities`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.linear_xeb_fidelity_from_probabilities(
    hilbert_space_dimension: int,
    probabilities: Sequence[float]
) -> float
</code></pre>



<!-- Placeholder for "Used in" -->

Estimates fidelity from ideal probabilities of observed bitstrings.

This estimator makes two assumptions. First, it assumes that the circuit
used in experiment is sufficiently scrambling that its output probabilities
follow the Porter-Thomas distribution. This assumption holds for typical
instances of random quantum circuits of sufficient depth. Second, it assumes
that the circuit uses enough qubits so that the Porter-Thomas distribution
can be approximated with the exponential distribution.

In practice the validity of these assumptions can be confirmed by plotting
a histogram of output probabilities and comparing it to the exponential
distribution.

The mean of this estimator is the true fidelity f and the variance is

    (1 + 2f - f^2) / M

where f is the fidelity and M the number of observations, equal to
len(probabilities). This is better than logarithmic XEB (see below)
when fidelity is f < 0.32. Since this estimator is unbiased, the
variance is equal to the mean squared error of the estimator.

The estimator is intended for use with xeb_fidelity() below.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`hilbert_space_dimension`
</td>
<td>
Dimension of the Hilbert space on which
the channel whose fidelity is being estimated is defined.
</td>
</tr><tr>
<td>
`probabilities`
</td>
<td>
Ideal probabilities of bitstrings observed in
experiment.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Estimate of fidelity associated with an experimental realization
of a quantum circuit.
</td>
</tr>

</table>

