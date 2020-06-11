<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.compute_cphase_exponents_for_fsim_decomposition" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.compute_cphase_exponents_for_fsim_decomposition

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/cphase_to_fsim.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns intervals of CZPowGate exponents valid for FSim decomposition.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.compute_cphase_exponents_for_fsim_decomposition`, `cirq.optimizers.cphase_to_fsim.compute_cphase_exponents_for_fsim_decomposition`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.compute_cphase_exponents_for_fsim_decomposition(
    fsim_gate: "cirq.FSimGate"
) -> Sequence[Tuple[float, float]]
</code></pre>



<!-- Placeholder for "Used in" -->

Ideal intervals associated with the constraints are closed, but due to
numerical error the caller should not assume the endpoints themselves
are valid for the decomposition. See `decompose_cphase_into_two_fsim`
for details on how FSimGate parameters constrain the phase angle of
CZPowGate.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`fsim_gate`
</td>
<td>
FSimGate into which CZPowGate would be decomposed.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Sequence of 2-tuples each consisting of the minimum and maximum
value of the exponent for which CZPowGate can be decomposed into
two FSimGates. The intervals are cropped to [0, 2]. The function
returns zero, one or two intervals.
</td>
</tr>

</table>

