<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.two_qubit_gates.math_utils.random_qubit_unitary" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.two_qubit_gates.math_utils.random_qubit_unitary

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/two_qubit_gates/math_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Random qubit unitary distributed over the Haar measure.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.two_qubit_gates.math_utils.random_qubit_unitary(
    shape: Sequence[int] = (),
    randomize_global_phase: bool = False,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

The implementation is vectorized for speed.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`
</td>
<td>
The broadcasted shape of the output. This is used to generate
a tensor of random unitaries with dimensions tuple(shape) + (2,2).
</td>
</tr><tr>
<td>
`randomize_global_phase`
</td>
<td>
(Default False) If True, a global phase is also
sampled randomly. This corresponds to sampling over U(2) instead of
SU(2).
</td>
</tr><tr>
<td>
`rng`
</td>
<td>
Random number generator to be used in sampling. Default is
numpy.random.
</td>
</tr>
</table>

