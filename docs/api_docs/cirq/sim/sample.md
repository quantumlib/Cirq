<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.sample" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.sim.sample

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/mux.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Simulates sampling from the given circuit.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.sample`, `cirq.sim.mux.sample`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.sample(
    program: "cirq.Circuit",
    *,
    noise: "cirq.NOISE_MODEL_LIKE" = None,
    param_resolver: Optional[<a href="../../cirq/study/ParamResolver.md"><code>cirq.study.ParamResolver</code></a>] = None,
    repetitions: int = 1,
    dtype: Type[np.number] = np.complex64,
    seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None
) -> <a href="../../cirq/study/TrialResult.md"><code>cirq.study.TrialResult</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit to sample from.
</td>
</tr><tr>
<td>
`noise`
</td>
<td>
Noise model to use while running the simulation.
</td>
</tr><tr>
<td>
`param_resolver`
</td>
<td>
Parameters to run with the program.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of samples to take.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The `numpy.dtype` used by the simulation. Typically one of
`numpy.complex64` or `numpy.complex128`.
Favors speed over precision by default, i.e. uses `numpy.complex64`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
The random seed to use for this simulator.
</td>
</tr>
</table>

