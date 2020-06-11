<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.sample_sweep" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.sim.sample_sweep

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/mux.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Runs the supplied Circuit, mimicking quantum hardware.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.sample_sweep`, `cirq.sim.mux.sample_sweep`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.sample_sweep(
    program: "cirq.Circuit",
    params: <a href="../../cirq/study/Sweepable.md"><code>cirq.study.Sweepable</code></a>,
    *,
    noise: "cirq.NOISE_MODEL_LIKE" = None,
    repetitions: int = 1,
    dtype: Type[np.number] = np.complex64,
    seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None
) -> List[<a href="../../cirq/study/TrialResult.md"><code>cirq.study.TrialResult</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

In contrast to run, this allows for sweeping over different parameter
values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit to simulate.
</td>
</tr><tr>
<td>
`params`
</td>
<td>
Parameters to run with the program.
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
`repetitions`
</td>
<td>
The number of repetitions to simulate, per set of
parameter values.
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



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
TrialResult list for this run; one for each possible parameter
resolver.
</td>
</tr>

</table>

