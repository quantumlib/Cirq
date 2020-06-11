<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.t2_decay" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.t2_decay

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/t2_decay_experiment.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Runs a t2 transverse relaxation experiment.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.t2_decay_experiment.t2_decay`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.t2_decay(
    sampler: <a href="../../cirq/work/Sampler.md"><code>cirq.work.Sampler</code></a>,
    *,
    qubit: <a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a> = ExperimentType.RAMSEY,
    experiment_type: "ExperimentType" = cirq.experiments.t2_decay_experiment.ExperimentType.RAMSEY,
    num_points: int = 1000,
    max_delay: "cirq.DURATION_LIKE" = None,
    min_delay: "cirq.DURATION_LIKE" = None,
    repetitions: int = 1000,
    delay_sweep: Optional[<a href="../../cirq/study/Sweep.md"><code>cirq.study.Sweep</code></a>] = None
) -> "cirq.experiments.T2DecayResult"
</code></pre>



<!-- Placeholder for "Used in" -->

Initializes a qubit into a superposition state, evolves the system using
rules determined by the experiment type and by the delay parameters,
then rotates back for measurement.  This will measure the phase decoherence
decay.  This experiment has three types of T2 metrics, each which measure
a different slice of the noise spectrum.

For the Ramsey experiment type (often denoted T2*), the state will be
prepared with a square root Y gate (`cirq.Y ** 0.5`) and then waits for
a variable amount of time.  After this time, it will do basic state
tomography to measure the expectation of the Pauli-X and Pauli-Y operators
by performing either a `cirq.Y ** -0.5` or `cirq.X ** -0.5`.  The square of
these two measurements is summed to determine the length of the Bloch
vector. This experiment measures the phase decoherence of the system under
free evolution.

For the Hahn echo experiment (often denoted T2 or spin echo), the state
will also be prepared with a square root Y gate (`cirq.Y ** 0.5`).
However, during the mid-point of the delay time being measured, a pi-pulse
(<a href="../../cirq/ops/X.md"><code>cirq.X</code></a>) gate will be applied to cancel out inhomogeneous dephasing.
The same method of measuring the final state as Ramsey experiment is applied
after the second half of the delay period.

CPMG, or the Carr-Purcell-Meiboom-Gill sequence, is currently not
implemented.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sampler`
</td>
<td>
The quantum engine or simulator to run the circuits.
</td>
</tr><tr>
<td>
`qubit`
</td>
<td>
The qubit under test.
</td>
</tr><tr>
<td>
`experiment_type`
</td>
<td>
The type of T2 test to run.
</td>
</tr><tr>
<td>
`num_points`
</td>
<td>
The number of evenly spaced delays to test.
</td>
</tr><tr>
<td>
`max_delay`
</td>
<td>
The largest delay to test.
</td>
</tr><tr>
<td>
`min_delay`
</td>
<td>
The smallest delay to test. Defaults to no delay.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of repetitions of the circuit
for each delay and for each tomography result.
</td>
</tr><tr>
<td>
`delay_sweep`
</td>
<td>
Optional range of time delays to sweep across.  This should
be a SingleSweep using the 'delay_ns' with values in integer number
of nanoseconds.  If specified, this will override the max_delay and
min_delay parameters.  If not specified, the experiment will sweep
from min_delay to max_delay with linear steps.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A T2DecayResult object that stores and can plot the data.
</td>
</tr>

</table>

