<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.SingleQubitReadoutCalibrationResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
</div>

# cirq.experiments.SingleQubitReadoutCalibrationResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/single_qubit_readout_calibration.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Result of estimating single qubit readout error.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.single_qubit_readout_calibration.SingleQubitReadoutCalibrationResult`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.SingleQubitReadoutCalibrationResult(
    zero_state_errors, one_state_errors, repetitions, timestamp
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`zero_state_errors`
</td>
<td>
A dictionary from qubit to probability of measuring
a 1 when the qubit is initialized to |0⟩.
</td>
</tr><tr>
<td>
`one_state_errors`
</td>
<td>
A dictionary from qubit to probability of measuring
a 0 when the qubit is initialized to |1⟩.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of repetitions that were used to estimate the
probabilities.
</td>
</tr><tr>
<td>
`timestamp`
</td>
<td>
The time the data was taken, in seconds since the epoch.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>






