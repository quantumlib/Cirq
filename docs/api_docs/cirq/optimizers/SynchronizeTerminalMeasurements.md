<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.SynchronizeTerminalMeasurements" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="optimize_circuit"/>
</div>

# cirq.optimizers.SynchronizeTerminalMeasurements

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/synchronize_terminal_measurements.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Move measurements to the end of the circuit.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.SynchronizeTerminalMeasurements`, `cirq.optimizers.synchronize_terminal_measurements.SynchronizeTerminalMeasurements`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.SynchronizeTerminalMeasurements(
    after_other_operations: bool = True
)
</code></pre>



<!-- Placeholder for "Used in" -->

Move all measurements in a circuit to the final moment if it can accomodate
them (without overlapping with other operations). If
self._after_other_operations is true then a new moment will be added to the
end of the circuit containing all the measurements that should be brought
forward.

## Methods

<h3 id="optimize_circuit"><code>optimize_circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/synchronize_terminal_measurements.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>optimize_circuit(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
) -> None
</code></pre>




<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/synchronize_terminal_measurements.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
)
</code></pre>

Call self as a function.




