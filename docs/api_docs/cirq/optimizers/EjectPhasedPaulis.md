<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.EjectPhasedPaulis" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="optimize_circuit"/>
</div>

# cirq.optimizers.EjectPhasedPaulis

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/eject_phased_paulis.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Pushes X, Y, and PhasedX gates towards the end of the circuit.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.EjectPhasedPaulis`, `cirq.optimizers.eject_phased_paulis.EjectPhasedPaulis`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.EjectPhasedPaulis(
    tolerance: float = 1e-08,
    eject_parameterized: bool = False
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

As the gates get pushed, they may absorb Z gates, cancel against other
X, Y, or PhasedX gates with exponent=1, get merged into measurements (as
output bit flips), and cause phase kickback operations across CZs (which can
then be removed by the EjectZ optimization).

## Methods

<h3 id="optimize_circuit"><code>optimize_circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/eject_phased_paulis.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>optimize_circuit(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
)
</code></pre>






