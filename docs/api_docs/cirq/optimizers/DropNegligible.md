<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.DropNegligible" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="optimize_circuit"/>
</div>

# cirq.optimizers.DropNegligible

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/drop_negligible.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An optimization pass that removes operations with tiny effects.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.DropNegligible`, `cirq.optimizers.drop_negligible.DropNegligible`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.DropNegligible(
    tolerance: float = 1e-08
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="optimize_circuit"><code>optimize_circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/drop_negligible.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>optimize_circuit(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
) -> None
</code></pre>




<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/drop_negligible.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
)
</code></pre>

Call self as a function.




