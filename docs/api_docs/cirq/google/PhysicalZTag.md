<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.PhysicalZTag" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
</div>

# cirq.google.PhysicalZTag

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/ops/physical_z_tag.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Class to add as a tag onto an Operation to denote a Physical Z operation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.ops.PhysicalZTag`, `cirq.google.ops.physical_z_tag.PhysicalZTag`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

By default, all Z rotations on Google devices are considered to be virtual.
When performing the Z operation, the device will update its internal phase
tracking mechanisms, essentially commuting it forwards through the circuit
until it hits a non-commuting operation (Such as a sqrt(iSwap)).

When applied to a Z rotation operation, this tag indicates to the hardware
that an actual physical operation should be done instead.  This class can
only be applied to instances of <a href="../../cirq/ops/ZPowGate.md"><code>cirq.ZPowGate</code></a>.  If applied to other gates
(such as PhasedXZGate), this class will have no effect.

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/ops/physical_z_tag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
) -> bool
</code></pre>

Return self==value.




