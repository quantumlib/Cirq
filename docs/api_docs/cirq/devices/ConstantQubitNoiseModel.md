<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.devices.ConstantQubitNoiseModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="from_noise_model_like"/>
<meta itemprop="property" content="is_virtual_moment"/>
<meta itemprop="property" content="noisy_moment"/>
<meta itemprop="property" content="noisy_moments"/>
<meta itemprop="property" content="noisy_operation"/>
</div>

# cirq.devices.ConstantQubitNoiseModel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/noise_model.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Applies noise to each qubit individually at the start of every moment.

Inherits From: [`NoiseModel`](../../cirq/devices/NoiseModel.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ConstantQubitNoiseModel`, `cirq.devices.noise_model.ConstantQubitNoiseModel`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.devices.ConstantQubitNoiseModel(
    qubit_noise_gate: "cirq.Gate"
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is the noise model that is wrapped around an operation when that
operation is given as "the noise to use" for a `NOISE_MODEL_LIKE` parameter.

## Methods

<h3 id="from_noise_model_like"><code>from_noise_model_like</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/noise_model.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_noise_model_like(
    noise: "cirq.NOISE_MODEL_LIKE"
) -> "cirq.NoiseModel"
</code></pre>

Transforms an object into a noise model if umambiguously possible.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`noise`
</td>
<td>
``None``, a `<a href="../../cirq/devices/NoiseModel.md"><code>cirq.NoiseModel</code></a>`, or a single qubit operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`<a href="../../cirq.md#NO_NOISE"><code>cirq.NO_NOISE</code></a>` when given ``None``,
`<a href="../../cirq/devices/ConstantQubitNoiseModel.md"><code>cirq.ConstantQubitNoiseModel(gate)</code></a>` when given a single qubit
gate, or the given value if it is already a `<a href="../../cirq/devices/NoiseModel.md"><code>cirq.NoiseModel</code></a>`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
The input is not a ``cirq.NOISE_MODE_LIKE``.
</td>
</tr>
</table>



<h3 id="is_virtual_moment"><code>is_virtual_moment</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/noise_model.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_virtual_moment(
    moment: "cirq.Moment"
) -> bool
</code></pre>

Returns true iff the given moment is non-empty and all of its
operations are virtual.

Moments for which this method returns True should not have additional
noise applied to them.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`moment`
</td>
<td>
`<a href="../../cirq/ops/Moment.md"><code>cirq.Moment</code></a>` to check for non-virtual operations.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
True if "moment" is non-empty and all operations in "moment" are
virtual; false otherwise.
</td>
</tr>

</table>



<h3 id="noisy_moment"><code>noisy_moment</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/noise_model.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>noisy_moment(
    moment: "cirq.Moment",
    system_qubits: Sequence['cirq.Qid']
)
</code></pre>

Adds noise to the operations from a moment.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`moment`
</td>
<td>
The moment to add noise to.
</td>
</tr><tr>
<td>
`system_qubits`
</td>
<td>
A list of all qubits in the system.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An OP_TREE corresponding to the noisy operations for the moment.
</td>
</tr>

</table>



<h3 id="noisy_moments"><code>noisy_moments</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/noise_model.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>noisy_moments(
    moments: "Iterable[cirq.Moment]",
    system_qubits: Sequence['cirq.Qid']
) -> Sequence['cirq.OP_TREE']
</code></pre>

Adds possibly stateful noise to a series of moments.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`moments`
</td>
<td>
The moments to add noise to.
</td>
</tr><tr>
<td>
`system_qubits`
</td>
<td>
A list of all qubits in the system.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A sequence of OP_TREEs, with the k'th tree corresponding to the
noisy operations for the k'th moment.
</td>
</tr>

</table>



<h3 id="noisy_operation"><code>noisy_operation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/devices/noise_model.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>noisy_operation(
    operation: "cirq.Operation"
) -> "cirq.OP_TREE"
</code></pre>

Adds noise to an individual operation.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`operation`
</td>
<td>
The operation to make noisy.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An OP_TREE corresponding to the noisy operations implementing the
noisy version of the given operation.
</td>
</tr>

</table>



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: _SupportsValueEquality
) -> bool
</code></pre>






