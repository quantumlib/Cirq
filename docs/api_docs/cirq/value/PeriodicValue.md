<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.PeriodicValue" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# cirq.value.PeriodicValue

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/periodic_value.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Wrapper for periodic numerical values.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.PeriodicValue`, `cirq.value.periodic_value.PeriodicValue`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.PeriodicValue(
    value: Union[int, float],
    period: Union[int, float]
)
</code></pre>



<!-- Placeholder for "Used in" -->

Wrapper for periodic numerical types which implements `__eq__`, `__ne__`,
`__hash__` and `_approx_eq_` so that values which are in the same
equivalence class are treated as equal.

Internally the `value` passed to `__init__` is normalized to the interval
[0, `period`) and stored as that. Specialized version of `_approx_eq_` is
provided to cover values which end up at the opposite edges of this
interval.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`
</td>
<td>
numerical value to wrap.
</td>
</tr><tr>
<td>
`period`
</td>
<td>
periodicity of the numerical value.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/periodic_value.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: Any
) -> bool
</code></pre>

Return self==value.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/periodic_value.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: Any
) -> bool
</code></pre>

Return self!=value.




