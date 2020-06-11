<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.Timestamp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__radd__"/>
<meta itemprop="property" content="__sub__"/>
<meta itemprop="property" content="raw_picos"/>
</div>

# cirq.value.Timestamp

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/timestamp.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A location in time with picosecond accuracy.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.Timestamp`, `cirq.value.timestamp.Timestamp`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.Timestamp(
    *,
    picos: Union[int, float] = 0,
    nanos: Union[int, float] = 0
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Supports affine operations against Duration.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`picos`
</td>
<td>
How many picoseconds away from time zero?
</td>
</tr><tr>
<td>
`nanos`
</td>
<td>
How many nanoseconds away from time zero?
</td>
</tr>
</table>



## Methods

<h3 id="raw_picos"><code>raw_picos</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/timestamp.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>raw_picos() -> float
</code></pre>

The timestamp's location in picoseconds from arbitrary time zero.


<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/timestamp.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other
) -> "Timestamp"
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/timestamp.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__ge__"><code>__ge__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/timestamp.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ge__(
    other
)
</code></pre>

Return self>=value.


<h3 id="__gt__"><code>__gt__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/timestamp.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    other
)
</code></pre>

Return self>value.


<h3 id="__le__"><code>__le__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/timestamp.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    other
)
</code></pre>

Return self<=value.


<h3 id="__lt__"><code>__lt__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/timestamp.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    other
)
</code></pre>

Return self<value.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/timestamp.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Return self!=value.


<h3 id="__radd__"><code>__radd__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/timestamp.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__radd__(
    other
) -> "Timestamp"
</code></pre>




<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/timestamp.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    other
)
</code></pre>






