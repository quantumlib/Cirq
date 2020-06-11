<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.Calibration" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="heatmap"/>
<meta itemprop="property" content="items"/>
<meta itemprop="property" content="keys"/>
<meta itemprop="property" content="timestamp_str"/>
<meta itemprop="property" content="values"/>
</div>

# cirq.google.Calibration

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/calibration.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A convenience wrapper for calibrations that acts like a dictionary.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.engine.Calibration`, `cirq.google.engine.calibration.Calibration`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.Calibration(
    calibration: <a href="../../cirq/google/api/v2/metrics_pb2/MetricsSnapshot.md"><code>cirq.google.api.v2.metrics_pb2.MetricsSnapshot</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Calibrations act as dictionaries whose keys are the names of the metric,
and whose values are the metric values.  The metric values themselves are
represented as a dictionary.  These metric value dictionaries have
keys that are tuples of <a href="../../cirq/devices/GridQubit.md"><code>cirq.GridQubit</code></a>s and values that are lists of the
metric values for those qubits. If a metric acts globally and is attached
to no specified number of qubits, the map will be from the empty tuple
to the metrics values.

Calibrations act just like a python dictionary. For example you can get
a list of all of the metric names using

    `calibration.keys()`

and query a single value by looking up the name by index:

    `calibration['t1']`



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`timestamp`
</td>
<td>
The time that this calibration was run, in milliseconds since
the epoch.
</td>
</tr>
</table>



## Methods

<h3 id="get"><code>get</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get(
    key, default=None
)
</code></pre>

D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.


<h3 id="heatmap"><code>heatmap</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/calibration.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>heatmap(
    key: str
) -> <a href="../../cirq/vis/Heatmap.md"><code>cirq.vis.Heatmap</code></a>
</code></pre>

Return a heatmap for metrics that target single qubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`key`
</td>
<td>
The metric key to return a heatmap for.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../cirq/vis/Heatmap.md"><code>cirq.Heatmap</code></a> for the metric.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
AssertionError if the heatmap is not for single qubits or the metric
values are not single floats.
</td>
</tr>

</table>



<h3 id="items"><code>items</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>items()
</code></pre>

D.items() -> a set-like object providing a view on D's items


<h3 id="keys"><code>keys</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>keys()
</code></pre>

D.keys() -> a set-like object providing a view on D's keys


<h3 id="timestamp_str"><code>timestamp_str</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/calibration.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>timestamp_str(
    tz: Optional[datetime.tzinfo] = None,
    timespec: str = 'auto'
) -> str
</code></pre>

Return a string for the calibration timestamp.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tz`
</td>
<td>
The timezone for the string. If None, the method uses the
platform's local date and time.
</td>
</tr><tr>
<td>
`timespec`
</td>
<td>
See datetime.isoformat for valid values.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The string in ISO 8601 format YYYY-MM-DDTHH:MM:SS.ffffff.
</td>
</tr>

</table>



<h3 id="values"><code>values</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>values()
</code></pre>

D.values() -> an object providing a view on D's values


<h3 id="__contains__"><code>__contains__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__(
    key
)
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/calibration.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    key: str
) -> Dict[Tuple['cirq.GridQubit', ...], Any]
</code></pre>

Supports getting calibrations by index.

Calibration may be accessed by key:

    `calibration['t1']`.

This returns a map from tuples of <a href="../../cirq/devices/GridQubit.md"><code>cirq.GridQubit</code></a>s to a list of the
values of the metric. If there are no targets, the only key will only
be an empty tuple.

<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/calibration.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__() -> Iterator
</code></pre>




<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/calibration.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__() -> int
</code></pre>






