<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.study.TrialResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_single_parameter_set"/>
<meta itemprop="property" content="histogram"/>
<meta itemprop="property" content="multi_measurement_histogram"/>
</div>

# cirq.study.TrialResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/trial_result.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The results of multiple executions of a circuit with fixed parameters.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.TrialResult`, `cirq.study.trial_result.TrialResult`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.study.TrialResult(
    *,
    params: <a href="../../cirq/study/ParamResolver.md"><code>cirq.study.ParamResolver</code></a>,
    measurements: Dict[str, np.ndarray]
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
Stored as a Pandas DataFrame that can be accessed through the "data"
attribute. The repetition number is the row index and measurement keys
are the columns of the DataFrame. Each element is a big endian integer
representation of measurement outcomes for the measurement key in that
repitition.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`params`
</td>
<td>
A ParamResolver of settings used when sampling result.
</td>
</tr><tr>
<td>
`data`
</td>
<td>

</td>
</tr><tr>
<td>
`measurements`
</td>
<td>

</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="from_single_parameter_set"><code>from_single_parameter_set</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/trial_result.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_single_parameter_set(
    *,
    params: <a href="../../cirq/study/ParamResolver.md"><code>cirq.study.ParamResolver</code></a>,
    measurements: Dict[str, np.ndarray]
) -> "TrialResult"
</code></pre>

Packages runs of a single parameterized circuit into a TrialResult.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`params`
</td>
<td>
A ParamResolver of settings used for this result.
</td>
</tr><tr>
<td>
`measurements`
</td>
<td>
A dictionary from measurement gate key to measurement
results. The value for each key is a 2-D array of booleans,
with the first index running over the repetitions, and the
second index running over the qubits for the corresponding
measurements.
</td>
</tr>
</table>



<h3 id="histogram"><code>histogram</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/trial_result.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>histogram(
    *,
    key: <a href="../../cirq/study/trial_result/TMeasurementKey.md"><code>cirq.study.trial_result.TMeasurementKey</code></a> = value.big_endian_bits_to_int,
    fold_func: Callable[[Tuple], T] = cirq.value.big_endian_bits_to_int
) -> collections.Counter
</code></pre>

Counts the number of times a measurement result occurred.

For example, suppose that:

    - fold_func is not specified
    - key='abc'
    - the measurement with key 'abc' measures qubits a, b, and c.
    - the circuit was sampled 3 times.
    - the sampled measurement values were:
        1. a=1 b=0 c=0
        2. a=0 b=1 c=0
        3. a=1 b=0 c=0

Then the counter returned by this method will be:

    collections.Counter({
        0b100: 2,
        0b010: 1
    })

Where '0b100' is binary for '4' and '0b010' is binary for '2'. Notice
that the bits are combined in a big-endian way by default, with the
first measured qubit determining the highest-value bit.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`key`
</td>
<td>
Keys of measurements to include in the histogram.
</td>
</tr><tr>
<td>
`fold_func`
</td>
<td>
A function used to convert a sampled measurement result
into a countable value. The input is a list of bits sampled
together by a measurement. If this argument is not specified,
it defaults to interpreting the bits as a big endian
integer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A counter indicating how often a measurement sampled various
results.
</td>
</tr>

</table>



<h3 id="multi_measurement_histogram"><code>multi_measurement_histogram</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/trial_result.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>multi_measurement_histogram(
    *,
    keys: Iterable[<a href="../../cirq/study/trial_result/TMeasurementKey.md"><code>cirq.study.trial_result.TMeasurementKey</code></a>] = _tuple_of_big_endian_int,
    fold_func: Callable[[Tuple], T] = <function _tuple_of_big_endian_int at 0x7f2de9319320>
) -> collections.Counter
</code></pre>

Counts the number of times combined measurement results occurred.

This is a more general version of the 'histogram' method. Instead of
only counting how often results occurred for one specific measurement,
this method tensors multiple measurement results together and counts
how often the combined results occurred.

For example, suppose that:

    - fold_func is not specified
    - keys=['abc', 'd']
    - the measurement with key 'abc' measures qubits a, b, and c.
    - the measurement with key 'd' measures qubit d.
    - the circuit was sampled 3 times.
    - the sampled measurement values were:
        1. a=1 b=0 c=0 d=0
        2. a=0 b=1 c=0 d=1
        3. a=1 b=0 c=0 d=0

Then the counter returned by this method will be:

    collections.Counter({
        (0b100, 0): 2,
        (0b010, 1): 1
    })


Where '0b100' is binary for '4' and '0b010' is binary for '2'. Notice
that the bits are combined in a big-endian way by default, with the
first measured qubit determining the highest-value bit.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`fold_func`
</td>
<td>
A function used to convert sampled measurement results
into countable values. The input is a tuple containing the
list of bits measured by each measurement specified by the
keys argument. If this argument is not specified, it defaults
to returning tuples of integers, where each integer is the big
endian interpretation of the bits a measurement sampled.
</td>
</tr><tr>
<td>
`keys`
</td>
<td>
Keys of measurements to include in the histogram.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A counter indicating how often measurements sampled various
results.
</td>
</tr>

</table>



<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/trial_result.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other: "cirq.TrialResult"
) -> "cirq.TrialResult"
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/trial_result.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.




