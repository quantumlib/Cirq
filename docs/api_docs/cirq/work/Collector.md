<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.work.Collector" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="collect"/>
<meta itemprop="property" content="collect_async"/>
<meta itemprop="property" content="next_job"/>
<meta itemprop="property" content="on_job_result"/>
</div>

# cirq.work.Collector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/collector.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Collects data from a sampler, in parallel, towards some purpose.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.Collector`, `cirq.work.collector.Collector`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Child classes must override the `next_job` and `on_job_result` methods,
which respectively determine what to sample and how to process the results.
Utility methods on the base class such as `collect` and `collect_async` can
then be given a sampler to collect from, and will request samples with some
specified amount of parallelism.

## Methods

<h3 id="collect"><code>collect</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/collector.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>collect(
    sampler: "cirq.Sampler",
    *,
    concurrency: int = 2,
    max_total_samples: Optional[int] = None
) -> None
</code></pre>

Collects needed samples from a sampler.


#### Examples:


```
collector = cirq.PauliStringCollector(...)
sampler.collect(collector, concurrency=3)
print(collector.estimated_energy())
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`sampler`
</td>
<td>
The simulator or service to collect samples from.
</td>
</tr><tr>
<td>
`concurrency`
</td>
<td>
Desired number of sampling jobs to have in flight at
any given time.
</td>
</tr><tr>
<td>
`max_total_samples`
</td>
<td>
Optional limit on the maximum number of samples
to collect.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The collector's result after all desired samples have been
collected.
</td>
</tr>

</table>



#### See Also:

Python 3 documentation "Coroutines and Tasks"
https://docs.python.org/3/library/asyncio-task.html


<h3 id="collect_async"><code>collect_async</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/collector.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>collect_async(
    sampler, *, concurrency=2, max_total_samples=None
)
</code></pre>

Asynchronously collects needed samples from a sampler.


#### Examples:


```
collector = cirq.PauliStringCollector(...)
await sampler.collect_async(collector, concurrency=3)
print(collector.estimated_energy())
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`sampler`
</td>
<td>
The simulator or service to collect samples from.
</td>
</tr><tr>
<td>
`concurrency`
</td>
<td>
Desired number of sampling jobs to have in flight at
any given time.
</td>
</tr><tr>
<td>
`max_total_samples`
</td>
<td>
Optional limit on the maximum number of samples
to collect.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The collector's result after all desired samples have been
collected.
</td>
</tr>

</table>



#### See Also:

Python 3 documentation "Coroutines and Tasks"
https://docs.python.org/3/library/asyncio-task.html


<h3 id="next_job"><code>next_job</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/collector.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>next_job() -> Optional[<a href="../../cirq/work/CircuitSampleJob.md"><code>cirq.work.CircuitSampleJob</code></a>]
</code></pre>

Determines what to sample next.

This method is called by driving code when more samples can be
requested.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A CircuitSampleJob describing the circuit to sample, how many
samples to take, and a key value that can be used in the
`on_job_result` method to recognize which job this is.

Can also return a nested iterable of such jobs.

Returning None, an empty list, or any other result which flattens
into an empty list of work, indicates that the driving code should
await more results (and pass them into on_job_results) before
bothering to ask for more jobs again.
</td>
</tr>

</table>



<h3 id="on_job_result"><code>on_job_result</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/collector.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>on_job_result(
    job: <a href="../../cirq/work/CircuitSampleJob.md"><code>cirq.work.CircuitSampleJob</code></a>,
    result: <a href="../../cirq/study/TrialResult.md"><code>cirq.study.TrialResult</code></a>
) -> None
</code></pre>

Incorporates sampled results.

This method is called by driving code when sample results have become
available.

The results should be incorporated into the collector's state.



