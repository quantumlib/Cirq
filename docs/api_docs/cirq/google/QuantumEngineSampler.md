<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.QuantumEngineSampler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run"/>
<meta itemprop="property" content="run_async"/>
<meta itemprop="property" content="run_sweep"/>
<meta itemprop="property" content="run_sweep_async"/>
<meta itemprop="property" content="sample"/>
</div>

# cirq.google.QuantumEngineSampler

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_sampler.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A sampler that samples from processors managed by the Quantum Engine.

Inherits From: [`Sampler`](../../cirq/work/Sampler.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.engine.QuantumEngineSampler`, `cirq.google.engine.engine.engine_sampler.QuantumEngineSampler`, `cirq.google.engine.engine_sampler.QuantumEngineSampler`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.QuantumEngineSampler(
    *,
    engine: "cirq.google.Engine",
    processor_id: Union[str, List[str]],
    gate_set: "cirq.google.SerializableGateSet"
)
</code></pre>



<!-- Placeholder for "Used in" -->

Exposes a <a href="../../cirq/google/Engine.md"><code>cirq.google.Engine</code></a> instance as a <a href="../../cirq/work/Sampler.md"><code>cirq.Sampler</code></a>.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`engine`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="run"><code>run</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/sampler.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run(
    program: "cirq.Circuit",
    param_resolver: "cirq.ParamResolverOrSimilarType" = None,
    repetitions: int = 1
) -> "cirq.TrialResult"
</code></pre>

Samples from the given Circuit.

By default, the `run_async` method invokes this method on another
thread. So this method is supposed to be thread safe.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit to sample from.
</td>
</tr><tr>
<td>
`param_resolver`
</td>
<td>
Parameters to run with the program.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of times to sample.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
TrialResult for a run.
</td>
</tr>

</table>



<h3 id="run_async"><code>run_async</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/sampler.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run_async(
    program, *, repetitions
)
</code></pre>

Asynchronously samples from the given Circuit.

By default, this method invokes `run` synchronously and simply exposes
its result is an awaitable. Child classes that are capable of true
asynchronous sampling should override it to use other strategies.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit to sample from.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of times to sample.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An awaitable TrialResult.
</td>
</tr>

</table>



<h3 id="run_sweep"><code>run_sweep</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine_sampler.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run_sweep(
    program: Union['cirq.Circuit', 'cirq.google.EngineProgram'],
    params: "cirq.Sweepable",
    repetitions: int = 1
) -> List['cirq.TrialResult']
</code></pre>

Samples from the given Circuit.

In contrast to run, this allows for sweeping over different parameter
values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit to sample from.
</td>
</tr><tr>
<td>
`params`
</td>
<td>
Parameters to run with the program.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of times to sample.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
TrialResult list for this run; one for each possible parameter
resolver.
</td>
</tr>

</table>



<h3 id="run_sweep_async"><code>run_sweep_async</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/sampler.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run_sweep_async(
    program, params, repetitions=1
)
</code></pre>

Asynchronously sweeps and samples from the given Circuit.

By default, this method invokes `run_sweep` synchronously and simply
exposes its result is an awaitable. Child classes that are capable of
true asynchronous sampling should override it to use other strategies.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit to sample from.
</td>
</tr><tr>
<td>
`params`
</td>
<td>
One or more mappings from parameter keys to parameter values
to use. For each parameter assignment, `repetitions` samples
will be taken.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of times to sample.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An awaitable TrialResult.
</td>
</tr>

</table>



<h3 id="sample"><code>sample</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/work/sampler.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sample(
    program: "cirq.Circuit",
    *,
    repetitions: int = 1,
    params: "cirq.Sweepable" = None
) -> "pd.DataFrame"
</code></pre>

Samples the given Circuit, producing a pandas data frame.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit to sample from.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of times to sample the program, for each
parameter mapping.
</td>
</tr><tr>
<td>
`params`
</td>
<td>
Maps symbols to one or more values. This argument can be
a dictionary, a list of dictionaries, a <a href="../../cirq/study/Sweep.md"><code>cirq.Sweep</code></a>, a list of
<a href="../../cirq/study/Sweep.md"><code>cirq.Sweep</code></a>, etc. The program will be sampled `repetition`
times for each mapping. Defaults to a single empty mapping.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `pandas.DataFrame` with a row for each sample, and a column for
each measurement result as well as a column for each symbolic
parameter. There is an also index column containing the repetition
number, for each parameter assignment.
</td>
</tr>

</table>



#### Examples:

>>> a, b, c = cirq.LineQubit.range(3)
>>> sampler = cirq.Simulator()
>>> circuit = cirq.Circuit(cirq.X(a),
...                        cirq.measure(a, key='out'))
>>> print(sampler.sample(circuit, repetitions=4))
   out
0    1
1    1
2    1
3    1

```
>>> circuit = cirq.Circuit(cirq.X(a),
...                        cirq.CNOT(a, b),
...                        cirq.measure(a, b, c, key='out'))
>>> print(sampler.sample(circuit, repetitions=4))
   out
0    6
1    6
2    6
3    6
```

```
>>> circuit = cirq.Circuit(cirq.X(a)**sympy.Symbol('t'),
...                        cirq.measure(a, key='out'))
>>> print(sampler.sample(
...     circuit,
...     repetitions=3,
...     params=[{'t': 0}, {'t': 1}]))
   t  out
0  0    0
1  0    0
2  0    0
0  1    1
1  1    1
2  1    1
```




