<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.line.placement.anneal_minimize" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.line.placement.anneal_minimize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/line/placement/optimization.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Minimize solution using Simulated Annealing meta-heuristic.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.line.placement.optimization.anneal_minimize`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.line.placement.anneal_minimize(
    initial: <a href="../../../../cirq/google/line/placement/optimization/T.md"><code>cirq.google.line.placement.optimization.T</code></a>,
    cost_func: Callable[[T], float],
    move_func: Callable[[T], T],
    random_sample: Callable[[], float],
    temp_initial: float = 0.01,
    temp_final: float = 1e-06,
    cooling_factor: float = 0.99,
    repeat: int = 100,
    trace_func: Callable[[T, float, float, float, bool], None] = None
) -> <a href="../../../../cirq/google/line/placement/optimization/T.md"><code>cirq.google.line.placement.optimization.T</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`initial`
</td>
<td>
Initial solution of type T to the problem.
</td>
</tr><tr>
<td>
`cost_func`
</td>
<td>
Callable which takes current solution of type T, evaluates it
and returns float with the cost estimate. The better solution is,
the lower resulting value should be; negative values are allowed.
</td>
</tr><tr>
<td>
`move_func`
</td>
<td>
Callable which takes current solution of type T and returns a
new solution candidate of type T which is random iteration over
input solution. The input solution, which is argument to this
callback should not be mutated.
</td>
</tr><tr>
<td>
`random_sample`
</td>
<td>
Callable which gives uniformly sampled random value from
the [0, 1) interval on each call.
</td>
</tr><tr>
<td>
`temp_initial`
</td>
<td>
Optional initial temperature for simulated annealing
optimization. Scale of this value is cost_func-dependent.
</td>
</tr><tr>
<td>
`temp_final`
</td>
<td>
Optional final temperature for simulated annealing
optimization, where search should be stopped. Scale of this value is
cost_func-dependent.
</td>
</tr><tr>
<td>
`cooling_factor`
</td>
<td>
Optional factor to be applied to the current temperature
and give the new temperature, this must be strictly greater than 0
and strictly lower than 1.
</td>
</tr><tr>
<td>
`repeat`
</td>
<td>
Optional number of iterations to perform at each given
temperature.
</td>
</tr><tr>
<td>
`trace_func`
</td>
<td>
Optional callback for tracing simulated annealing progress.
This is going to be called at each algorithm step for the arguments:
solution candidate (T), current temperature (float), candidate cost
(float), probability of accepting candidate (float), and acceptance
decision (boolean).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The best solution found.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
When supplied arguments are invalid.
</td>
</tr>
</table>

