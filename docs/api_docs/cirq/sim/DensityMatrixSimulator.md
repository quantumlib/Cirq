<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim.DensityMatrixSimulator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run"/>
<meta itemprop="property" content="run_async"/>
<meta itemprop="property" content="run_sweep"/>
<meta itemprop="property" content="run_sweep_async"/>
<meta itemprop="property" content="sample"/>
<meta itemprop="property" content="simulate"/>
<meta itemprop="property" content="simulate_moment_steps"/>
<meta itemprop="property" content="simulate_sweep"/>
</div>

# cirq.sim.DensityMatrixSimulator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/density_matrix_simulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A simulator for density matrices and noisy quantum circuits.

Inherits From: [`SimulatesSamples`](../../cirq/sim/SimulatesSamples.md), [`SimulatesIntermediateState`](../../cirq/sim/SimulatesIntermediateState.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.DensityMatrixSimulator`, `cirq.sim.density_matrix_simulator.DensityMatrixSimulator`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.sim.DensityMatrixSimulator(
    *,
    dtype: Type[np.number] = np.complex64,
    noise: "cirq.NOISE_MODEL_LIKE" = None,
    seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None,
    ignore_measurement_results: bool = False
)
</code></pre>



<!-- Placeholder for "Used in" -->

This simulator can be applied on circuits that are made up of operations
that have:
    * a `_channel_` method
    * a `_mixture_` method for a probabilistic combination of unitary gates.
    * a `_unitary_` method
    * a `_has_unitary_` and `_apply_unitary_` method.
    * measurements
    * a `_decompose_` that eventually yields one of the above
That is, the circuit must have elements that follow on of the protocols:
    * <a href="../../cirq/protocols/SupportsChannel.md"><code>cirq.SupportsChannel</code></a>
    * <a href="../../cirq/protocols/SupportsMixture.md"><code>cirq.SupportsMixture</code></a>
    * <a href="../../cirq/protocols/SupportsConsistentApplyUnitary.md"><code>cirq.SupportsConsistentApplyUnitary</code></a>
    * <a href="../../cirq/protocols/SupportsUnitary.md"><code>cirq.SupportsUnitary</code></a>
    * <a href="../../cirq/protocols/SupportsDecompose.md"><code>cirq.SupportsDecompose</code></a>
or is a measurement.

This simulator supports three types of simulation.

Run simulations which mimic running on actual quantum hardware. These
simulations do not give access to the density matrix (like actual hardware).
There are two variations of run methods, one which takes in a single
(optional) way to resolve parameterized circuits, and a second which
takes in a list or sweep of parameter resolver:

    run(circuit, param_resolver, repetitions)

    run_sweep(circuit, params, repetitions)

These methods return `TrialResult`s which contain both the measurement
results, but also the parameters used for the parameterized
circuit operations. The initial state of a run is always the all 0s state
in the computational basis.

By contrast the simulate methods of the simulator give access to the density
matrix of the simulation at the end of the simulation of the circuit.
Note that if the circuit contains measurements then the density matrix
is that result for those particular measurement results. For example
if there is one measurement, then the simulation may result in the
measurement result for this measurement, and the density matrix will
be that conditional on that result. It will not be the density matrix formed
by summing over the different measurements and their probabilities.
The simulate methods take in two parameters that the run methods do not: a
qubit order and an initial state. The qubit order is necessary because an
ordering must be chosen for the kronecker product (see
`DensityMatrixTrialResult` for details of this ordering). The initial
state can be either the full density matrix, the full wave function (for
pure states), or an integer which represents the initial state of being
in a computational basis state for the binary representation of that
integer. Similar to run methods, there are two simulate methods that run
for single simulations or for sweeps across different parameters:

    simulate(circuit, param_resolver, qubit_order, initial_state)

    simulate_sweep(circuit, params, qubit_order, initial_state)

The simulate methods in contrast to the run methods do not perform
repetitions. The result of these simulations is a
`DensityMatrixTrialResult` which contains, in addition to measurement
results and information about the parameters that were used in the
simulation, access to the density matrix via the `density_matrix` method.

If one wishes to perform simulations that have access to the
density matrix as one steps through running the circuit there is a generator
which can be iterated over and each step is an object that gives access
to the density matrix.  This stepping through a `Circuit` is done on a
`Moment` by `Moment` manner.

    simulate_moment_steps(circuit, param_resolver, qubit_order,
                          initial_state)

One can iterate over the moments via

    for step_result in simulate_moments(circuit):
       # do something with the density matrix via
       # step_result.density_matrix()

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dtype`
</td>
<td>
The `numpy.dtype` used by the simulation. One of
`numpy.complex64` or `numpy.complex128`
</td>
</tr><tr>
<td>
`noise`
</td>
<td>
A noise model to apply while simulating.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
The random seed to use for this simulator.
</td>
</tr><tr>
<td>
`ignore_measurement_results`
</td>
<td>
if True, then the simulation
will treat measurement as dephasing instead of collapsing
process.

Example:
>>> (q0,) = cirq.LineQubit.range(1)
>>> circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))

Default case (ignore_measurement_results = False):
>>> simulator = cirq.DensityMatrixSimulator()
>>> result = simulator.run(circuit)

The measurement result will be strictly one of 0 or 1.

In the other case:
>>> simulator = cirq.DensityMatrixSimulator(
...     ignore_measurement_results = True)
>>> result = simulator.run(circuit)

The measurement result will be the maximally mixed state
with equal probability for 0 and 1.
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run_sweep(
    program: "cirq.Circuit",
    params: <a href="../../cirq/study/Sweepable.md"><code>cirq.study.Sweepable</code></a>,
    repetitions: int = 1
) -> List[<a href="../../cirq/study/TrialResult.md"><code>cirq.study.TrialResult</code></a>]
</code></pre>

Runs the supplied Circuit, mimicking quantum hardware.

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
The circuit to simulate.
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
The number of repetitions to simulate.
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


<h3 id="simulate"><code>simulate</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>simulate(
    program: "cirq.Circuit",
    param_resolver: "study.ParamResolverOrSimilarType" = None,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT,
    initial_state: Any = None
) -> "SimulationTrialResult"
</code></pre>

Simulates the supplied Circuit.

This method returns a result which allows access to the entire
simulator's final state.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit to simulate.
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
`qubit_order`
</td>
<td>
Determines the canonical ordering of the qubits. This
is often used in specifying the initial state, i.e. the
ordering of the computational basis states.
</td>
</tr><tr>
<td>
`initial_state`
</td>
<td>
The initial state for the simulation. The  form of
this state depends on the simulation implementation.  See
documentation of the implementing class for details.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
SimulationTrialResults for the simulation. Includes the final state.
</td>
</tr>

</table>



<h3 id="simulate_moment_steps"><code>simulate_moment_steps</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>simulate_moment_steps(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    param_resolver: "study.ParamResolverOrSimilarType" = None,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT,
    initial_state: Any = None
) -> Iterator
</code></pre>

Returns an iterator of StepResults for each moment simulated.

If the circuit being simulated is empty, a single step result should
be returned with the state being set to the initial state.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The Circuit to simulate.
</td>
</tr><tr>
<td>
`param_resolver`
</td>
<td>
A ParamResolver for determining values of Symbols.
</td>
</tr><tr>
<td>
`qubit_order`
</td>
<td>
Determines the canonical ordering of the qubits. This
is often used in specifying the initial state, i.e. the
ordering of the computational basis states.
</td>
</tr><tr>
<td>
`initial_state`
</td>
<td>
The initial state for the simulation. The form of
this state depends on the simulation implementation. See
documentation of the implementing class for details.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Iterator that steps through the simulation, simulating each
moment and returning a StepResult for each moment.
</td>
</tr>

</table>



<h3 id="simulate_sweep"><code>simulate_sweep</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/simulator.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>simulate_sweep(
    program: "cirq.Circuit",
    params: <a href="../../cirq/study/Sweepable.md"><code>cirq.study.Sweepable</code></a>,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT,
    initial_state: Any = None
) -> List['SimulationTrialResult']
</code></pre>

Simulates the supplied Circuit.

This method returns a result which allows access to the entire
state vector. In contrast to simulate, this allows for sweeping
over different parameter values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`program`
</td>
<td>
The circuit to simulate.
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
`qubit_order`
</td>
<td>
Determines the canonical ordering of the qubits. This
is often used in specifying the initial state, i.e. the
ordering of the computational basis states.
</td>
</tr><tr>
<td>
`initial_state`
</td>
<td>
The initial state for the simulation. The form of
this state depends on the simulation implementation. See
documentation of the implementing class for details.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of SimulationTrialResults for this run, one for each
possible parameter resolver.
</td>
</tr>

</table>





