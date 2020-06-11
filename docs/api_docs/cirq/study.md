<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.study" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="UnitSweep"/>
</div>

# Module: cirq.study

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Types and methods for running studies (repeated trials).



## Modules

[`flatten_expressions`](../cirq/study/flatten_expressions.md) module: Resolves symbolic expressions to unique symbols.

[`resolver`](../cirq/study/resolver.md) module: Resolves ParameterValues to assigned values.

[`sweepable`](../cirq/study/sweepable.md) module: Defines which types are Sweepable.

[`sweeps`](../cirq/study/sweeps.md) module

[`trial_result`](../cirq/study/trial_result.md) module: Defines trial results.

[`visualize`](../cirq/study/visualize.md) module: Tool to visualize the results of a study.

## Classes

[`class ExpressionMap`](../cirq/study/ExpressionMap.md): A dictionary with sympy expressions and symbols for keys and sympy

[`class Linspace`](../cirq/study/Linspace.md): A simple sweep over linearly-spaced values.

[`class ListSweep`](../cirq/study/ListSweep.md): A wrapper around a list of `ParamResolver`s.

[`class ParamResolver`](../cirq/study/ParamResolver.md): Resolves sympy.Symbols to actual values.

[`class Points`](../cirq/study/Points.md): A simple sweep with explicitly supplied values.

[`class Product`](../cirq/study/Product.md): Cartesian product of one or more sweeps.

[`class Sweep`](../cirq/study/Sweep.md): A sweep is an iterator over ParamResolvers.

[`class TrialResult`](../cirq/study/TrialResult.md): The results of multiple executions of a circuit with fixed parameters.

[`class Zip`](../cirq/study/Zip.md): Zip product (direct sum) of one or more sweeps.

## Functions

[`flatten(...)`](../cirq/study/flatten.md): Creates a copy of `val` with any symbols or expressions replaced with

[`flatten_with_params(...)`](../cirq/study/flatten_with_params.md): Creates a copy of `val` with any symbols or expressions replaced with

[`flatten_with_sweep(...)`](../cirq/study/flatten_with_sweep.md): Creates a copy of `val` with any symbols or expressions replaced with

[`plot_state_histogram(...)`](../cirq/study/plot_state_histogram.md): Plot the state histogram from a single result with repetitions.

[`to_resolvers(...)`](../cirq/study/to_resolvers.md): Convert a Sweepable to a list of ParamResolvers.

[`to_sweep(...)`](../cirq/study/to_sweep.md): Converts the argument into a `<a href="../cirq/study/Sweep.md"><code>cirq.Sweep</code></a>`.

[`to_sweeps(...)`](../cirq/study/to_sweeps.md): Converts a Sweepable to a list of Sweeps.

## Type Aliases

[`ParamDictType`](../cirq/study/ParamDictType.md)

[`ParamResolverOrSimilarType`](../cirq/study/ParamResolverOrSimilarType.md)

[`Sweepable`](../cirq/study/Sweepable.md)

## Other Members

* `UnitSweep` <a id="UnitSweep"></a>
