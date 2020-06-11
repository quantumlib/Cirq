<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.study.ListSweep" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="param_tuples"/>
</div>

# cirq.study.ListSweep

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/sweeps.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A wrapper around a list of `ParamResolver`s.

Inherits From: [`Sweep`](../../cirq/study/Sweep.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ListSweep`, `cirq.study.sweeps.ListSweep`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.study.ListSweep(
    resolver_list: Iterable[<a href="../../cirq/study/ParamResolverOrSimilarType.md"><code>cirq.study.ParamResolverOrSimilarType</code></a>]
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`resolver_list`
</td>
<td>
The list of parameter resolvers to use in the sweep.
All resolvers must resolve the same set of parameters.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`keys`
</td>
<td>
The keys for the all of the sympy.Symbols that are resolved.
</td>
</tr>
</table>



## Methods

<h3 id="param_tuples"><code>param_tuples</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/sweeps.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>param_tuples() -> Iterator[<a href="../../cirq/study/sweeps/Params.md"><code>cirq.study.sweeps.Params</code></a>]
</code></pre>

An iterator over (key, value) pairs assigning Symbol key to value.


<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/sweeps.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other: "Sweep"
) -> "Sweep"
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/sweeps.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/sweeps.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    val
)
</code></pre>




<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/sweeps.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__() -> Iterator[<a href="../../cirq/study/ParamResolver.md"><code>cirq.study.ParamResolver</code></a>]
</code></pre>




<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/sweeps.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__() -> int
</code></pre>




<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/sweeps.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    other: "Sweep"
) -> "Sweep"
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/sweeps.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Return self!=value.




