<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.study.to_sweep" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.study.to_sweep

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/sweepable.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Converts the argument into a `<a href="../../cirq/study/Sweep.md"><code>cirq.Sweep</code></a>`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.study.sweepable.to_sweep`, `cirq.to_sweep`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.study.to_sweep(
    sweep_or_resolver_list: Union['Sweep', ParamResolverOrSimilarType, Iterable[ParamResolverOrSimilarType]
        ]
) -> "Sweep"
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sweep_or_resolver_list`
</td>
<td>
The object to try to turn into a
`<a href="../../cirq/study/Sweep.md"><code>cirq.Sweep</code></a>` . A `<a href="../../cirq/study/Sweep.md"><code>cirq.Sweep</code></a>`, a single `<a href="../../cirq/study/ParamResolver.md"><code>cirq.ParamResolver</code></a>`,
or a list of `<a href="../../cirq/study/ParamResolver.md"><code>cirq.ParamResolver</code></a>` s.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A sweep equal to or containing the argument.
</td>
</tr>

</table>

