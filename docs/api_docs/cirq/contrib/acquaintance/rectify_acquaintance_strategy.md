<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.rectify_acquaintance_strategy" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.acquaintance.rectify_acquaintance_strategy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/mutation_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Splits moments so that they contain either only acquaintance gates

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.acquaintance.mutation_utils.rectify_acquaintance_strategy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.rectify_acquaintance_strategy(
    circuit: "cirq.Circuit",
    acquaint_first: bool = True
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
or only permutation gates. Orders resulting moments so that the first one
is of the same type as the previous one.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The acquaintance strategy to rectify.
</td>
</tr><tr>
<td>
`acquaint_first`
</td>
<td>
Whether to make acquaintance moment first in when
splitting the first mixed moment.
</td>
</tr>
</table>

