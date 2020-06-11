<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.parse_random_state" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.value.parse_random_state

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/random_state.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Interpret an object as a pseudorandom number generator.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.value.random_state.parse_random_state`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.parse_random_state(
    random_state: RANDOM_STATE_OR_SEED_LIKE
) -> np.random.RandomState
</code></pre>



<!-- Placeholder for "Used in" -->

If `random_state` is None, returns the module `np.random`.
If `random_state` is an integer, returns
`np.random.RandomState(random_state)`.
Otherwise, returns `random_state` unmodified.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`random_state`
</td>
<td>
The object to be used as or converted to a pseudorandom
number generator.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The pseudorandom number generator object.
</td>
</tr>

</table>

