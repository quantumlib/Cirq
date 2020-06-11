<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.EqualsTester" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_equality_group"/>
<meta itemprop="property" content="make_equality_group"/>
</div>

# cirq.testing.EqualsTester

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/equals_tester.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Tests equality against user-provided disjoint equivalence groups.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.equals_tester.EqualsTester`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.EqualsTester()
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="add_equality_group"><code>add_equality_group</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/equals_tester.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_equality_group(
    *group_items
)
</code></pre>

Tries to add a disjoint equivalence group to the equality tester.

This methods asserts that items within the group must all be equal to
each other, but not equal to any items in other groups that have been
or will be added.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*group_items`
</td>
<td>
The items making up the equivalence group.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`AssertionError`
</td>
<td>
Items within the group are not equal to each other,
or items in another group are equal to items within the new
group, or the items violate the equals-implies-same-hash rule.
</td>
</tr>
</table>



<h3 id="make_equality_group"><code>make_equality_group</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/equals_tester.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>make_equality_group(
    *factories
)
</code></pre>

Tries to add a disjoint equivalence group to the equality tester.

Uses the factory methods to produce two different objects with the same
initialization for each factory. Asserts that the objects are equal, but
not equal to any items in other groups that have been or will be added.
Adds the objects as a group.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`factories`
</td>
<td>
Methods for producing independent copies of an item.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`AssertionError`
</td>
<td>
The factories produce items not equal to the others,
or items in another group are equal to items from the factory,
or the items violate the equal-implies-same-hash rule.
</td>
</tr>
</table>





