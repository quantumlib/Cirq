<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.OrderTester" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_ascending"/>
<meta itemprop="property" content="add_ascending_equivalence_group"/>
</div>

# cirq.testing.OrderTester

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/order_tester.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Tests ordering against user-provided disjoint ordered groups or items.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.order_tester.OrderTester`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.OrderTester()
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="add_ascending"><code>add_ascending</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/order_tester.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_ascending(
    *items
)
</code></pre>

Tries to add a sequence of ascending items to the order tester.

This methods asserts that items must all be ascending
with regard to both each other and the elements which have been already
added during previous calls.
Some of the previously added elements might be equivalence groups,
which are supposed to be equal to each other within that group.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*items`
</td>
<td>
The sequence of strictly ascending items.
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
Items are not ascending either
with regard to each other, or with regard to the elements
which have been added before.
</td>
</tr>
</table>



<h3 id="add_ascending_equivalence_group"><code>add_ascending_equivalence_group</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/order_tester.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_ascending_equivalence_group(
    *group_items
)
</code></pre>

Tries to add an ascending equivalence group to the order tester.

Asserts that the group items are equal to each other, but strictly
ascending with regard to the already added groups.

Adds the objects as a group.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`group_items`
</td>
<td>
items making the equivalence group
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
The group elements aren't equal to each other,
or items in another group overlap with the new group.
</td>
</tr>
</table>





