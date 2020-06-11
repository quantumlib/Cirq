<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.order_tester" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.testing.order_tester

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/order_tester.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A utility class for testing ordering methods.


To test an ordering method, create an OrderTester and add several
equivalence groups or items to it. The order tester will check that
the items within each group are all equal to each other, and every new
added item or group is strictly ascending with regard to the previously
added items or groups.

It will also check that a==b implies hash(a)==hash(b).

## Classes

[`class OrderTester`](../../cirq/testing/OrderTester.md): Tests ordering against user-provided disjoint ordered groups or items.

