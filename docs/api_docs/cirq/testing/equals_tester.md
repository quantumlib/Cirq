<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.equals_tester" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.testing.equals_tester

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/equals_tester.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A utility class for testing equality methods.


To test an equality method, create an EqualityTester and add several groups
of items to it. The equality tester will check that the items within each
group are all equal to each other, but that items between each group are never
equal to each other. It will also check that a==b implies hash(a)==hash(b).

## Classes

[`class EqualsTester`](../../cirq/testing/EqualsTester.md): Tests equality against user-provided disjoint equivalence groups.

