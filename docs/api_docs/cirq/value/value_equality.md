<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.value_equality" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.value.value_equality

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Implements __eq__/__ne__/__hash__ via a _value_equality_values_ method.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.value_equality`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.value_equality(
    *,
    unhashable: bool = False,
    distinct_child_types: bool = False,
    manual_cls: bool = False,
    approximate: bool = False
) -> Union[Callable[[type], type], type]
</code></pre>



<!-- Placeholder for "Used in" -->

_value_equality_values_ is a method that the decorated class must implement.

_value_equality_approximate_values_ is a method that the decorated class
might implement if special support for approximate equality is required.
This is only used when approximate argument is set. When approximate
argument is set and _value_equality_approximate_values_ is not defined,
_value_equality_values_ values are used for approximate equality.
For example, this can be used to compare periodic values like angles: the
angle value can be wrapped with `PeriodicValue`. When returned as part of
approximate values a special normalization will be done automatically to
guarantee correctness.

Note that the type of the decorated value is included as part of the value
equality values. This is so that completely separate classes with identical
equality values (e.g. a Point2D and a Vector2D) don't compare as equal.
Further note that this means that child types of the decorated type will be
considered equal to each other, though this behavior can be changed via
the 'distinct_child_types` argument. The type logic is implemented behind
the scenes by a `_value_equality_values_cls_` method added to the class.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cls`
</td>
<td>
The type to decorate. Automatically passed in by python when using
the @cirq.value_equality decorator notation on a class.
</td>
</tr><tr>
<td>
`unhashable`
</td>
<td>
When set, the __hash__ method will be set to None instead of
to a hash of the equality class and equality values. Useful for
mutable types such as dictionaries.
</td>
</tr><tr>
<td>
`distinct_child_types`
</td>
<td>
When set, classes that inherit from the decorated
class will not be considered equal to it. Also, different child
classes will not be considered equal to each other. Useful for when
the decorated class is an abstract class or trait that is helping to
define equality for many conceptually distinct concrete classes.
</td>
</tr><tr>
<td>
`manual_cls`
</td>
<td>
When set, the method '_value_equality_values_cls_' must be
implemented. This allows a new class to compare as equal to another
existing class that is also using value equality, by having the new
class return the existing class' type.
Incompatible with `distinct_child_types`.
</td>
</tr><tr>
<td>
`approximate`
</td>
<td>
When set, the decorated class will be enhanced with
`_approx_eq_` implementation and thus start to support the
`SupportsApproximateEquality` protocol.
</td>
</tr>
</table>

