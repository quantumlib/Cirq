<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.LinearDict" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__neg__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="__sub__"/>
<meta itemprop="property" content="__truediv__"/>
<meta itemprop="property" content="clean"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="fromkeys"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="items"/>
<meta itemprop="property" content="keys"/>
<meta itemprop="property" content="pop"/>
<meta itemprop="property" content="popitem"/>
<meta itemprop="property" content="setdefault"/>
<meta itemprop="property" content="update"/>
<meta itemprop="property" content="values"/>
<meta itemprop="property" content="TSelf"/>
</div>

# cirq.value.LinearDict

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Represents linear combination of things.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.LinearDict`, `cirq.value.linear_dict.LinearDict`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.LinearDict(
    terms: Optional[Mapping[<a href="../../cirq/value/linear_dict/TVector.md"><code>cirq.value.linear_dict.TVector</code></a>, <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>]] = None,
    validator: Callable[[<a href="../../cirq/value/linear_dict/TVector.md"><code>cirq.value.linear_dict.TVector</code></a>], bool] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

LinearDict implements the basic linear algebraic operations of vector
addition and scalar multiplication for linear combinations of abstract
vectors. Keys represent the vectors, values represent their coefficients.
The only requirement on the keys is that they be hashable (i.e. are
immutable and implement __hash__ and __eq__ with equal objects hashing
to equal values).

A consequence of treating keys as opaque is that all relationships between
the keys other than equality are ignored. In particular, keys are allowed
to be linearly dependent.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`terms`
</td>
<td>
Mapping of abstract vectors to coefficients in the linear
combination being initialized.
</td>
</tr><tr>
<td>
`validator`
</td>
<td>
Optional predicate that determines whether a vector is
valid or not. Dictionary and linear algebra operations that
would lead to the inclusion of an invalid vector into the
combination raise ValueError exception. By default all vectors
are valid.
</td>
</tr>
</table>



## Methods

<h3 id="clean"><code>clean</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>clean(
    *,
    atol: float = 1e-09
) -> "TSelf"
</code></pre>

Remove terms with coefficients of absolute value atol or less.


<h3 id="clear"><code>clear</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>clear()
</code></pre>

D.clear() -> None.  Remove all items from D.


<h3 id="copy"><code>copy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy() -> "TSelf"
</code></pre>




<h3 id="fromkeys"><code>fromkeys</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>fromkeys(
    vectors, coefficient=0
)
</code></pre>




<h3 id="get"><code>get</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get(
    vector, default=0
)
</code></pre>

D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.


<h3 id="items"><code>items</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>items() -> ItemsView[<a href="../../cirq/value/linear_dict/TVector.md"><code>cirq.value.linear_dict.TVector</code></a>, <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>]
</code></pre>

D.items() -> a set-like object providing a view on D's items


<h3 id="keys"><code>keys</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>keys() -> KeysView[<a href="../../cirq/value/linear_dict/TVector.md"><code>cirq.value.linear_dict.TVector</code></a>]
</code></pre>

D.keys() -> a set-like object providing a view on D's keys


<h3 id="pop"><code>pop</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pop(
    key, default=__marker
)
</code></pre>

D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
If key is not found, d is returned if given, otherwise KeyError is raised.

<h3 id="popitem"><code>popitem</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>popitem()
</code></pre>

D.popitem() -> (k, v), remove and return some (key, value) pair
as a 2-tuple; but raise KeyError if D is empty.

<h3 id="setdefault"><code>setdefault</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>setdefault(
    key, default=None
)
</code></pre>

D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D


<h3 id="update"><code>update</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update(
    *args, **kwargs
)
</code></pre>

D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.
If E present and has a .keys() method, does:     for k in E: D[k] = E[k]
If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
In either case, this is followed by: for k, v in F.items(): D[k] = v

<h3 id="values"><code>values</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>values() -> ValuesView[<a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>]
</code></pre>

D.values() -> an object providing a view on D's values


<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other: "TSelf"
) -> "TSelf"
</code></pre>




<h3 id="__bool__"><code>__bool__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__() -> bool
</code></pre>




<h3 id="__contains__"><code>__contains__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__(
    vector: Any
) -> bool
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: Any
) -> bool
</code></pre>

Checks whether two linear combinations are exactly equal.

Presence or absence of terms with coefficients exactly equal to
zero does not affect outcome.

Not appropriate for most practical purposes due to sensitivity to
numerical error in floating point coefficients. Use cirq.approx_eq()
instead.

<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    vector: <a href="../../cirq/value/linear_dict/TVector.md"><code>cirq.value.linear_dict.TVector</code></a>
) -> <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>
</code></pre>




<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__() -> Iterator[<a href="../../cirq/value/linear_dict/TVector.md"><code>cirq.value.linear_dict.TVector</code></a>]
</code></pre>




<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__() -> int
</code></pre>




<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    a: <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>
) -> "TSelf"
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: Any
) -> bool
</code></pre>

Checks whether two linear combinations are not exactly equal.

See __eq__().

<h3 id="__neg__"><code>__neg__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__neg__() -> "TSelf"
</code></pre>




<h3 id="__rmul__"><code>__rmul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    a: <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>
) -> "TSelf"
</code></pre>




<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    other: "TSelf"
) -> "TSelf"
</code></pre>




<h3 id="__truediv__"><code>__truediv__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/linear_dict.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__truediv__(
    a: <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>
) -> "TSelf"
</code></pre>






## Class Variables

* `TSelf` <a id="TSelf"></a>
