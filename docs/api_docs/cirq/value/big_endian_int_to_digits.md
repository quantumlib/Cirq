<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.big_endian_int_to_digits" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.value.big_endian_int_to_digits

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/digits.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Separates an integer into big-endian digits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.big_endian_int_to_digits`, `cirq.value.digits.big_endian_int_to_digits`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.big_endian_int_to_digits(
    val: int,
    *,
    digit_count: Optional[int] = None,
    base: Union[int, Iterable[int]]
) -> List[int]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The integer to get digits from. Must be non-negative and less than
the maximum representable value, given the specified base(s) and
digit count.
</td>
</tr><tr>
<td>
`base`
</td>
<td>
The base, or list of per-digit bases, to separate `val` into. When
a list of bases is specified, the last entry in the list is the
base for the last entry of the result (i.e. the least significant
digit). That is to say, the bases are also specified in big endian
order.
</td>
</tr><tr>
<td>
`digit_count`
</td>
<td>
The length of the desired result.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The list of digits.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
Unknown digit count. The `base` was specified as an integer and a
`digit_count` was not provided.
Inconsistent digit count. The `base` was specified as a per-digit
list, and `digit_count` was also provided, but they disagree.
</td>
</tr>
</table>



#### Examples:

>>> cirq.big_endian_int_to_digits(11, digit_count=4, base=10)
[0, 0, 1, 1]

```
>>> cirq.big_endian_int_to_digits(11, base=[2, 3, 4])
[0, 2, 3]
```
