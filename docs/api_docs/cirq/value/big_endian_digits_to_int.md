<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.big_endian_digits_to_int" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.value.big_endian_digits_to_int

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/digits.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the big-endian integer specified by the given digits and base.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.big_endian_digits_to_int`, `cirq.value.digits.big_endian_digits_to_int`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.big_endian_digits_to_int(
    digits: Iterable[int],
    *,
    base: Union[int, Iterable[int]]
) -> int
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`digits`
</td>
<td>
Digits of the integer, with the least significant digit at the
end.
</td>
</tr><tr>
<td>
`base`
</td>
<td>
The base, or list of per-digit bases, to use when combining the
digits into an integer. When a list of bases is specified, the last
entry in the list is the base for the last entry of the digits list
(i.e. the least significant digit). That is to say, the bases are
also specified in big endian order.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The integer.
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
One of the digits is out of range for its base.
The base was specified per-digit (as a list) but the length of the
bases list is different from the number of digits.
</td>
</tr>
</table>



#### Examples:


```
>>> cirq.big_endian_digits_to_int([0, 1], base=10)
1
```

```
>>> cirq.big_endian_digits_to_int([1, 0], base=10)
10
```

```
>>> cirq.big_endian_digits_to_int([1, 2, 3], base=[2, 3, 4])
23
```
