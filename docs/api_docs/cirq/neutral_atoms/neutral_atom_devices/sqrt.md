<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.neutral_atoms.neutral_atom_devices.sqrt" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.neutral_atoms.neutral_atom_devices.sqrt

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



sqrt(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.pasqal.pasqal_device.sqrt`, `cirq.pasqal.pasqal_qubits.sqrt`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.neutral_atoms.neutral_atom_devices.sqrt()
</code></pre>



<!-- Placeholder for "Used in" -->

Return the non-negative square-root of an array, element-wise.

Parameters
----------
x : array_like
    The values whose square-roots are required.
out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
y : ndarray
    An array of the same shape as `x`, containing the positive
    square-root of each element in `x`.  If any element in `x` is
    complex, a complex array is returned (and the square-roots of
    negative reals are calculated).  If all of the elements in `x`
    are real, so is `y`, with negative elements returning ``nan``.
    If `out` was provided, `y` is a reference to it.
    This is a scalar if `x` is a scalar.

See Also
--------
lib.scimath.sqrt
    A version which returns complex numbers when given negative reals.

Notes
-----
*sqrt* has--consistent with common convention--as its branch cut the
real "interval" [`-inf`, 0), and is continuous from above on it.
A branch cut is a curve in the complex plane across which a given
complex function fails to be continuous.

Examples
--------
>>> np.sqrt([1,4,9])
array([ 1.,  2.,  3.])

```
>>> np.sqrt([4, -1, -3+4J])
array([ 2.+0.j,  0.+1.j,  1.+2.j])
```

```
>>> np.sqrt([4, -1, np.inf])
array([ 2., nan, inf])
```