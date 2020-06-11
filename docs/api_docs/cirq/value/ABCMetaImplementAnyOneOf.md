<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.ABCMetaImplementAnyOneOf" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="register"/>
</div>

# cirq.value.ABCMetaImplementAnyOneOf

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/abc_alt.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A metaclass extending `abc.ABCMeta` for defining abstract base classes

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ABCMetaImplementAnyOneOf`, `cirq.value.abc_alt.ABCMetaImplementAnyOneOf`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.ABCMetaImplementAnyOneOf(
    mcls, name, bases, namespace, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->
(ABCs) with more flexibility in which methods must be overridden.

Use this metaclass in the same way as `abc.ABCMeta` to create an ABC.

In addition to the decorators in the` abc` module, the decorator
`@alternative(...)` may be used.

## Methods

<h3 id="register"><code>register</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>register(
    subclass
)
</code></pre>

Register a virtual subclass of an ABC.

Returns the subclass, to allow usage as a class decorator.



