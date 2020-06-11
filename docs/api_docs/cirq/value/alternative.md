<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.alternative" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.value.alternative

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/abc_alt.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A decorator indicating an abstract method with an alternative default

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.alternative`, `cirq.value.abc_alt.alternative`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.alternative(
    *,
    requires: str,
    implementation: <a href="../../cirq/value/abc_alt/T.md"><code>cirq.value.abc_alt.T</code></a>
) -> Callable[[T], T]
</code></pre>



<!-- Placeholder for "Used in" -->
implementation.

This decorator may be used multiple times on the same function to specify
multiple alternatives.  If multiple alternatives are available, the
outermost (lowest line number) alternative is used.

#### Usage:

class Parent(metaclass=ABCMetaImplementAnyOneOf):
    def _default_do_a_using_b(self, ...):
        ...
    def _default_do_a_using_c(self, ...):
        ...

    # Abstract method with alternatives
    @alternative(requires='do_b', implementation=_default_do_a_using_b)
    @alternative(requires='do_c', implementation=_default_do_a_using_c)
    def do_a(self, ...):
        '''Method docstring.'''

    # Abstract or concrete methods `do_b` and `do_c`:
    ...

class Child(Parent):
    def do_b(self):
        ...

child = Child()
child.do_a(...)



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`requires`
</td>
<td>
The name of another abstract method in the same class that
`implementation` needs to be implemented.
</td>
</tr><tr>
<td>
`implementation`
</td>
<td>
A function that uses the method named by requires to
implement the default behavior of the wrapped abstract method.  This
function must have the same signature as the decorated function.
</td>
</tr>
</table>

