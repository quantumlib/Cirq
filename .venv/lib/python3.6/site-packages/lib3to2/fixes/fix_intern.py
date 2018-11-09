"""
Fixer for sys.intern(s) -> intern(s).
"""

from lib2to3 import pytree
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, BlankLine, find_binding, find_root


class FixIntern(fixer_base.BaseFix):

    PATTERN = """
    power< 'sys' trailer < '.' 'intern' >
           trailer< lpar='('
                    ( not(arglist | argument<any '=' any>) obj=any
                      | obj=arglist<(not argument<any '=' any>) any ','> )
                    rpar=')' >
           after=any* >
    |
    import_from< 'from' 'sys' 'import'
                import_as_names< pre=any* binding='intern' post=any* > any* >
    |
    import_from< 'from' 'sys' 'import' simple='intern' >
    """

    def transform(self, node, results):
        name = results.get("name")
        binding = results.get("binding")
        pre = results.get("pre")
        post = results.get("post")
        simple = results.get("simple")
        if simple:
            binding = find_binding("intern", find_root(node), "sys")
            binding.remove()
            return
        if binding:
            if not pre and not post:
                new_binding = find_binding("intern", find_root(node), "sys")
                new_binding.remove()
                return
            elif not pre and post:
                for ch in node.children:
                    if type(ch) == pytree.Node:
                        assert ch.children[0].prefix + "intern" \
                                                       == str(ch.children[0])
                        ch.children[0].remove() # intern
                        assert ch.children[0].prefix + "," \
                                                       == str(ch.children[0])
                        ch.children[0].remove() # ,
                return
            elif not post and pre:
                for ch in node.children:
                    if type(ch) == pytree.Node:
                        assert ch.children[-1].prefix + "intern" \
                                                       == str(ch.children[-1])
                        ch.children[-1].remove() # intern
                        assert ch.children[-1].prefix + "," \
                                                       == str(ch.children[-1])
                        ch.children[-1].remove() # ,
                return
            elif post and pre:
                for ch in node.children:
                    if type(ch) == pytree.Node:
                        for ch_ in ch.children:
                            if ch_ and ch_.prefix + "intern" == str(ch_):
                                last_ch_ = ch_.prev_sibling
                                ch_.remove() # intern
                                assert last_ch_.prefix + "," \
                                                       == str(last_ch_)
                                last_ch_.remove() # ,
                return
        syms = self.syms
        obj = results["obj"].clone()
        if obj.type == syms.arglist:
            newarglist = obj.clone()
        else:
            newarglist = pytree.Node(syms.arglist, [obj.clone()])
        after = results["after"]
        if after:
            after = [n.clone() for n in after]

        new = pytree.Node(syms.power,
                          [Name("intern")] +
                          [pytree.Node(syms.trailer,
                                       [results["lpar"].clone(),
                                        newarglist,
                                        results["rpar"].clone()] + after)])
        new.prefix = node.prefix
        return new
