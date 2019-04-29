import re
from typing import Optional

import ply.lex as lex


class QasmLexer(object):
    NATURAL_NUMBER = "NATURAL_NUMBER"

    def __init__(self, qasm: str):
        self.qasm = qasm
        self.lex = lex.lex(object=self, debug=True)
        self.lex.input(qasm)

    literals = "{}[]();,"

    reserved = {
        'qreg': 'QREG',
        'creg': 'CREG',
    }

    tokens = [
                 'FORMAT_SPEC',
                 'NATURAL_NUMBER',
                 'QELIBINC',
                 'ID',
             ] + list(reserved.values())

    def t_newline(self, t):
        r"""\n+"""
        t.lexer.lineno += len(t.value)

    t_ignore = ' \t'

    def t_NATURAL_NUMBER(self, t):
        r"""\d+"""
        t.value = int(t.value)
        return t

    def t_FORMAT_SPEC(self, t):
        r"""OPENQASM(\s+)([^\s\t\;]*);"""
        match = re.match(r"""OPENQASM(\s+)([^\s\t;]*);""", t.value)
        t.value = match.groups()[1]
        return t

    def t_QELIBINC(self, t):
        r"""include(\s+)"qelib1.inc";"""
        return t

    def t_QREG(self, t):
        r"""qreg"""
        return t

    def t_CREG(self, t):
        r"""creg"""
        return t

    def t_ID(self, t):
        r"""[a-zA-Z][a-zA-Z\d_]*"""
        return t

    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        # t.lexer.skip(1)

    def token(self) -> Optional[lex.Token]:
        return self.lex.token()
