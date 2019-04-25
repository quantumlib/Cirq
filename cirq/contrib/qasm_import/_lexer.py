from typing import List, Optional

import ply.lex as lex


class QasmLexer(object):

    def __init__(self, qasm: str):
        self.qasm = qasm
        self.lex = lex.lex(object=self, debug=True)
        self.lex.input(qasm)

    tokens = [
        'QASM20',
        'NUMBER'
    ]

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    t_ignore = ' \t'

    def t_NUMBER(self, t):
        r'\d+'
        t.value = int(t.value)
        return t

    def t_QASM20(self, t):
        r'OPENQASM\s 2.0;'
        return t

    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        # t.lexer.skip(1)

    def token(self) -> Optional[lex.Token]:
        return self.lex.token()
