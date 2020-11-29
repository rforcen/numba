'''
z expression compiler

ZExpression class usage
zx=ZExpression(zexpression_string)
zx.eval(z)
'''
from cmath import sin, cos, tan, asin, acos, atan, log, exp
from timeit import default_timer as timer

import numba
import numpy as np
from lark import Lark, Transformer, v_args

z_grammar = """
    ?start: sum

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: power
        | product "*" power  -> mul
        | product "/" power  -> div
    
    ?power: atom
        | power "**" atom     -> pow 

    ?atom: NUMBER            -> number
         | "-" atom          -> neg
         | "z"               -> z
         | "c"   "(" NUMBER "," NUMBER ")"  -> complex
         | funcs "(" sum ")"          -> func
         | "(" sum ")"
         
    funcs : FUNCS
    FUNCS: "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "log" | "exp"
    
    %import common.NUMBER
    %import common.WS_INLINE
    %ignore " "

    %ignore WS_INLINE
"""

# tokens
SADD, SSUB, SMUL, SDIV, SPOW, SPUSHC, SNEG, SPUSHZ, SCOMPLEX, SSIN, SCOS, STAN, SASIN, SACOS, SATAN, SLOG, SEXP = range(
    17)


@v_args(inline=True)  # print nmemotecnic p-code
class Tree_StackPcode(Transformer):

    def __init__(self):
        pass

    def add(self, v0, v1):
        print('add')

    def sub(self, v0, v1):
        print('sub')

    def mul(self, v0, v1):
        print('mul')

    def div(self, v0, v1):
        print('div')

    def pow(self, v0, v1):
        print('pow')

    def number(self, v0):
        print(f'pushc {v0.value}')

    def neg(self, v0): print('neg')

    def z(self): print('pushz')

    def complex(self, v0, v1): print('complex')

    def func(self, v0, v1): print(v0.children[0].value)


@v_args(inline=True)  # evaluate expression
class Tree_StackInterpret(Transformer):
    stack = []
    _z = complex(0, 0)

    def __init__(self):
        pass

    def setz(self, z):
        self._z = z

    def get_result(self):
        return self.stack[-1]

    def add(self, v0, v1):
        z1 = self.stack.pop()
        z0 = self.stack.pop()
        self.stack.append(z0 + z1)

    def sub(self, v0, v1):
        z1 = self.stack.pop()
        z0 = self.stack.pop()
        self.stack.append(z0 - z1)

    def mul(self, v0, v1):
        z1 = self.stack.pop()
        z0 = self.stack.pop()
        self.stack.append(z0 * z1)

    def div(self, v0, v1):
        z1 = self.stack.pop()
        z0 = self.stack.pop()
        self.stack.append(z0 / z1)

    def pow(self, v0, v1):
        z1 = self.stack.pop()
        z0 = self.stack.pop()
        self.stack.append(z0 ** z1)

    def number(self, v0):
        self.stack.append(float(v0.value))

    def neg(self, v0):
        self.stack[-1] = -self.stack[-1]

    def z(self):
        self.stack.append(self._z)

    def complex(self, v0, v1):
        self.stack.append(complex(float(v0), float(v1)))

    def func(self, v0, v1):
        # print(v0.children[0].value)
        f = v0.children[0].value
        if f == 'sin':
            self.stack[-1] = sin(self.stack[-1])
        elif f == 'cos':
            self.stack[-1] = cos(self.stack[-1])
        elif f == 'tan':
            self.stack[-1] = tan(self.stack[-1])
        if f == 'asin':
            self.stack[-1] = asin(self.stack[-1])
        elif f == 'acos':
            self.stack[-1] = acos(self.stack[-1])
        elif f == 'atan':
            self.stack[-1] = atan(self.stack[-1])
        elif f == 'log':
            self.stack[-1] = log(self.stack[-1])
        elif f == 'exp':
            self.stack[-1] = exp(self.stack[-1])


@v_args(inline=True)  # generate 'code' and constant table
class Tree_pCodeGenerator(Transformer):
    func_dict = {'sin': SSIN, 'cos': SCOS, 'tan': STAN, 'asin': SASIN, 'acos': SACOS, 'atan': SATAN, 'log': SLOG,
                 'exp': SEXP}
    sym_dict = {SADD: '+', SSUB: '-', SMUL: '*', SDIV: '/', SPOW: '**', SPUSHC: 'pushc', SNEG: 'neg', SPUSHZ: 'pushz',
                SCOMPLEX: 'complex', SSIN: 'sin', SCOS: 'cos', STAN: 'tan', SASIN: 'asin', SACOS: 'acos', SATAN: 'atan',
                SLOG: 'log', SEXP: 'exp'}

    def __init__(self):
        self.reset()

    def reset(self):
        self.code = np.empty(0, np.int8)
        self.const_tab = np.empty(0, np.complex)

    def gen(self, instr):
        self.code = np.append(self.code, instr)

    def insert_const(self, z_const):
        self.const_tab = np.append(self.const_tab, z_const)

    def add(self, v0, v1):
        self.gen(SADD)

    def sub(self, v0, v1):
        self.gen(SSUB)

    def mul(self, v0, v1):
        self.gen(SMUL)

    def div(self, v0, v1):
        self.gen(SDIV)

    def pow(self, v0, v1):
        self.gen(SPOW)

    def number(self, v0):
        self.gen(SPUSHC)
        self.gen(len(self.const_tab))

        self.insert_const(complex(float(v0), 0))

    def neg(self, v0):
        self.gen(SNEG)

    def z(self):
        self.gen(SPUSHZ)

    def complex(self, v0, v1):
        self.gen(SPUSHC)
        self.gen(len(self.const_tab))

        self.insert_const(complex(float(v0), float(v1)))

    def func(self, v0, v1):
        f = v0.children[0].value
        self.gen(self.func_dict[f])

    def print(self):
        print('code:', end='')
        i = 0
        while i < len(self.code):
            c = self.code[i]
            if c == SPUSHC:
                print(self.sym_dict[c], '[', self.code[i + 1], ']', self.const_tab[self.code[i + 1]], end='| ')
                i += 1
            else:
                print(self.sym_dict[c], end='| ')
            i += 1
        print('consts:', end='')
        for t in self.const_tab:
            print(t, end=', ')
        print()

    def execute_pcode(self, z):
        def pop2():
            return stack.pop(), stack.pop()

        pc = 0
        stack = []

        while pc < len(self.code):
            c = self.code[pc]
            if c == SPUSHC:
                stack.append(self.const_tab[self.code[pc + 1]])
                pc += 1
            elif c == SPUSHZ:
                stack.append(z)
            elif c == SADD:
                z1, z0 = pop2()
                stack.append(z0 + z1)
            elif c == SSUB:
                z1, z0 = pop2()
                stack.append(z0 - z1)
            elif c == SMUL:
                z1, z0 = pop2()
                stack.append(z0 * z1)
            elif c == SDIV:
                z1, z0 = pop2()
                stack.append(z0 / z1)
            elif c == SPOW:
                z1, z0 = pop2()
                stack.append(z0 ** z1)
            elif c == SNEG:
                stack[-1] = -stack[-1]
            elif c == SSIN:
                stack[-1] = sin(stack[-1])
            elif c == SCOS:
                stack[-1] = cos(stack[-1])
            elif c == STAN:
                stack[-1] = tan(stack[-1])
            if c == SASIN:
                stack[-1] = asin(stack[-1])
            elif c == SACOS:
                stack[-1] = acos(stack[-1])
            elif c == SATAN:
                stack[-1] = atan(stack[-1])
            elif c == SLOG:
                stack[-1] = log(stack[-1])
            elif c == SEXP:
                stack[-1] = exp(stack[-1])

            pc += 1
        return stack[0]


@numba.njit(fastmath=True, cache=True)
def execute_pcode(z, code, const_tab):
    stack = np.empty(1024, dtype=np.complex64)
    pc: int = 0
    sp: int = 0
    zero = 0 + 0j

    while pc < len(code):
        c = code[pc]
        if c == SPUSHC:
            stack[sp] = const_tab[code[pc + 1]]
            sp += 1
            pc += 1
        elif c == SPUSHZ:
            stack[sp] = z
            sp += 1
        elif c == SADD:
            sp -= 2
            stack[sp] += stack[sp + 1]
            sp += 1
        elif c == SSUB:
            sp -= 2
            stack[sp] -= stack[sp + 1]
            sp += 1
        elif c == SMUL:
            sp -= 2
            stack[sp] *= stack[sp + 1]
            sp += 1
        elif c == SDIV:
            sp -= 2
            stack[sp] = stack[sp] / stack[sp + 1] if stack[sp + 1] != zero else zero
            sp += 1
        elif c == SPOW:
            sp -= 2
            stack[sp] = stack[sp] ** stack[sp + 1]
            sp += 1
        elif c == SNEG:
            stack[sp - 1] = -stack[sp - 1]
        elif c == SSIN:
            stack[sp - 1] = sin(stack[sp - 1])
        elif c == SCOS:
            stack[sp - 1] = cos(stack[sp - 1])
        elif c == STAN:
            stack[sp - 1] = tan(stack[sp - 1])
        if c == SASIN:
            stack[sp - 1] = asin(stack[sp - 1])
        elif c == SACOS:
            stack[sp - 1] = acos(stack[sp - 1])
        elif c == SATAN:
            stack[sp - 1] = atan(stack[sp - 1])
        elif c == SLOG:
            stack[sp - 1] = log(stack[sp - 1]) if stack[sp - 1] != zero else zero
        elif c == SEXP:
            stack[sp - 1] = exp(stack[sp - 1])

        pc += 1
    return stack[0]


'''
ZExpression usage
zx=ZExpression(zexpression_string)
zx.eval(z)
'''


class ZExpression:
    def __init__(self, fx=None):
        self.tr = Tree_pCodeGenerator()
        self.z_parser = Lark(z_grammar, parser='lalr', transformer=self.tr)
        self.parser = self.z_parser.parse
        if fx is not None:
            self.compile(fx)

    @numba.njit(fastmath=True, cache=True)
    def execute_pcode(z, code, const_tab):
        stack = np.empty(1024, dtype=np.complex64)
        pc: int = 0
        sp: int = 0
        zero = 0 + 0j

        while pc < len(code):
            c = code[pc]
            if c == SPUSHC:
                stack[sp] = const_tab[code[pc + 1]]
                sp += 1
                pc += 1
            elif c == SPUSHZ:
                stack[sp] = z
                sp += 1
            elif c == SADD:
                sp -= 2
                stack[sp] += stack[sp + 1]
                sp += 1
            elif c == SSUB:
                sp -= 2
                stack[sp] -= stack[sp + 1]
                sp += 1
            elif c == SMUL:
                sp -= 2
                stack[sp] *= stack[sp + 1]
                sp += 1
            elif c == SDIV:
                sp -= 2
                stack[sp] = stack[sp] / stack[sp + 1] if stack[sp + 1] != zero else zero
                sp += 1
            elif c == SPOW:
                sp -= 2
                stack[sp] = stack[sp] ** stack[sp + 1]
                sp += 1
            elif c == SNEG:
                stack[sp - 1] = -stack[sp - 1]
            elif c == SSIN:
                stack[sp - 1] = sin(stack[sp - 1])
            elif c == SCOS:
                stack[sp - 1] = cos(stack[sp - 1])
            elif c == STAN:
                stack[sp - 1] = tan(stack[sp - 1])
            if c == SASIN:
                stack[sp - 1] = asin(stack[sp - 1])
            elif c == SACOS:
                stack[sp - 1] = acos(stack[sp - 1])
            elif c == SATAN:
                stack[sp - 1] = atan(stack[sp - 1])
            elif c == SLOG:
                stack[sp - 1] = log(stack[sp - 1]) if stack[sp - 1] != zero else zero
            elif c == SEXP:
                stack[sp - 1] = exp(stack[sp - 1])

            pc += 1
        return stack[0]

    def compile(self, fx):
        self.tree = self.parser(fx)

    def eval(self, z):
        return execute_pcode(z=z, code=self.tr.code, const_tab=self.tr.const_tab)


if __name__ == '__main__':
    def c(re, im):
        return complex(re, im)


    predefFuncs = ['acos(c(1,2)*log(sin(z**3-1)/z))', 'c(1,1)*log(sin(z**3-1)/z)', 'c(1,1)*sin(z)',
                   'z + z**2/sin(z**4-1)', 'log(sin(z))', 'cos(z)/(sin(z**4-1))', 'z**6-1',
                   '(z**2-1) * (z-c(2,1))**2 / (z**2+c(2,1))', 'sin(z)*c(1,2)', 'sin(1/z)', 'sin(z)*sin(1/z)',
                   '1/sin(1/sin(z))', 'z', '(z**2+1)/(z**2-1)', '(z**2+1)/z', '(z+3)*(z+1)**2',
                   '(z/2)**2*(z+c(1,2))*(z+c(2,2))/z**3', '(z**2)-0.75-c(0,0.2)']

    predef_funcs = ['acos(1+2j*log(sin(z**3-1)/z))', '1+1j*log(sin(z**3-1)/z)', '1+1j*sin(z)',
                    'z + z**2/sin(z**4-1)', 'log(sin(z))', 'cos(z)/(sin(z**4-1))', 'z**6-1',
                    '(z**2-1) * (z-2+1j)**2 / (z**2+(2+1j))', 'sin(z)*1+2j', 'sin(1/z)', 'sin(z)*sin(1/z)',
                    '1/sin(1/sin(z))', 'z', '(z**2+1)/(z**2-1)', '(z**2+1)/z', '(z+3)*(z+1)**2',
                    '(z/2)**2*(z+1+2j)*(z+2+2j)/z**3', '(z**2)-0.75-0.2j']


    def test_interpreter():
        tr = Tree_pCodeGenerator()  # Tree_StackInterpret()
        z_parser = Lark(z_grammar, parser='lalr', transformer=tr)
        z_calc = z_parser.parse

        global z_res
        # = complex(0, 0)
        for f in predefFuncs:
            print('-' * 10, '> ', f)
            tr.setz(z := complex(1, 1))
            tree = z_calc(f)
            print(tr.get_result() == eval(f))


    def test_gen():
        tr = Tree_pCodeGenerator()
        z_parser = Lark(z_grammar, parser='lalr', transformer=tr)
        z_gen = z_parser.parse

        z = 1 + 1j
        execute_pcode(z=z, code=tr.code, const_tab=tr.const_tab)  # warm up

        tot_t = 0.
        n = int(5e4)
        for f in predefFuncs:
            print('-' * 10, '> ', f)
            tr.reset()
            tree = z_gen(f)
            # tr.print()
            t = timer()

            for i in range(n):
                zr = execute_pcode(z=z, code=tr.code, const_tab=tr.const_tab)
            tot_t += timer() - t
            # print(eval(f), zr)
        print(f'total time: {tot_t}')


    def test_expr():
        n = int(5e4)
        z = 1 + 1j
        tot_t = 0.

        print(f'running zexpression on {n} loops...')

        for f in predefFuncs:
            zx = ZExpression(f)
            t = timer()

            for _ in range(n):
                zr = zx.eval(z)

            tot_t += timer() - t
        print(f'total time: {tot_t}')


    # test_gen()
    test_expr()