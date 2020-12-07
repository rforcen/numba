'''
z expression compiler

ZExpression class usage
zx=ZExpression(zexpression_string)
zx.eval(z)
'''
from cmath import isfinite, sin, cos, tan, asin, acos, atan, log, exp
from timeit import default_timer as timer

import numpy as np
from lark import Lark, Transformer, v_args
from numba import njit, prange, int32, int8, complex64

z_grammar = """
    ?start: sum -> expression

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
SPUSHC, SPUSHZ, SCOMPLEX, SADD, SSUB, SMUL, SDIV, SPOW, SNEG, SSIN, SCOS, STAN, SASIN, SACOS, SATAN, SLOG, SEXP, SEND = range(
    18)


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

    def expression(self, v0): print('end')


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

    MAX_CODE = 1024 * 10
    MAX_TAB = 1024

    def __init__(self):
        self.reset()

    def reset(self):
        self.code = np.empty(self.MAX_CODE, np.int8)
        self.const_tab = np.empty(self.MAX_TAB, np.complex)
        self.pc = 0
        self.itab = 0

    def gen(self, instr):
        self.code[self.pc] = instr
        self.pc += 1

    def insert_const(self, z_const):
        self.const_tab[self.itab] = z_const
        self.itab += 1

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
        self.gen(self.itab)

        self.insert_const(complex(float(v0), 0))

    def neg(self, v0):
        self.gen(SNEG)

    def z(self):
        self.gen(SPUSHZ)

    def complex(self, v0, v1):
        self.gen(SPUSHC)
        self.gen(self.itab)

        self.insert_const(complex(float(v0), float(v1)))

    def expression(self, v0):
        self.gen(SEND)

    def finish(self):
        self.code = np.resize(self.code, self.pc)
        self.const_tab = np.resize(self.const_tab, self.itab)

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

        while pc < self.pc:
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
            elif c == SEND:
                break

            pc += 1
        return stack[0]


'''
ZExpression usage
zx=ZExpression(zexpression_string)
zx.eval(z)
'''


class ZExpression:

    def __init__(_, fz='z'):
        _.tr, _.z_parser, _.parser = None, None, None
        _.set_fz(fz)

    def set_fz(_, fz):
        _.tr = Tree_pCodeGenerator()
        _.z_parser = Lark(z_grammar, parser='lalr', transformer=_.tr)
        _.parser = _.z_parser.parse

        if fz is not None:
            _.compile(fz)
        return _.get_exec()

    def get_exec(_):
        return ZExpression.execute_pcode, _.tr.code, _.tr.const_tab

    @staticmethod
    @njit(fastmath=True, cache=True)
    def execute_pcode(z, code, const_tab):
        MAX_STACK: int32 = 1024
        stack = np.empty(MAX_STACK, dtype=complex64)

        sp: int32 = 0
        pc: int32 = 0
        cc: int8 = code[pc]
        zero: complex64 = 0 + 0j

        while cc != SEND:
            if cc == SPUSHC:
                stack[sp] = const_tab[code[pc + 1]]
                sp += 1
                pc += 1
            elif cc == SPUSHZ:
                stack[sp] = z
                sp += 1
            elif cc == SADD:
                sp -= 2
                stack[sp] += stack[sp + 1]
                sp += 1
            elif cc == SSUB:
                sp -= 2
                stack[sp] -= stack[sp + 1]
                sp += 1
            elif cc == SMUL:
                sp -= 2
                stack[sp] *= stack[sp + 1]
                sp += 1
            elif cc == SDIV:
                sp -= 2
                stack[sp] = stack[sp] / stack[sp + 1] if stack[sp + 1] != zero and isfinite(stack[sp + 1]) and isfinite(
                    stack[sp]) else zero
                sp += 1

            elif cc == SPOW:
                sp -= 2
                stack[sp] = stack[sp] ** stack[sp + 1]
                sp += 1

            elif cc == SNEG:
                stack[sp - 1] = -stack[sp - 1]
            elif cc == SSIN:
                stack[sp - 1] = sin(stack[sp - 1])
            elif cc == SCOS:
                stack[sp - 1] = cos(stack[sp - 1])
            elif cc == STAN:
                stack[sp - 1] = tan(stack[sp - 1])
            elif cc == SASIN:
                stack[sp - 1] = asin(stack[sp - 1])
            elif cc == SACOS:
                stack[sp - 1] = acos(stack[sp - 1])
            elif cc == SATAN:
                stack[sp - 1] = atan(stack[sp - 1])
            elif cc == SLOG:
                stack[sp - 1] = log(stack[sp - 1]) if stack[sp - 1] != zero else zero
            elif cc == SEXP:
                stack[sp - 1] = exp(stack[sp - 1])

            pc += 1
            cc = code[pc]

        return stack[0]

    def compile(_, fx):
        _.tree = _.parser(fx)
        _.tr.finish()
        _.eval()  # warm up

    def eval(self, z=1 + 1j):
        return ZExpression.execute_pcode(z=z, code=self.tr.code, const_tab=self.tr.const_tab)


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

        tot_t = 0.
        n = int(5e4)
        for f in predefFuncs:
            print('-' * 10, '> ', f)
            tr.reset()
            tree = z_gen(f)
            # tr.print()
            t = timer()

            for i in range(n):
                zr = ZExpression.execute_pcode(z=z, code=tr.code, const_tab=tr.const_tab)
            tot_t += timer() - t
            # print(eval(f), zr)
        print(f'total time: {tot_t}')


    # @numba.njit(parallel=True, fastmath=True)  # eval code in njit using zeval evaluator
    def njit_eval_sum(zeval, code, const_tab, z, n):
        # print(zexec, code, const_tab)

        s = 0.0
        for i in prange(n):  # do parallel sum
            s += zeval(z, code, const_tab)

        return s


    def test_expr():
        n = int(1e4)
        z = 1 + 1j
        tot_t = 0.

        print(f'running zexpression on {n} loops...')
        zx = ZExpression()

        for f in predefFuncs:
            zeval, code, const_tab = zx.set_fz(f)
            # print(zexec, code, const_tab)

            t = timer()

            zrs = njit_eval_sum(zeval, code, const_tab, z, n)

            tot_t += timer() - t

            print(f'{f:50} {zrs}')
        print(f'total time: {tot_t}')


    # test_gen()
    test_expr()
