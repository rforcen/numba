'''
Domain Coloring using dynamic llvm compiled z expression
'''
from math import pi, e, isfinite
from timeit import default_timer as time

import numpy as np
from PIL import Image
from numba import njit, prange

from z_parser import ZExpression


class DomainColoring:  # jit, parallel and expression evaluator
    @staticmethod
    def z_compiler(fz):  # must be used externally and generated func as parameter of 'generate'
        ns = {}
        exec(f'''
    from cmath import sin, cos, tan, asin, acos, atan, log, exp
    def __z_expr(z):
        def c(re, im): return complex(re, im)
        return {fz}
        ''', ns, ns)

        return njit(ns['__z_expr'])

    @staticmethod
    def generate_llvm_exec(fz, h, w):
        return DomainColoring._generate_z_compiled(DomainColoring.z_compiler(fz), w, h)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _generate_z_compiled_llvm_exec(zExpression, h, w):
        pass

    @staticmethod
    def generate(fz, h, w):
        zeval, code, const_tab = ZExpression().set_fz(fz)
        return DomainColoring._generate_z_compiled(zeval, code, const_tab, w, h)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _generate_z_compiled(zeval, code, const_tab, h, w):

        pow3 = lambda x: x * x * x
        rgb2i = lambda r, g, b: 0xff00_0000 | (int(r * 255.) << 16) | (int(g * 255.) << 8) | int(b * 255.)

        def hsv_to_rgb(h, s, v):
            if s == 0:
                return rgb2i(v, v, v)

            i, f = divmod(h * 6, 1)

            p = v * (1 - s)
            q = v * (1 - s * f)
            t = v * (1 - s + s * f)

            i %= 6

            if i == 0:
                return rgb2i(v, t, p)
            if i == 1:
                return rgb2i(q, v, p)
            if i == 2:
                return rgb2i(p, v, t)
            if i == 3:
                return rgb2i(p, q, v)
            if i == 4:
                return rgb2i(t, p, v)
            if i == 5:
                return rgb2i(v, p, q)
            else:
                return rgb2i(0, 0, 0)

        pi2 = pi * 2
        limit = pi

        rmi, rma, imi, ima = -limit, limit, -limit, limit

        colors = np.empty((w, h), dtype=np.int32)

        for j in prange(h):
            im = ima - (ima - imi) * j / h

            for i in range(w):
                re = rma - (rma - rmi) * i / w

                try:
                    v = zeval(complex(re, im), code, const_tab)
                except:
                    continue

                if isfinite(m := abs(v)):
                    _ranges, _rangee = 0, 1  # fast way to calc exp(n), exp(n+1) -> n=log(m)
                    while m > _rangee:
                        _ranges = _rangee
                        _rangee *= e

                    k = (m - _ranges) / (_rangee - _ranges)
                    kk = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2

                    hue = v.real
                    hue = (hue / pi2 - 1 - int(hue / pi2))  # while hue < 0: hue += pi2; hue/=pi2
                    sat = 0.4 + (1 - pow3(1 - kk)) * 0.6
                    val = 0.6 + (1 - pow3(kk)) * 0.4

                    colors[i, j] = hsv_to_rgb(hue, sat, val)
                else:
                    colors[i, j] = 0xff00_0000

        return colors


if __name__ == '__main__':
    def test_dc(n):
        predefFuncs = ['acos(c(1,2)*log(sin(z**3-1)/z))', 'c(1,1)*log(sin(z**3-1)/z)', 'c(1,1)*sin(z)',
                       'z + z**2/sin(z**4-1)', 'log(sin(z))', 'cos(z)/(sin(z**4-1))', 'z**6-1',
                       '(z**2-1) * (z-c(2,1))**2 / (z**2+c(2,1))', 'sin(z)*c(1,2)', 'sin(1/z)', 'sin(z)*sin(1/z)',
                       '1/sin(1/sin(z))', 'z', '(z**2+1)/(z**2-1)', '(z**2+1)/z', '(z+3)*(z+1)**2',
                       '(z/2)**2*(z+c(1,2))*(z+c(2,2))/z**3', '(z**2)-0.75-c(0,0.2)']

        for i, fz in enumerate(predefFuncs):
            t0 = time()
            img = DomainColoring.generate(fz, n, n)
            t0 = time() - t0

            print(f'domain coloring: {fz}, lap for {n} x {n} items, numba parallel njit expression:{t0}')
            Image.frombytes(mode='RGBA', size=(n, n), data=img).show(f'dc{i}.png')  # save(f'dc{i}.png', format='png')


    test_dc(1024)
