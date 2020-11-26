'''
Domain Coloring
'''
from PIL import Image
from math import pi, e, sin, cos, log
from cmath import sin, cos, log, acos
import numpy as np
from colorsys import hsv_to_rgb
from timeit import default_timer as time
from numba import njit, jit, prange

predefFuncs = ['acos((1+1j)*log(sin(z**3-1)/z))', '(1+1j)*log(sin(z**3-1)/z)', '(1+1j)*sin(z)',
               'z + z**2/sin(z**4-1)', 'log(sin(z))', 'cos(z)/(sin(z**4-1)', 'z**6-1',
               '(z**2-1) * (z-2-1j)**2 / (z**2+2*1j)', 'sin(z)*(1+2j)', 'sin(1/z)', 'sin(z)*sin(1/z)',
               '1/sin(1/sin(z))', 'z', '(z**2+1)/(z**2-1)', '(z**2+1)/z', '(z+3)*(z+1)**2',
               '(z/2)**2*(z+1-2j)*(z+2+2j)/z**3', '(z**2)-0.75-(0.2*j)']


class DomainColoring:

    def generateColors(self, h, w):
        def zExpression(z):
            return z ** 6 - 1

        def pow3(x):
            return x * x * x

        def color3f_2_rgbaint(c):
            return 0xff00_0000 | (int(c[0] * 255.) << 16) | (int(c[1] * 255.) << 8) | int(c[2] * 255.)

        pi2 = pi * 2
        limit = pi

        rmi, rma, imi, ima = -limit, limit, -limit, limit

        colors = np.zeros((w, h), dtype=np.int32)

        for j in range(h):
            im = ima - (ima - imi) * j / (h - 1)
            for i in range(w):
                re = rma - (rma - rmi) * i / (w - 1)

                v = zExpression(complex(re, im))

                hue = v.real
                while hue < 0: hue += pi2
                hue /= pi2
                m, _ranges, _rangee = abs(v), 0, 1
                while m > _rangee:
                    _ranges = _rangee
                    _rangee *= e

                k = (m - _ranges) / (_rangee - _ranges)
                kk = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2
                sat = 0.4 + (1 - pow3((1 - kk))) * 0.6
                val = 0.6 + (1 - pow3((1 - (1 - kk)))) * 0.4

                colors[i, j] = color3f_2_rgbaint(hsv_to_rgb(hue, sat, val))
        return colors


class DomainColoring_jit:

    @njit(fastmath=True)
    def generateColors(h, w):
        def _zExpression(z):
            return z ** 6 - 1

        def hsv_to_rgb(h, s, v):
            def c2i(r, g, b):
                return 0xff00_0000 | (int(r * 255.) << 16) | (int(g * 255.) << 8) | int(b * 255.)

            if s == 0.0:
                return c2i(v, v, v)
            i = int(h * 6.0)  # XXX assume int() truncates!
            f = (h * 6.0) - i
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))
            i = i % 6
            if i == 0:
                return c2i(v, t, p)
            if i == 1:
                return c2i(q, v, p)
            if i == 2:
                return c2i(p, v, t)
            if i == 3:
                return c2i(p, q, v)
            if i == 4:
                return c2i(t, p, v)
            if i == 5:
                return c2i(v, p, q)
            else:
                return c2i(0, 0, 0)

        def pow3(x):
            return x * x * x

        pi2 = pi * 2
        limit = pi

        rmi, rma, imi, ima = -limit, limit, -limit, limit

        colors = np.zeros((w, h), dtype=np.int32)

        for j in range(h):
            im = ima - (ima - imi) * j / (h - 1)
            for i in range(w):
                re = rma - (rma - rmi) * i / (w - 1)

                v = _zExpression(complex(re, im))

                hue = v.real
                while hue < 0: hue += pi2
                hue /= pi2
                m, _ranges, _rangee = abs(v), 0, 1
                while m > _rangee:
                    _ranges = _rangee
                    _rangee *= e

                k = (m - _ranges) / (_rangee - _ranges)
                kk = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2
                sat = 0.4 + (1 - pow3(1 - kk)) * 0.6
                val = 0.6 + (1 - pow3(kk)) * 0.4

                colors[i, j] = hsv_to_rgb(hue, sat, val)
        return colors


class DomainColoring_paralell:

    @njit(parallel=True, fastmath=True, cache=True)
    def generateColors(h, w):

        zExpression = lambda z: z * sin(z) / cos(z)

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

                v = zExpression(complex(re, im))

                hue = v.real
                while hue < 0: hue += pi2
                hue /= pi2

                m, _ranges, _rangee = abs(v), 0, 1
                while m > _rangee:
                    _ranges = _rangee
                    _rangee *= e

                k = (m - _ranges) / (_rangee - _ranges)
                kk = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2

                sat = 0.4 + (1 - pow3(1 - kk)) * 0.6
                val = 0.6 + (1 - pow3(kk)) * 0.4

                colors[i, j] = hsv_to_rgb(hue, sat, val)
        return colors


class DomainColoring_paralell_jit_expression:

    def z_compiler(fz): # must be used externally and generated func as parameter of 'generateColors'
        ns = {}
        exec(f'''
from cmath import sin, cos, tan, asin, acos, atan, log, exp
def __z_expr(z):
    def c(re, im): return complex(re, im)
    return {fz}
    ''', ns, ns)

        return jit(ns['__z_expr'])

    @njit(parallel=True, fastmath=True)
    def generateColors(zExpression, h, w):

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

                v = zExpression(complex(re, im))

                hue = v.real
                while hue < 0: hue += pi2
                hue /= pi2

                m, _ranges, _rangee = abs(v), 0, 1
                while m > _rangee:
                    _ranges = _rangee
                    _rangee *= e

                k = (m - _ranges) / (_rangee - _ranges)
                kk = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2

                sat = 0.4 + (1 - pow3(1 - kk)) * 0.6
                val = 0.6 + (1 - pow3(kk)) * 0.4

                colors[i, j] = hsv_to_rgb(hue, sat, val)
        return colors


def test_dc(n):
    dc = DomainColoring()
    t0 = time()
    c = dc.generateColors(n, n)
    t0 = time() - t0

    print(f'lap pure python:{t0}')
    Image.frombytes(mode='RGBA', size=(n, n), data=c).show()  # .save('voronoi.png', format='png')


def test_dc_jit(n):
    DomainColoring_jit.generateColors(2, 2)  # warm up

    t0 = time()
    c = DomainColoring_jit.generateColors(n, n)
    t0 = time() - t0

    print(f'lap numba jit:{t0}')
    Image.frombytes(mode='RGBA', size=(n, n), data=c).show()  # .save('voronoi.png', format='png')


def test_dc_paralell(n):
    DomainColoring_paralell.generateColors(2, 2)  # warm up

    t0 = time()
    img = DomainColoring_paralell.generateColors(n, n)
    t0 = time() - t0

    print(f'lap for {n} x {n} items, numba parallel:{t0}')
    Image.frombytes(mode='RGBA', size=(n, n), data=img).show()  # .save('voronoi.png', format='png')


def test_dc_paralell_jit_expression(n):
    predef_funcs = ['acos(c(1,2)*log(sin(z**3-1)/z))', 'c(1,1)*log(sin(z**3-1)/z)', 'c(1,1)*sin(z)',
                    'z + z**2/sin(z**4-1)', 'log(sin(z))', 'cos(z)/(sin(z**4-1))', 'z**6-1',
                    '(z**2-1) * (z-c(2,1))**2 / (z**2+c(2,1))', 'sin(z)*c(1,2)', 'sin(1/z)', 'sin(z)*sin(1/z)',
                    '1/sin(1/sin(z))', 'z', '(z**2+1)/(z**2-1)', '(z**2+1)/z', '(z+3)*(z+1)**2',
                    '(z/2)**2*(z+c(1,2))*(z+c(2,2))/z**3', '(z**2)-0.75-c(0,0.2)']

    fx=predef_funcs[5]
    z_expr = DomainColoring_paralell_jit_expression.z_compiler(fx)
    DomainColoring_paralell_jit_expression.generateColors(z_expr, 2, 2)  # warm up

    t0 = time()
    img = DomainColoring_paralell_jit_expression.generateColors(z_expr, n, n)
    t0 = time() - t0

    print(f'domain coloring: {fx}, lap for {n} x {n} items, numba parallel jit expression:{t0}')
    Image.frombytes(mode='RGBA', size=(n, n), data=img).show()  # .save('voronoi.png', format='png')


if __name__ == '__main__':
    n = 1024 * 3

    # test_dc(n)
    # test_dc_jit(n)
    # test_dc_paralell(n)
    test_dc_paralell_jit_expression(n)
