'''
mandelbrot/julia  fractal w/numba
'''

from timeit import default_timer as time

import numpy as np
from PIL import Image
from numba import njit, prange, int32, float32, complex64

fire_palette = np.array((0, 0, 4, 12, 16, 24, 32, 36, 44, 48, 56, 64, 68, 76, 80, 88, 96, 100, 108, 116, 120, 128, 132,
                         140, 148, 152, 160, 164, 172, 180, 184, 192, 200, 1224, 3272, 4300, 6348, 7376, 9424, 10448,
                         12500, 14548, 15576, 17624, 18648, 20700, 21724, 23776, 25824, 26848, 28900, 29924, 31976,
                         33000, 35048, 36076, 38124, 40176, 41200, 43248, 44276, 46324, 47352, 49400, 51452, 313596,
                         837884, 1363196, 1887484, 2412796, 2937084, 3461372, 3986684, 4510972, 5036284, 5560572,
                         6084860, 6610172, 7134460, 7659772, 8184060, 8708348, 9233660, 9757948, 10283260, 10807548,
                         11331836, 11857148, 12381436, 12906748, 13431036, 13955324, 14480636, 15004924, 15530236,
                         16054524, 16579836, 16317692, 16055548, 15793404, 15269116, 15006972, 14744828, 14220540,
                         13958396, 13696252, 13171964, 12909820, 12647676, 12123388, 11861244, 11599100, 11074812,
                         10812668, 10550524, 10288380, 9764092, 9501948, 9239804, 8715516, 8453372, 8191228, 7666940,
                         7404796, 7142652, 6618364, 6356220, 6094076, 5569788, 5307644, 5045500, 4783356, 4259068,
                         3996924, 3734780, 3210492, 2948348, 2686204, 2161916, 1899772, 1637628, 1113340, 851196,
                         589052, 64764, 63740, 62716, 61692, 59644, 58620, 57596, 55548, 54524, 53500, 51452, 50428,
                         49404, 47356, 46332, 45308, 43260, 42236, 41212, 40188, 38140, 37116, 36092, 34044, 33020,
                         31996, 29948, 28924, 27900, 25852, 24828, 23804, 21756, 20732, 19708, 18684, 16636, 15612,
                         14588, 12540, 11516, 10492, 8444, 7420, 6396, 4348, 3324, 2300, 252, 248, 244, 240, 236, 232,
                         228, 224, 220, 216, 212, 208, 204, 200, 196, 192, 188, 184, 180, 176, 172, 168, 164, 160, 156,
                         152, 148, 144, 140, 136, 132, 128, 124, 120, 116, 112, 108, 104, 100, 96, 92, 88, 84, 80, 76,
                         72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 0, 0), dtype=np.int32)


class Fractal:

    @njit(parallel=True, fastmath=True, cache=True)
    def mandelbrot(w, h):

        center = complex(0.15, 0.0)  # could be parameters
        scale: float32 = 1.5
        iter: int32 = 200
        _range = -1, 1

        ratio: float32 = w / h
        cr = complex(_range[0], _range[0])

        def do_scale(i, j):
            return cr + complex((_range[1] - _range[0]) * i / w,
                                (_range[1] - _range[0]) * j / h)

        img = np.empty((h, w), dtype=np.int32)

        for j in prange(h):  # thread chop this index

            for i in range(w):

                z = c0 = complex(scale * ratio * do_scale(i, j) - center)

                for ix in range(iter):
                    z = z * z + c0  # z*z is the typical 2nd order fractal
                    if abs(z) > 2: break

                ix_col: int32 = 0 if ix == iter - 1 else int32(len(fire_palette) * ix // 50) % len(fire_palette)

                img[j, i] = 0xff00_0000 | fire_palette[ix_col]

        return img

    @njit(parallel=True, fastmath=True, cache=True)
    def julia(w, h):
        fract = lambda x: x - int(x)

        def f2color(cm):
            return int32(0xff00_0000 | \
                         (int32(255. * fract(cm + 0.0 / 3.0))) | \
                         (int32(255. * fract(cm + 1.0 / 3.0)) << 8) | \
                         (int32(255. * fract(cm + 2.0 / 3.0)) << 16))

        center = complex(0.49, 0.32)  # params?
        iter: int32 = 150
        _range = 0, +1
        cr = complex(_range[0], _range[0])

        def do_scale(i, j):
            return cr + complex((_range[1] - _range[0]) * i / w,
                                (_range[1] - _range[0]) * j / h)

        img = np.empty((h, w), dtype=np.int32)

        for j in prange(h):  # thread chop this index

            for i in range(w):

                c0: complex = do_scale(i, j)
                z = complex(5.0 * (c0.real - 0.5), 3.0 * (c0.imag - 0.5))
                c = center

                for ix in range(iter):
                    z = z * z + c
                    if abs(z) > 2: break

                img[j, i] = f2color(fract((0 if ix == iter - 1 else ix) * 10. / iter))

        return img


def test_fractal(n, type='mandel'):
    print('running fractal...')

    t = time()
    img = Fractal.mandelbrot(n, n) if type == 'mandel' else Fractal.julia(n, n)
    t = time() - t

    Image.frombytes(mode='RGBA', size=(n, n), data=img).show()  # .save('voronoi.png', format='png')
    print(f'lap fractal {n} x {n} :{t:.2}"')


if __name__ == '__main__':
    n = 1024 * 4
    test_fractal(n, 'mandel')
    test_fractal(n, 'julia')
