'''
Voronoi problem solved with numba vectorize & njit(paralell)
'''
import timeit

import numpy as np
from PIL import Image
from numba import vectorize, njit, prange, int32
from math import sin


def voronoi(size, points, colors):
    h, w = size
    n: int = w * h
    n_points: int = len(points)
    amask: int = np.int32(0xff00_0000)
    max_int: int = np.iinfo(np.int32).max

    @vectorize('int32(int32)', target='parallel', nopython=True, fastmath=True)
    def calc_color(ix):  # current index 0..n -> color

        def distance_squared(p0, p1):
            d0, d1 = p0[0] - p1[0], p0[1] - p1[1]
            return d0 * d0 + d1 * d1

        min_dist = max_int
        circ_diam = 1  # as distance is squared
        ind = -1

        current_point = ix % w, ix // w

        for i in range(n_points):
            d = distance_squared(points[i], current_point)

            if d < circ_diam: break
            if d < min_dist:
                min_dist = d
                ind = i

        return amask if ind == -1 else colors[ind] | amask

    return calc_color(np.arange(n).astype('i4'))


@njit(parallel=True, fastmath=True)
def voronoi_jit(size, points, colors):
    h, w = size
    n: int = w * h
    n_points: int = len(points)
    amask: int = np.int32(0xff00_0000)
    max_int: int = np.iinfo(np.int32).max

    def calc_color(ix):  # current index 0..n -> color

        def distance_squared(p0, p1):
            d0, d1 = p0[0] - p1[0], p0[1] - p1[1]
            return d0 * d0 + d1 * d1

        min_dist = max_int
        circ_diam = 1  # as distance is squared
        ind = -1

        current_point = ix % w, ix // w

        for i in range(n_points):
            d = distance_squared(points[i], current_point)

            if d < circ_diam: break
            if d < min_dist:
                min_dist = d
                ind = i

        return amask if ind == -1 else colors[ind] | amask

    img = np.empty(n, dtype=int32)

    for i in prange(n):
        img[i] = calc_color(i)

    return img


def test_voronoi():
    sz = 1024 * 1

    size = (sz, sz)
    n = sz
    n_points = n * 3
    points = np.random.uniform(0, min(size), size=n_points * 2).reshape(n_points, 2).astype('i4')  # x,y
    colors = np.random.uniform(0x0000_0000, 0x00ff_ffff, size=n_points).astype('i4')

    t0 = timeit.default_timer()

    image = voronoi(size, points, colors)
    # image = voronoi_jit(size, points, colors)

    t0 = timeit.default_timer() - t0

    img = Image.frombytes(mode='RGBA', size=size, data=image).show()  # .save('voronoi.png', format='png')

    print(f'generated voronoi, {n_points} points, of {size} in {t0:.3} secs')


if __name__ == '__main__':
    test_voronoi()
