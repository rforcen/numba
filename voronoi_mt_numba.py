from PIL import Image
from math import sqrt
from numba import njit, jit, guvectorize, vectorize, int32, float32, get_num_threads, prange
import timeit
import numpy as np


@njit(parallel=True, fastmath=True)
def voronoi_njit(n, points, colors):
    h: int32 = n
    w: int32 = n
    n2: int32 = n * n

    amask: int32 = np.int32(0xff00_0000)
    max_int: int32 = np.iinfo(np.int32).max
    n_pnts: int32 = len(points)

    img = np.empty((n2), dtype=int32)

    for ix in prange(n2):
        y: int32 = ix // w
        x: int32 = ix % w

        min_dist: int32 = max_int
        circ_diam: int32 = 1  # as distance is squared
        ind: int32 = -1

        d: int32 = 0

        for i in range(n_pnts):
            # d = distance_squared points[i] - x,y
            pnts = points[i]
            d0, d1 = pnts[0] - x, pnts[1] - y
            d = d0 * d0 + d1 * d1

            if d < circ_diam: break
            if d < min_dist:
                min_dist = d
                ind = i

        img[ix] = amask if ind == -1 else colors[ind] | amask

    return img


@njit(parallel=True)  # -> np.arange(n * n, dtype='i4').reshape(n, n)
def grid(n, m):
    v = np.empty((n, m), dtype=np.int32)
    for i in prange(n):
        for j in prange(m):
            v[i][j] = i * m + j
    return v


@njit(parallel=True)  # -> np.arange(n, dtype='i4')
def xrange(n):
    v = np.empty((n), dtype=np.int32)
    for i in prange(n):
        v[i] = i
    return v


def voronoi(n, points, colors):
    h: int32 = n
    w: int32 = n

    amask: int32 = np.int32(0xff00_0000)
    max_int: int32 = np.iinfo(np.int32).max
    n_pnts: int32 = len(points)

    @vectorize([int32(int32)], target='parallel', nopython=True, fastmath=True)  # seq -> color
    def vect_voronoi(ix):

        y: int32 = ix // w
        x: int32 = ix % w

        min_dist: int32 = max_int
        circ_diam: int32 = 1  # as distance is squared
        ind: int32 = -1

        d: int32 = 0

        for i in range(n_pnts):
            # d = distance_squared points[i] - x,y
            pnts = points[i]
            d0, d1 = pnts[0] - x, pnts[1] - y
            d = d0 * d0 + d1 * d1

            if d < circ_diam: break
            if d < min_dist:
                min_dist = d
                ind = i

        return amask if ind == -1 else colors[ind] | amask

    return vect_voronoi(xrange(n * n))


def test_voronoi():
    n = 1024 * 1
    n_pts = n * 3

    points = np.random.uniform(0, n, (n_pts, 2)).astype('i4')
    colors = np.random.uniform(0, 0x00ff_0000, (n_pts)).astype('i4')

    voronoi(n, points, colors)  # warm up...
    voronoi_njit(n, points, colors)

    print(f'doing Voronoi for {n * n} items, {n_pts} points, using {get_num_threads()} threads...')

    t0 = timeit.default_timer()

    img = voronoi(n, points, colors)
    # img = voronoi_njit(n, points, colors)

    t0 = timeit.default_timer() - t0
    print(f'done, lap: {t0}')

    Image.frombytes(mode='RGBA', size=(n, n), data=img).show()  # .save('voronoi.png', format='png')


if __name__ == '__main__':
    # test_gu()
    test_voronoi()
