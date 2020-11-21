'''
guvectorize playground
'''
from timeit import default_timer as time
from math import sin, cos
import numpy as np
from numba import vectorize, guvectorize, njit, float64, int32, int64, float32, config, prange


# set to const
@guvectorize([(float32[:, :], int32[:], float32[:, :])], '(n,n3), (n) -> (n,n3)', target='parallel')
def gu01(v_in, seq, v_out):
    v_out[:] = 1 + seq[0]


# set to const + seq[0]
@guvectorize([(float32[:, :], int32[:], float32[:, :])], '(n,n3), (n) -> (n,n3)', target='parallel')
def gu02(v_in, seq, v_out):
    v_out[:] = seq[0]


# set to [] seq[:]
@guvectorize([(float32[:, :], int32[:], float32[:, :])], '(n,n3), (n) -> (n,n3)', target='parallel')
def gu03(v_in, seq, v_out):
    v_out[:] = np.array([seq[0], seq[1], seq[2]], dtype=np.float32)


# use a calc
@guvectorize([(float32[:, :], int32[:], float32[:, :])], '(n,n3), (n) -> (n,n3)', target='parallel')
def gu04(v_in, seq, v_out):
    def calc_vect(x):
        return np.full((3), x, dtype=np.float32)

    v_out[:] = calc_vect(seq[0])


# use a calc
@guvectorize([(float32[:, :], int32[:], float32[:, :])], '(n,n3), (n) -> (n,n3)', target='parallel')
def gu05(v_in, seq, v_out):
    def calc_vect(x):
        v = np.empty(3, dtype=np.float32)
        for i in range(3):
            v[i] = x + i
        return v

    for i in range(v_in.shape[0]):
        v_out[i, :] = calc_vect(seq[i] + i)


# use sh
@guvectorize([(int32, float32[:], float32[:, :], int32[:], float32[:, :])],
             '(), (nvc), (n,n3), (n) -> (n,n3)', target='parallel', fastmath=True, nopython=True)
def gu06(n, vcode, v, seq, _):
    nn = n * n

    def calc_point3d(ix):
        du, dv = ix % nn, ix // nn

        f0 = sin(vcode[0] * dv) ** vcode[1] + cos(vcode[2] * dv) ** vcode[3]
        sin_du, cos_du = sin(dv), cos(dv)

        r = f0 + sin(vcode[4] * du) ** vcode[5] + cos(vcode[6] * du) ** vcode[7]
        return r * sin_du * cos(du), r * cos_du, r * sin_du * sin(du)

    for i in range(n):  # for current row
        v[i] = calc_point3d(seq[i])


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


def test_range_nn():  # 5:1 for numba
    grid(2)  # warm up

    n = 1024 * 30

    # print('starting arange.reshape...')
    t0 = time()
    seq = np.arange(n * n, dtype='i4').reshape(n, n)
    t0 = time() - t0

    # print('done!\nstarting xrange...')
    t1 = time()
    seq1 = grid(n,n)
    t1 = time() - t1

    print(f'range_nn test: arange {t0}, xrange {t1}, ratio numpy/numba: {t0 / t1}')
    # print(seq, seq1, np.all(seq == seq1))


def test_range_n():
    xrange(2)  # warm up

    n = 1024 * 30

    # print('starting arange.reshape...')
    t0 = time()
    seq = np.arange(n * n, dtype='i4')
    t0 = time() - t0

    # print('done!\nstarting xrange...')
    t1 = time()
    seq1 = xrange(n * n)
    t1 = time() - t1

    print(f'range_n test: arange {t0}, xrange {t1}, ratio numpy/numba: {t0 / t1}')
    print(seq, seq1, np.all(seq == seq1))


def test_gu():
    n = 1024 * 10
    v = vo = np.zeros((n, n, 3), dtype=np.float32)
    seq = grid(n)  # np.arange(n * n, dtype='i4').reshape(n, n)
    # vo = gu05(v, seq, v)
    t = time()

    vo = gu06(n, np.array([0, 1, 2, 2, 2, 4, 1, 2], dtype=np.float32), v, seq, v)

    print(vo[:1], time() - t)


test_range_n()
# test_gu()
