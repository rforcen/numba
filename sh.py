'''
Shperical Harmonics in njit format
'''
import timeit
from math import sin, cos, sqrt

import cmasher as cmr
import numpy as np
from numba import njit, prange, vectorize, guvectorize, float32, int32


class SpherHarm:

    @njit(parallel=True, fastmath=True)
    def calc_vertexes(n, vcode):
        vtx_out = np.empty((n, n, 3), dtype=np.float32)
        for i in prange(n):
            dv = i / n
            f0 = sin(vcode[0] * dv) ** vcode[1] + cos(vcode[2] * dv) ** vcode[3]
            sin_du, cos_du = sin(dv), cos(dv)

            for j in prange(n):
                du = j / n
                r = f0 + sin(vcode[4] * du) ** vcode[5] + cos(vcode[6] * du) ** vcode[7]
                vtx_out[i, j] = (r * sin_du * cos(du), r * cos_du, r * sin_du * sin(du))
        return vtx_out

    @njit(parallel=True, fastmath=True)
    def calc_vertex_texture(n, vcode):
        vtx_out = np.empty((n, n, 3), dtype=np.float32)
        txt_out = np.empty((n, n, 3), dtype=np.float32)

        for i in prange(n):
            dv = i / n
            f0 = sin(vcode[0] * dv) ** vcode[1] + cos(vcode[2] * dv) ** vcode[3]
            sin_du, cos_du = sin(dv), cos(dv)

            for j in prange(n):
                du = j / n
                r = f0 + sin(vcode[4] * du) ** vcode[5] + cos(vcode[6] * du) ** vcode[7]
                vtx_out[i, j] = (r * sin_du * cos(du), r * cos_du, r * sin_du * sin(du))
                txt_out[i, j] = (dv, du, 0.)
        return vtx_out, txt_out

    @njit(parallel=True, fastmath=True)
    def calc_normals(vertexes):
        def normal_3v(v0, v1, v2):  # use discrete vars for improved performance
            def sub(v0, v1): return v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]

            def div(v, x): return v[0] / x, v[1] / x, v[2] / x

            def cross(v0, v1):
                return v0[1] * v1[2] - v0[2] * v1[1], v0[2] * v1[0] - v0[0] * v1[2], v0[0] * v1[1] - v0[1] * v1[0]

            def mag(v): return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]

            nrm = cross(sub(v1, v0), sub(v2, v0))
            d = mag(nrm)

            return div(nrm, sqrt(d)) if d > 0 else (0., 0., 0.)

        normals = np.empty(vertexes.shape, vertexes.dtype)
        n = vertexes.shape[0]

        for i in prange(n):
            for j in prange(n):
                normals[i, j] = normal_3v(vertexes[i, j], vertexes[i, (j + 1) % n], vertexes[(i + 1) % n, j])
        return normals

    @njit(parallel=True, fastmath=True)
    def calc_texture(n):
        texture = np.empty((n, n, 3), dtype=float32)
        for i in prange(n):
            for j in prange(n):
                texture[i, j] = i / n, j / n, 0.
        return texture

    def calc_mesh(n, vcode):
        def fmt_code(a):
            return np.array(a, dtype=np.float32)

        _lap = timeit.default_timer()

        vertexes, textures = SpherHarm.calc_vertex_texture(n, fmt_code(vcode))
        normals = SpherHarm.calc_normals(vertexes)
        colors = cmr.take_cmap_colors('cmr.rainforest', n)

        return timeit.default_timer() - _lap, vertexes, normals, colors, textures


def test_sh():
    n = 1024 * 12
    _lap, vtx, nrm, col, txt = SpherHarm.calc_mesh(n, vcode=[0, 1, 2, 2, 2, 4, 1, 2])
    print(f'mesh: n={n}, items={n ** 2}, time:{_lap}')
    print(vtx[0, :3])


if __name__ == '__main__':
    test_sh()
