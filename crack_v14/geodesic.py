"""最短経路探索（ダイクストラ法）。画像の各画素を「通るのにかかる手間（コスト）」として、
起点の集まりから各画素まで、一番手間の少ない道とその手間の合計を求める。

  隣の画素への移動: 上下左右と斜めの8方向
  移動の手間     : (今の画素のコスト + 隣の画素のコスト) / 2 × 移動距離（斜めは √2）
  コストが inf   : 壁（通れない）
numba があれば高速版、無ければ純Python版（遅いが同じ結果）で動く。
"""
import heapq
import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

_DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int64)
_DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int64)
_DL = np.sqrt(_DY.astype(np.float64) ** 2 + _DX.astype(np.float64) ** 2)


def _dijkstra_py(cost, starts):
    h, w = cost.shape
    cum = np.full((h, w), np.inf); prev = np.full((h, w), -1, np.int64)
    pq = []
    for y, x in starts:
        if np.isfinite(cost[y, x]):
            cum[y, x] = 0.0; heapq.heappush(pq, (0.0, int(y), int(x)))
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > cum[y, x]:
            continue
        c0 = cost[y, x]
        for k in range(8):
            ny, nx = y + _DY[k], x + _DX[k]
            if 0 <= ny < h and 0 <= nx < w:
                c1 = cost[ny, nx]
                if not np.isfinite(c1):
                    continue
                nd = d + 0.5 * (c0 + c1) * _DL[k]
                if nd < cum[ny, nx]:
                    cum[ny, nx] = nd; prev[ny, nx] = y * w + x
                    heapq.heappush(pq, (nd, int(ny), int(nx)))
    return cum, prev


if _HAS_NUMBA:
    @numba.njit(cache=True)
    def _dijkstra_nb(cost, sy, sx, DY, DX, DL):
        h, w = cost.shape
        cum = np.full((h, w), np.inf)
        prev = np.full((h, w), -1, np.int64)
        # 「まだ調べていない画素」を手間の小さい順に取り出すための入れ物（二分ヒープ）
        cap = 1 << 16
        hd = np.empty(cap, np.float64); hi = np.empty(cap, np.int64); n = 0
        for i in range(sy.shape[0]):
            y, x = sy[i], sx[i]
            if np.isfinite(cost[y, x]) and cum[y, x] > 0.0:
                cum[y, x] = 0.0
                if n == cap:
                    cap *= 2; hd2 = np.empty(cap, np.float64); hi2 = np.empty(cap, np.int64)
                    hd2[:n] = hd[:n]; hi2[:n] = hi[:n]; hd = hd2; hi = hi2
                j = n; n += 1
                while j > 0 and hd[(j - 1) >> 1] > 0.0:
                    p = (j - 1) >> 1; hd[j] = hd[p]; hi[j] = hi[p]; j = p
                hd[j] = 0.0; hi[j] = y * w + x
        while n > 0:
            d = hd[0]; v = hi[0]
            n -= 1
            ld = hd[n]; li = hi[n]; j = 0
            while True:
                c = 2 * j + 1
                if c >= n:
                    break
                if c + 1 < n and hd[c + 1] < hd[c]:
                    c += 1
                if hd[c] >= ld:
                    break
                hd[j] = hd[c]; hi[j] = hi[c]; j = c
            if n > 0:
                hd[j] = ld; hi[j] = li
            y = v // w; x = v - y * w
            if d > cum[y, x]:
                continue
            c0 = cost[y, x]
            for k in range(8):
                ny = y + DY[k]; nx = x + DX[k]
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                c1 = cost[ny, nx]
                if not np.isfinite(c1):
                    continue
                nd = d + 0.5 * (c0 + c1) * DL[k]
                if nd < cum[ny, nx]:
                    cum[ny, nx] = nd; prev[ny, nx] = y * w + x
                    if n == cap:
                        cap *= 2; hd2 = np.empty(cap, np.float64); hi2 = np.empty(cap, np.int64)
                        hd2[:n] = hd[:n]; hi2[:n] = hi[:n]; hd = hd2; hi = hi2
                    j = n; n += 1
                    while j > 0 and hd[(j - 1) >> 1] > nd:
                        p = (j - 1) >> 1; hd[j] = hd[p]; hi[j] = hi[p]; j = p
                    hd[j] = nd; hi[j] = ny * w + nx
        return cum, prev


class MCP:
    """使い方: mcp = MCP(コスト画像); cum, _ = mcp.find_costs(起点の一覧); 道 = mcp.traceback(終点)
    cum[y, x] = 起点からその画素までの最小の手間。traceback は起点 -> 終点の (y, x) の並びを返す。"""

    def __init__(self, cost):
        self.cost = np.ascontiguousarray(cost, np.float64)
        self.prev = None

    def find_costs(self, starts):
        starts = np.asarray(starts, np.int64)
        if _HAS_NUMBA:
            cum, self.prev = _dijkstra_nb(self.cost, starts[:, 0].copy(), starts[:, 1].copy(), _DY, _DX, _DL)
        else:
            cum, self.prev = _dijkstra_py(self.cost, starts)
        return cum, None

    def traceback(self, end):
        w = self.cost.shape[1]
        y, x = int(end[0]), int(end[1]); out = [(y, x)]
        v = self.prev[y, x]
        while v >= 0:
            y, x = divmod(int(v), w); out.append((y, x)); v = self.prev[y, x]
        return out[::-1]
