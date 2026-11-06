"""「内縁から外縁まで、割れっぽい所だけを通って抜けられるか」を調べる部分。

考え方:
  割れらしさマップを「通りやすさ」にして（割れっぽい所ほど通りやすい）、
  U字の内縁（inner）から外縁（outer）までの一番通りやすい道を最短経路探索で求める。
  その道の上で「割れていない所」をどれだけ通らされたかで判定する。

  ratio     : 道の上で健全部を通った長さ ÷ その場所の健全な肉厚（0 = 端から端まで割れ、1 = 健全）
  max_gap   : 道の上で割れていない区間が一番長く続いた長さ [px]
  inner_gap : 割れが内側の端からどれだけ手前で止まっているか [px]
  outer_gap : 割れが外側の端からどれだけ手前で止まっているか [px]
"""
import cv2
import numpy as np

from .geodesic import MCP
from .geometry import build_masks

ON = 0.15   # 割れらしさがこれより大きい画素を「割れの上」とみなす


class Crossing:
    def __init__(self, geom, eps=0.03):
        """geom: 検査範囲の定義（u_detect.detect_geometry が作ったもの）。"""
        self.g, self.eps = geom, eps
        self.m = build_masks(geom)
        # 縁取りの太さ（u_detect が測ったもの）の分だけ内側まで調べる（それより内側は開口部の背景なので見ない）。
        # 割れが縁取りを内側の端まで切っているか（= U字の内側まで割れているか）を確かめるため
        rw = geom.get("rim_width")
        self._build_rim(geom["opening"], depth=16 if rw is None else max(8, int(round(rw))))
        y0 = geom["y_eval"]

        # 起点: 開口部の縁のすぐ外側の帯（U字の脚の上の方は除く）
        inner = self.m["inner"].copy()
        inner[: max(0, y0 - 60)] = False
        self.starts = np.argwhere(inner)

        # 歩ける範囲: 樹脂部分＋起点＋終点
        walk = self.m["band"] | self.m["inner"] | self.m["outer"] | self.m["rim"]
        walk[: max(0, y0 - 80)] = False
        self.walk = walk

        # 健全なときの肉厚（全部コスト1で歩いたときの距離）を終点ごとに測っておく
        cum0, _ = MCP(np.where(walk, 1.0, np.inf)).find_costs(self.starts)
        ends = np.argwhere(self.m["outer"])
        th = cum0[ends[:, 0], ends[:, 1]]
        # 起点を縁取りの内側に置いたときは、縁取りの太さぶん厚く測れるので、その分を足して比べる
        extra = (geom.get("rim_width") or 0) + 6 if self.m["rim"].any() else 0
        keep = (ends[:, 0] >= y0 - 60) & np.isfinite(th) & (th >= geom.get("min_thick", 60) + extra)
        self.ends, self.thick = ends[keep], th[keep]

        # プレートの端（縦リブの縁）は黒い線が必ずあるので、割れの証拠として数えない帯にする
        em = geom.get("edge_margin", 0)
        if em > 0:
            plate = self.m["plate"].astype(np.uint8)
            ez = ~cv2.erode(plate, np.ones((2 * em + 1,) * 2, np.uint8)).astype(bool)
            yb = int(max(y for _, y in geom["plate"]))
            ez[max(0, yb - (em + 4)):] = False   # プレートの下の端はリブではないので除外しない
            self.edge_zone = ez
        else:
            self.edge_zone = np.zeros_like(self.m["plate"])

    # -------------------------------------------------------------- U字の縁取り
    def _build_rim(self, opening_poly, step=2.0, depth=16):
        """開口部の輪郭を 2px おきの点にして、各点で「開口部の内側へ向かう向き」を求めておく。"""
        p = np.array(opening_poly, np.float32)
        seg = [np.linspace(p[i], p[i + 1], max(2, int(np.hypot(*(p[i + 1] - p[i])) / step)), endpoint=False)
               for i in range(len(p) - 1)]
        c = np.vstack(seg)
        t = np.gradient(c, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)            # 輪郭に垂直な向き
        probe = (c + 4 * nrm).astype(int)
        probe[:, 0] = probe[:, 0].clip(0, self.m["opening"].shape[1] - 1)
        probe[:, 1] = probe[:, 1].clip(0, self.m["opening"].shape[0] - 1)
        inside = self.m["opening"][probe[:, 1], probe[:, 0]]
        nrm[~inside] *= -1                                # 開口部の内側を向くようにそろえる
        self.rim_pts, self.rim_nrm, self.rim_depth = c, nrm, depth

    def rim_profile(self, gray):
        """輪郭の各点で、縁取りを横切る方向（線の前後 8px）に明るさを測り、縁取りの明るさ（上位15%の値）を返す。"""
        d = np.arange(-8, 9)          # 縁取りの線（縁取りの上にある）の前後 8px
        xy = self.rim_pts[:, None, :] + d[None, :, None] * self.rim_nrm[:, None, :]
        x = xy[..., 0].clip(0, gray.shape[1] - 1).astype(np.float32)
        y = xy[..., 1].clip(0, gray.shape[0] - 1).astype(np.float32)
        v = cv2.remap(cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32), x, y, cv2.INTER_LINEAR)
        return np.percentile(v, 85, axis=1)

    def nearest_rim(self, xy):
        """点 xy に一番近い輪郭点の番号。"""
        return int(np.argmin(np.hypot(*(self.rim_pts - xy).T)))

    def rim_break(self, gray, start_xy, win=20, ctx=100):
        """縁取りの途切れ度（0 = 途切れていない、1 = 完全に切れている）。

        割れの起点付近（±win 点）の縁取りの明るさを、その前後（ctx 点の範囲）の明るさと比べる。
        本物の割れは樹脂が割れて開いているので、縁取りの白い光も切れる。傷や影では切れない。
        """
        R = self.rim_profile(gray)
        i = self.nearest_rim(start_xy)
        lo, hi = max(0, i - win), min(len(R), i + win + 1)
        a, b = slice(max(0, i - ctx), max(0, i - 2 * win)), slice(i + 2 * win, i + ctx)
        ref = np.r_[R[a], R[b]]
        if len(ref) < 5 or len(R[a]) < 3 or len(R[b]) < 3:
            return 0.0, i
        return float(1 - R[lo:hi].min() / (np.median(ref) + 1e-6)), i

    def rim_ignored(self, i):
        """輪郭点 i が、最初から縁取りが切れている場所（ゲートの切り欠き）にあるか。"""
        x, y = self.rim_pts[i]
        return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in self.g.get("rim_ignore", []))

    def reach_inside(self, prob, start_xy, half=3):
        """（B用）割れが縁取りを越えて開口部までつながっているか（0〜1、1 = 全部つながっている）。"""
        i = self.nearest_rim(start_xy)
        d = np.arange(0, self.rim_depth + 1)
        vals = []
        for j in range(max(0, i - half), min(len(self.rim_pts), i + half + 1)):
            xy = self.rim_pts[j][None, :] + d[:, None] * self.rim_nrm[j][None, :]
            x = xy[:, 0].clip(0, prob.shape[1] - 1).astype(int)
            y = xy[:, 1].clip(0, prob.shape[0] - 1).astype(int)
            vals.append((prob[y, x] > ON).mean())
        return (float(max(vals)) if vals else 0.0), i

    # -------------------------------------------------------------- 経路探索
    def run(self, prob):
        """割れらしさマップ prob から、内縁 -> 外縁の一番割れっぽい道を1本求める。"""
        prob = np.where(self.edge_zone, 0.0, prob)
        cost = np.where(self.walk, self.eps + (1.0 - prob), np.inf)   # 割れっぽい所ほど安い
        mcp = MCP(cost)
        cum, _ = mcp.find_costs(self.starts)
        vals = cum[self.ends[:, 0], self.ends[:, 1]]
        ratio = vals / np.maximum(self.thick, 1)
        k = int(np.argmin(ratio))                                     # 一番割れっぽく抜けられる終点
        path = np.array(mcp.traceback(tuple(self.ends[k])))           # (y, x) の並び、起点 -> 終点

        on = prob[path[:, 0], path[:, 1]] > ON
        step = np.r_[0, np.hypot(*np.diff(path, axis=0).T)]
        gap, cur = 0.0, 0.0
        for o, s in zip(on, step):                                   # 割れていない区間の最長
            cur = 0.0 if o else cur + s
            gap = max(gap, cur)
        length = step.sum()
        cum_len = np.cumsum(step)
        idx = np.flatnonzero(on)
        inner_gap = float(cum_len[idx[0]]) if len(idx) else float(length)
        outer_gap = float(length - cum_len[idx[-1]]) if len(idx) else float(length)
        return dict(ratio=float(ratio[k]), max_gap=float(gap), length=float(length), path=path,
                    inner_gap=inner_gap, outer_gap=outer_gap, prob=prob)

    def run_multi(self, prob, k=3, ratio_th=0.35, gap_th=25.0, sep=35, reject=None, max_iter=10):
        """割れが複数あるときのために、見つけた道の周り（sep px）を消して最大 k 本まで探す。

        reject(道) が True の道（U字の縁や縦リブに沿って走る溝・影）は割れの候補にせず、
        その道の線だけを細く（5px）消して探し直す。太く消すと、縁から始まる本物の割れの根元まで消えるため。
        """
        out, p = [], prob.copy()
        for _ in range(max_iter):
            r = self.run(p)
            r["ng"] = r["ratio"] <= ratio_th and r["max_gap"] <= gap_th
            line = [r["path"][:, ::-1].astype(np.int32)]
            if r["ng"] and reject is not None and reject(r):
                m = np.zeros(p.shape, np.uint8)
                cv2.polylines(m, line, False, 1, 5)
                p[m.astype(bool)] = 0.0
                continue
            out.append(r)
            if not r["ng"] or len(out) >= k:
                break
            m = np.zeros(p.shape, np.uint8)
            cv2.polylines(m, line, False, 1, 2 * sep + 1)
            p[m.astype(bool)] = 0.0
        if not out:
            out.append(r)
        return out
