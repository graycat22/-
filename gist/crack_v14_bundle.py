"""crack_v14 bundle: run `python crack_v14_bundle.py` to write the crack_v14/ package here.

Then:
  pip install -r requirements.txt
  python -m crack_v14.make_ref --ok-dir pos1ok --out refs   # build references from OK images (angle auto-detected)
  python -m crack_v14.make_ref --ok-dir pos2ok --out refs
  python -m crack_v14.cli --input <images> --out out --ref-dir refs   # refs/ref_pos1.png, refs/ref_pos2.png
  python -m crack_v14.evaluate --labels labels/defect_annotations.csv --pred out
NOTE: reference images are NOT included. geometry.py coordinates are for the 960x660
reference frame and must be re-drawn on your own reference images.
"""
import os
FILES = {}
FILES['crack_v14/__init__.py'] = r''''''
FILES['crack_v14/cli.py'] = r'''"""python -m crack_v14.cli --input DIR --out DIR [--screen]

Writes per-image overlay, a grid of all results, and results.csv
(img, view, type[B=明るい割れ/D=暗い割れ], rank, ratio, max_gap, rim_break, ng, bbox in input coords)."""
import argparse, csv, glob, os, cv2, numpy as np
from .inspect_v14 import Inspector
from .geometry import GEOM
from .screen_crop import crop_screen

def load(path, screen):
    im = cv2.imread(path)
    return crop_screen(im) if screen else im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--ref-dir", default="data/refs", help="ref_pos1.png / ref_pos2.png")
    ap.add_argument("--screen", action="store_true", help="inputs are phone photos of the monitor")
    ap.add_argument("--aligned", action="store_true", help="draw on the image warped to the reference (debug)")
    a = ap.parse_args()
    refs = {}
    for k in GEOM:
        rp = os.path.join(a.ref_dir, f"ref_{k}.png")
        refs[k] = cv2.imread(rp, 0)
        if refs[k] is None:
            raise SystemExit(f"基準画像が読めません: {rp}  (--ref-dir に ref_pos1.png と ref_pos2.png を置いてください)")
    ins = Inspector(refs)
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(a.input, f"*.{e}")))
    rows, tiles, summary = [], [], []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        img = load(f, a.screen)
        r = ins.inspect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if r["judgment"] == "ERROR":
            print(name, "ERROR", r.get("reason")); continue
        # draw on the input image itself (not the warped one), so nothing looks tilted
        vis = cv2.cvtColor(r["warped"], cv2.COLOR_GRAY2BGR) if a.aligned else img.copy()
        lw = max(2, round(vis.shape[1] / 320))
        msg = (f"{name:24s} view={r['kind']} backlit={'Y' if r['backlit'] else 'N'} "
               f"inl={r['inliers']:<4d} {r['judgment']:2s} {r['detail']:4s}")
        for t, col in (("D", (0, 0, 255)), ("B", (255, 200, 0))):
            for i, q in enumerate(r["res"].get(t, [])):
                p = (q["path"][:, ::-1] if a.aligned else q["path_img"]).astype(np.int32)
                (x0, y0), (x1, y1) = p.min(0), p.max(0)
                if q["ng"]:
                    cv2.polylines(vis, [p], False, col, lw)
                rb = q.get("rim_break")
                msg += (f" | {t}{i} r={q['ratio']:.2f} gap={q['max_gap']:.0f} in={q['inner_gap']:.0f} out={q['outer_gap']:.0f}"
                        + (f" rim={rb:.2f}" if rb is not None else "") + (f" reach={q['reach']:.2f}" if "reach" in q else "")
                        + (" NG" if q["ng"] else ""))
                (u0, v0), (u1, v1) = q["path_img"].min(0).astype(int), q["path_img"].max(0).astype(int)
                rows.append([name, r["kind"], t, i,
                             round(q["ratio"], 3), round(q["max_gap"], 1),
                             "" if rb is None else round(rb, 3), q["ng"], u0, v0, u1, v1])
        print(msg)
        summary.append([name, r["judgment"], r["detail"], len(r["defects"])])
        cv2.putText(vis, f"{name} {r['judgment']}", (10, 14 * lw), 0, 0.4 * lw, (0, 255, 255), lw)
        cv2.imwrite(os.path.join(a.out, f"{name}.jpg"), vis)
        t = cv2.resize(vis, (480, round(480 * vis.shape[0] / vis.shape[1])))
        tile = np.zeros((360, 480, 3), np.uint8); tile[:min(360, t.shape[0])] = t[:360]
        tiles.append(tile)
    if tiles:
        while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(a.out, "result_grid.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img", "view", "type", "rank", "ratio", "max_gap", "rim_break", "ng", "x1", "y1", "x2", "y2"])   # bbox in input-image coords
        w.writerows(rows)
    with open(os.path.join(a.out, "judgments.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["img", "judgment", "evidence", "n_defects"]); w.writerows(summary)

if __name__ == "__main__":
    main()
'''
FILES['crack_v14/evaluate.py'] = r'''"""Compare results with labels/defect_annotations.csv.

python -m crack_v14.evaluate --labels labels/defect_annotations.csv --pred out [--role VALIDATION]

labels columns: record_type,path,dataset_role,label_status,type,judgment,severity,x1,y1,x2,y2,note
  - only label_status == CONFIRMED is used
  - an image is NG if any of its rows has judgment == NG
  - rows with x1..y2 filled are ground-truth defect boxes (input-image coordinates)
pred: judgments.csv + results.csv written by crack_v14.cli
A NG image counts as "located" if any predicted defect box overlaps a GT box (GT box padded by --pad px).
"""
import argparse, csv, os
from collections import defaultdict

def key(p):
    return os.path.splitext(os.path.basename(p))[0]

def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def overlap(a, b, pad):
    return not (a[2] < b[0] - pad or a[0] > b[2] + pad or a[3] < b[1] - pad or a[1] > b[3] + pad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pred", default="out")
    ap.add_argument("--role", default=None, help="TRAIN / VALIDATION / TEST (default: all)")
    ap.add_argument("--pad", type=float, default=20)
    a = ap.parse_args()

    gt_ng, gt_box, role = {}, defaultdict(list), {}
    for r in csv.DictReader(open(a.labels, encoding="utf-8-sig")):
        if r.get("label_status", "CONFIRMED").strip().upper() != "CONFIRMED":
            continue
        if a.role and r.get("dataset_role", "").strip().upper() != a.role.upper():
            continue
        k = key(r["path"]); role[k] = r.get("dataset_role", "")
        j = r.get("judgment", "").strip().upper()
        gt_ng[k] = gt_ng.get(k, False) or j == "NG"
        box = [num(r.get(c)) for c in ("x1", "y1", "x2", "y2")]
        if j == "NG" and None not in box:
            gt_box[k].append(box)

    pred = {r["img"]: r["judgment"] == "NG" for r in csv.DictReader(open(os.path.join(a.pred, "judgments.csv")))}
    pbox = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(a.pred, "results.csv"))):
        if r["ng"] == "True":
            pbox[r["img"]].append([float(r[c]) for c in ("x1", "y1", "x2", "y2")])

    tp = fp = fn = tn = located = loc_total = 0
    misses, false_alarms, mislocated = [], [], []
    for k, ng in sorted(gt_ng.items()):
        if k not in pred:
            print("not predicted:", k); continue
        p = pred[k]
        if ng and p:
            tp += 1
            if gt_box[k]:
                loc_total += 1
                if any(overlap(b, g, a.pad) for b in pbox[k] for g in gt_box[k]):
                    located += 1
                else:
                    mislocated.append(k)
        elif ng:
            fn += 1; misses.append(k)
        elif p:
            fp += 1; false_alarms.append(k)
        else:
            tn += 1
    n = tp + fp + fn + tn
    print(f"images={n}  TP={tp} FN={fn} FP={fp} TN={tn}")
    if tp + fn: print(f"NG recall (見逃し率の逆) = {tp / (tp + fn):.3f}")
    if fp + tn: print(f"OK false-alarm rate (過検出率) = {fp / (fp + tn):.3f}")
    if loc_total: print(f"location hit = {located}/{loc_total}")
    for name, lst in (("見逃し", misses), ("過検出", false_alarms), ("位置ずれ", mislocated)):
        if lst: print(f"{name}: {' '.join(lst)}")

if __name__ == "__main__":
    main()
'''
FILES['crack_v14/geodesic.py'] = r'''"""Minimum-cost path on an 8-connected grid (replacement for skimage MCP_Geometric).

edge cost (a->b) = 0.5 * (cost[a] + cost[b]) * step_length, inf = wall.
Uses numba when available (fast), otherwise pure Python heapq (slow but works).
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
        # binary heap of (dist, flat index)
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
    """Same usage as skimage.graph.MCP_Geometric: find_costs(starts) -> (cum, None); traceback(end)."""

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
'''
FILES['crack_v14/geometry.py'] = r'''"""Hand-defined geometry in reference coordinates (960x660 reference frame).

opening : polygon of the U opening, up to the OUTER edge of the rim
plate   : polygon of the cassette body that a crack has to cross
outer   : polylines where a crack is considered to have reached the "other end"
"""
import cv2, numpy as np

SIZE = (960, 660)  # w, h

GEOM = {
    "pos2": dict(    # U字の内側にネジ筒が見える画角
        ref="s03", y_eval=120, edge_margin=10, min_thick=30,
        opening=[(195, 0), (200, 250), (215, 300), (250, 360), (300, 402), (380, 428), (450, 434),
                 (530, 422), (592, 390), (640, 350), (690, 290), (720, 230), (736, 0)],
        plate=[(152, 0), (758, 0), (758, 660), (152, 660)],
        rim_ignore=[(430, 400, 520, 460)],          # gate notch at 6 o'clock
        outer=[[(756, 0), (756, 660)], [(154, 0), (154, 660)], [(152, 657), (758, 657)]],
    ),
    "pos1": dict(    # U字の脚の上まで見える画角
        ref="s12", y_eval=380, edge_margin=14,
        opening=[(262, 60), (265, 350), (285, 420), (330, 470), (400, 497), (450, 502), (520, 497),
                 (590, 470), (630, 420), (652, 350), (652, 60)],
        plate=[(236, 60), (688, 60), (688, 660), (236, 660)],
        rim_ignore=[(440, 470, 510, 530)],
        outer=[[(686, 60), (686, 660)], [(238, 60), (238, 660)], [(236, 657), (688, 657)]],
    ),
}

def _poly(pts, shape, closed=True, thick=None):
    m = np.zeros(shape, np.uint8)
    p = np.array(pts, np.int32)
    if thick is None:
        cv2.fillPoly(m, [p], 1)
    else:
        cv2.polylines(m, [p], closed, 1, thick)
    return m.astype(bool)

def build_masks(kind, rim_band=None):
    g = GEOM[kind]
    rim_band = g.get("rim_band", 10) if rim_band is None else rim_band
    shape = (SIZE[1], SIZE[0])
    opening = _poly(g["opening"], shape)
    plate = _poly(g["plate"], shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_band + 1,) * 2)
    open_dil = cv2.dilate(opening.astype(np.uint8), k).astype(bool)
    inner = open_dil & ~opening & plate                 # thin band just outside the rim
    band = plate & ~opening                             # material the crack must cross
    outer = np.zeros(shape, bool)
    for line in g["outer"]:
        outer |= _poly(line, shape, closed=False, thick=5)
    outer &= band & ~open_dil
    return dict(opening=opening, plate=plate, band=band, inner=inner, outer=outer)
'''
FILES['crack_v14/inspect_v14.py'] = r'''"""V14: geodesic crossing-path crack inspection.

crack = a path from the U rim (inner) to the outer edge (ribs / bottom) that runs
almost entirely over crack-like pixels.  score = fraction of *intact* material the
best path still has to traverse (0 = cut through end to end, ~1 = healthy).
"""
import cv2, numpy as np
from .geodesic import MCP
from .geometry import build_masks, SIZE

# ---------------------------------------------------------------- crack likelihood
def _robust_z(x, mask):
    v = x[mask]
    med = np.median(v); mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return (x - med) / mad

def dark_line_map(gray, band, scales=(1.5, 3.0, 5.0, 8.0), se_sizes=(13, 25, 37)):
    """D: thin dark valley, returned as contrast relative to the plate's gray range.

    black-hat (closing - image) gives the valley depth in gray levels; the Hessian
    eigen-ratio keeps only elongated structures (lines, not blobs / texture)."""
    f = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    floor = 0.35 * np.percentile(f[band], 90)          # dark plate bottom: don't amplify noise
    rel = np.zeros_like(f)
    for k in se_sizes:                                 # crack width varies 3..20 px
        clo = cv2.morphologyEx(f, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        rel = np.maximum(rel, (clo - f) / np.maximum(clo, floor))   # relative darkness
    elong = np.zeros_like(f)
    for s in scales:
        g = cv2.GaussianBlur(f, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        t = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        l1 = 0.5 * (dxx + dyy + t); l2 = 0.5 * (dxx + dyy - t)
        e = np.where(l1 > 0, 1 - np.clip(np.abs(l2) / (l1 + 1e-6), 0, 1), 0)
        elong = np.maximum(elong, e)
    return rel * (0.3 + 0.7 * elong)

def bright_gap_map(gray, band):
    """B: backlight leaking through the split -> much brighter than the body around it."""
    f = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    bg = cv2.medianBlur(gray, 41).astype(np.float32)          # local body level
    th = f - np.minimum(bg, np.percentile(f[band], 60))       # bright vs body
    z = _robust_z(th, band)
    return z

def z_to_prob(z, z0=3.0, z1=8.0):
    return np.clip((z - z0) / (z1 - z0), 0, 1)

# ---------------------------------------------------------------- crossing path
class Crossing:
    def __init__(self, kind, eps=0.03, min_thick=None, edge_margin=None):
        from .geometry import GEOM
        self.kind, self.eps = kind, eps
        self.m = build_masks(kind)
        self._build_rim(GEOM[kind]["opening"])
        y0 = GEOM[kind]["y_eval"]
        inner = self.m["inner"].copy(); inner[:y0 - 60] = False
        self.starts = np.argwhere(inner)
        walk = self.m["band"] | self.m["inner"] | self.m["outer"]
        walk[:y0 - 80] = False
        self.walk = walk
        base = np.where(walk, 1.0, np.inf)
        cum0, _ = MCP(base).find_costs(self.starts)
        ends = np.argwhere(self.m["outer"])
        th = cum0[ends[:, 0], ends[:, 1]]
        if min_thick is None:
            min_thick = GEOM[kind].get("min_thick", 60)
        keep = (ends[:, 0] >= y0) & np.isfinite(th) & (th >= min_thick)
        self.ends, self.thick = ends[keep], th[keep]               # healthy thickness per end
        # plate border: rib / frame edges are strong lines that are not cracks -> no credit there
        if edge_margin is None:
            edge_margin = GEOM[kind].get("edge_margin", 0)
        plate = self.m["plate"].astype(np.uint8)
        if edge_margin > 0:
            ez = ~cv2.erode(plate, np.ones((2 * edge_margin + 1,) * 2, np.uint8)).astype(bool)
            ez[-(edge_margin + 4):] = False          # frame bottom is an image border, not a rib
            self.edge_zone = ez
        else:
            self.edge_zone = np.zeros_like(self.m["plate"])

    def _build_rim(self, opening_poly, step=2.0, depth=16):
        """Dense rim contour + inward sample offsets (rim highlight lies just inside the opening polygon)."""
        p = np.array(opening_poly, np.float32)
        seg = [np.linspace(p[i], p[i + 1], max(2, int(np.hypot(*(p[i + 1] - p[i])) / step)), endpoint=False)
               for i in range(len(p) - 1)]
        c = np.vstack(seg)
        t = np.gradient(c, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        # orient normals into the opening
        probe = (c + 4 * nrm).astype(int)
        probe[:, 0] = probe[:, 0].clip(0, self.m["opening"].shape[1] - 1)
        probe[:, 1] = probe[:, 1].clip(0, self.m["opening"].shape[0] - 1)
        inside = self.m["opening"][probe[:, 1], probe[:, 0]]
        nrm[~inside] *= -1
        self.rim_pts, self.rim_nrm, self.rim_depth = c, nrm, depth

    def rim_profile(self, gray):
        """max brightness across the rim at each contour position (rim highlight strength)."""
        d = np.arange(-4, self.rim_depth)
        xy = self.rim_pts[:, None, :] + d[None, :, None] * self.rim_nrm[:, None, :]
        x = xy[..., 0].clip(0, gray.shape[1] - 1).astype(np.float32)
        y = xy[..., 1].clip(0, gray.shape[0] - 1).astype(np.float32)
        v = cv2.remap(cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32), x, y, cv2.INTER_LINEAR)
        return np.percentile(v, 85, axis=1), d[np.argmax(v, axis=1)].astype(np.float32)

    def rim_break(self, gray, start_xy, win=10, ctx=60):
        """1 - (rim strength at crack start) / (rim strength around it).  Real crack cuts the rim."""
        R, off = self.rim_profile(gray)
        i = int(np.argmin(np.hypot(*(self.rim_pts - start_xy).T)))
        lo, hi = max(0, i - win), min(len(R), i + win + 1)
        a, b = slice(max(0, i - ctx), max(0, i - 2 * win)), slice(i + 2 * win, i + ctx)
        ref = np.r_[R[a], R[b]]
        if len(ref) < 5 or len(R[a]) < 3 or len(R[b]) < 3:
            return 0.0, i
        drop = 1 - R[lo:hi].min() / (np.median(ref) + 1e-6)          # highlight cut
        step = abs(np.median(off[a]) - np.median(off[b])) / 6.0          # highlight displaced (6px = full)
        return float(drop), i          # offset step was too noisy (double rim line)

    def rim_ignored(self, i):
        from .geometry import GEOM
        x, y = self.rim_pts[i]
        return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in GEOM[self.kind].get("rim_ignore", []))

    def run(self, prob):
        prob = np.where(self.edge_zone, 0.0, prob)
        cost = np.where(self.walk, self.eps + (1.0 - prob), np.inf)
        mcp = MCP(cost)
        cum, _ = mcp.find_costs(self.starts)
        vals = cum[self.ends[:, 0], self.ends[:, 1]]
        ratio = vals / np.maximum(self.thick, 1)
        k = int(np.argmin(ratio))
        path = np.array(mcp.traceback(tuple(self.ends[k])))
        # longest run of pixels on the path that are NOT crack-like (the "bridge" that still holds)
        on = prob[path[:, 0], path[:, 1]] > 0.15
        step = np.r_[0, np.hypot(*np.diff(path, axis=0).T)]
        gap, cur = 0.0, 0.0
        for o, s in zip(on, step):
            cur = 0.0 if o else cur + s
            gap = max(gap, cur)
        length = step.sum()
        cum_len = np.cumsum(step)
        idx = np.flatnonzero(on)
        # how far the crack stops short of each end (0 = it really reaches that edge)
        inner_gap = float(cum_len[idx[0]]) if len(idx) else float(length)
        outer_gap = float(length - cum_len[idx[-1]]) if len(idx) else float(length)
        return dict(ratio=float(ratio[k]), max_gap=float(gap), length=float(length), path=path,
                    inner_gap=inner_gap, outer_gap=outer_gap, prob=prob)

    def reach_inside(self, prob, start_xy, half=3):
        """Does the crack continue through the rim into the U opening?
        Fraction of crack-like pixels on the rim normals at the start point (rim -> opening)."""
        i = int(np.argmin(np.hypot(*(self.rim_pts - start_xy).T)))
        js = range(max(0, i - half), min(len(self.rim_pts), i + half + 1))
        d = np.arange(0, self.rim_depth + 1)
        vals = []
        for j in js:
            xy = self.rim_pts[j][None, :] + d[:, None] * self.rim_nrm[j][None, :]
            x = xy[:, 0].clip(0, prob.shape[1] - 1).astype(int); y = xy[:, 1].clip(0, prob.shape[0] - 1).astype(int)
            vals.append((prob[y, x] > 0.15).mean())
        return float(max(vals)) if vals else 0.0, i

    def run_multi(self, prob, k=3, ratio_th=0.35, gap_th=25.0, sep=35):
        """Up to k disjoint crossing paths; stops when the next one is no longer crack-like."""
        out, p = [], prob.copy()
        for _ in range(k):
            r = self.run(p)
            r["ng"] = r["ratio"] <= ratio_th and r["max_gap"] <= gap_th
            out.append(r)
            if not r["ng"]:
                break
            m = np.zeros(p.shape, np.uint8)
            cv2.polylines(m, [r["path"][:, ::-1].astype(np.int32)], False, 1, 2 * sep + 1)
            p[m.astype(bool)] = 0.0
        return out

# ---------------------------------------------------------------- full inspection
class Inspector:
    def __init__(self, refs, ratio_th=0.35, gap_th=25.0):
        """refs: {"pos1": gray_ref, "pos2": gray_ref}  (camera angles, in the 960x660 reference frame).

        Crack types: B = bright crack (light leaking through), D = dark crack line."""
        from .register import Reference
        self.cross = {k: Crossing(k) for k in refs}
        self.refs = {k: Reference(k, g) for k, g in refs.items()}
        self.shape_mask, self.ref_grad = {}, {}
        for k, g in refs.items():
            m = self.cross[k].m
            ring = cv2.dilate(m["opening"].astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & \
                   ~cv2.erode(m["opening"].astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
            self.shape_mask[k] = m["band"] | ring
            gg = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2)
            self.ref_grad[k] = cv2.magnitude(cv2.Sobel(gg, cv2.CV_32F, 1, 0), cv2.Sobel(gg, cv2.CV_32F, 0, 1))
        self.cas = {}
        for k in refs:
            m = self.cross[k].m
            self.cas[k] = cv2.dilate((m["plate"] | m["opening"]).astype(np.uint8), np.ones((41, 41), np.uint8))
        self.ratio_th, self.gap_th = ratio_th, gap_th
        self.d_lo, self.d_hi = 0.10, 0.30
        self.rim_break_th = 0.30
        self.end_gap_inner, self.end_gap_outer = 10.0, 25.0   # px the crack may stop short of each edge
        self.reach_th = 0.90                                 # B: light must continue into the opening      # D contrast (fraction of plate range) -> prob

    @staticmethod
    def classify(gray):
        """B = back-light visible through the U opening (large saturated area in the centre)."""
        h, w = gray.shape
        c = gray[int(h * .1):int(h * .6), int(w * .3):int(w * .7)]
        p10, p90 = np.percentile(c, [10, 90])
        return "pos1" if p10 > 0.5 * p90 else "pos2"   # (unused for the decision; rough hint only)

    def _shape_score(self, warped, k):
        """Lighting-independent fit: correlation of edge strength with the reference,
        measured on the cassette body and rim only (the U opening shows back-light or
        background depending on lighting, so it is left out)."""
        def grad(g):
            g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2)
            return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        m = self.shape_mask[k]
        a, b = grad(warped)[m], self.ref_grad[k][m]
        a = (a - a.mean()) / (a.std() + 1e-6); b = (b - b.mean()) / (b.std() + 1e-6)
        return float((a * b).mean())

    def classify_and_align(self, gray):
        """Find the camera angle (which reference view) and align to it.

        The view is decided by SHAPE only (edges of body / rim), never by brightness,
        because either angle may be shot with or without back-light."""
        from .register import register, refine_ecc, features
        f = min(1.0, 1.3 * max(SIZE) / max(gray.shape))     # register on a reference-sized copy
        small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
        S = np.diag([f, f, 1.0])
        feats = features(small)
        best = None
        for k, ref in self.refs.items():
            H, n = register(small, ref, feats=feats)
            if H is None:
                continue
            Hf, _ = refine_ecc(gray, ref.gray, H @ S, self.cas[k])
            # a degenerate fit (input does not cover the reference body) cannot be this angle
            cover = cv2.warpPerspective(np.ones_like(gray), Hf, SIZE).astype(bool)
            if n < 8 or cover[self.shape_mask[k]].mean() < 0.85:
                continue
            sc = self._shape_score(cv2.warpPerspective(gray, Hf, SIZE), k)
            if best is None or sc > best[0]:
                best = (sc, k, Hf, n)
        if best is None:
            return None, None, 0
        _, k, Hf, n = best
        return k, Hf, n
        return k, Hf, n

    def backlit(self, warped, kind):
        """Is there light behind the part?  Checked inside the U opening after alignment."""
        op = cv2.erode(self.cross[kind].m["opening"].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        op[: int(op.shape[0] * 0.15)] = False
        v = warped[op]
        top = np.percentile(warped, 99)
        return bool(len(v) and np.percentile(v, 25) > 0.6 * top)

    def inspect(self, gray):
        kind, H, n = self.classify_and_align(gray)
        if kind is None or H is None:
            return dict(kind=None, judgment="ERROR", reason="registration failed")
        warped = cv2.warpPerspective(gray, H, SIZE)
        cr = self.cross[kind]
        band = cr.m["band"]
        res = {}
        # D-path (dark line) on every image, B-path (bright gap) only where backlight exists
        pd = z_to_prob(dark_line_map(warped, band), self.d_lo, self.d_hi)
        # dark-line evidence on every image; light-leak evidence whenever back-light is present
        # (a front-lit view has no light behind the part, so a leak cannot show there)
        lit = self.backlit(warped, kind)
        if lit:
            pb = z_to_prob(bright_gap_map(warped, band), 2.5, 6.0)
            res["B"] = cr.run_multi(pb, ratio_th=self.ratio_th, gap_th=self.gap_th)
            # the dark rim of a bright opening is not a separate dark crack
            near_b = cv2.dilate((pb > 0.3).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            pd = np.where(near_b, 0.0, pd)
        res["D"] = cr.run_multi(pd, ratio_th=self.ratio_th, gap_th=self.gap_th)
        # human rule: a real crack opens the part, so the bright rim highlight is CUT where it
        # starts.  A scratch / mould line / shadow leaves the rim intact.
        for q in res["D"]:
            q["cross_ng"] = q["ng"]
            q["rim_break"], ri = cr.rim_break(warped, q["path"][0][::-1].astype(np.float32))
            if cr.rim_ignored(ri):
                q["rim_break"] = 0.0                       # gate notch etc.: fixed rim gap
            q["ng"] = q["cross_ng"] and q["rim_break"] >= self.rim_break_th
        # definition: the crack runs END TO END -- from the U opening through the wall to the outer edge.
        #   both ends: the crack itself must reach the inner band and the outer edge (no intact stub left)
        #   B: the light leak must connect through the rim into the opening (reach_inside)
        #   D: the rim highlight must be cut where it starts (rim_break, above)
        for t, cands in res.items():
            for q in cands:
                q.setdefault("cross_ng", q["ng"])
                ends_ok = q["inner_gap"] <= self.end_gap_inner and q["outer_gap"] <= self.end_gap_outer
                if t == "B":
                    q["reach"], _ = cr.reach_inside(q["prob"], q["path"][0][::-1].astype(np.float32))
                    q["ng"] = q["cross_ng"] and ends_ok and q["reach"] >= self.reach_th
                else:
                    q["ng"] = q["ng"] and ends_ok
                q.pop("prob", None)
        # defects in ORIGINAL image coordinates (for comparison with annotations)
        Hinv = np.linalg.inv(H)
        defects = []
        for t, cands in res.items():
            for q in cands:
                pts = cv2.perspectiveTransform(q["path"][:, ::-1].reshape(-1, 1, 2).astype(np.float64), Hinv)
                q["path_img"] = pts.reshape(-1, 2)
                if q["ng"]:
                    (x1, y1), (x2, y2) = q["path_img"].min(0), q["path_img"].max(0)
                    defects.append(dict(type=t,
                                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                                        ratio=q["ratio"], max_gap=q["max_gap"], rim_break=q.get("rim_break")))
        types = sorted({t for t in res if any(q["ng"] for q in res[t])})
        judgment = "NG" if defects else "OK"
        return dict(kind=kind, backlit=lit, inliers=n, H=H, warped=warped, res=res, judgment=judgment,
                    defects=defects, detail="+".join(types))
'''
FILES['crack_v14/make_ref.py'] = r'''"""Build a reference image from your own OK images (e.g. pos1ok) in the existing reference frame.

python -m crack_v14.make_ref --ok-dir pos1ok --out refs     # angle is detected automatically
python -m crack_v14.make_ref --ok-dir pos2ok --out refs
python -m crack_v14.make_ref --ok-dir pos1ok --view pos1 --out refs   # without provisional refs:
                                                    # the first OK image becomes the reference

How it works
  1. the first OK image is aligned ONCE to the current reference (data/refs/ref_<view>.png)
  2. every other OK image is aligned to that first OK image (real-to-real: accurate)
  3. all of them are warped into the reference frame and the pixel-wise median is taken
     -> refs_new/ref_<view>.png   (same frame as geometry.py, so geometry needs no redrawing)
  4. refs_new/check_<view>.jpg shows the geometry on the new reference: check it by eye
     (green = inner rim band, red = outer edge, blue = area a crack has to cross)
Also writes ok_median_<view>.png / ok_mad_<view>.png for the OK-deviation model later.
"""
import argparse, glob, os, cv2, numpy as np
from .geometry import SIZE, build_masks
from .register import Reference, register, refine_ecc, features

def load_gray(p):
    g = cv2.imread(p, 0)
    if g is None:
        raise SystemExit(f"読めない画像: {p}")
    return g

def small_of(g):
    f = min(1.0, 1.3 * max(SIZE) / max(g.shape))
    return (cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else g), np.diag([f, f, 1.0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ok-dir", required=True)
    ap.add_argument("--view", default=None, choices=["pos1", "pos2"], help="省略すると自動判定")
    ap.add_argument("--old-ref-dir", default="data/refs")
    ap.add_argument("--out", default="refs_new")
    ap.add_argument("--max", type=int, default=40, help="use at most this many OK images")
    ap.add_argument("--bootstrap", action="store_true", help="仮の基準画像を使わず、先頭のOK画像を基準にする")
    ap.add_argument("--move-geom", type=float, nargs=2, default=(0, 0), metavar=("DX", "DY"),
                    help="check画像の緑・赤の線を右にDX、下にDY px動かす（960x660座標）")
    ap.add_argument("--scale-geom", type=float, default=1.0, help="線を中心から何倍に広げるか（例 1.03）")
    a = ap.parse_args()
    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(a.ok_dir, f"*.{e}")))[: a.max]
    if not files:
        raise SystemExit(f"画像がありません: {a.ok_dir}")
    os.makedirs(a.out, exist_ok=True)
    from .inspect_v14 import Inspector
    from .geometry import GEOM
    first = load_gray(files[0])
    fs, S = small_of(first)
    old_paths = {k: os.path.join(a.old_ref_dir, f"ref_{k}.png") for k in GEOM}
    have_old = all(os.path.exists(p) for p in old_paths.values()) and not a.bootstrap
    if have_old:
        olds = {k: load_gray(p) for k, p in old_paths.items()}
        ins = Inspector(olds if a.view is None else {a.view: olds[a.view]})
        view, H0, n0 = ins.classify_and_align(first)
        if H0 is None:
            raise SystemExit("1枚目のOK画像を今の基準画像に位置合わせできませんでした（別の画像を先頭にしてください）")
        print(f"{os.path.basename(files[0])}: 画角={view} として仮の基準画像に位置合わせ (inliers={n0})")
    else:
        # no provisional reference: the first OK image itself, resized to the reference frame,
        # becomes the frame.  geometry.py was drawn on a full-frame view, so it fits roughly;
        # use --move-geom / --scale-geom after looking at check_<view>.jpg if it is off.
        if a.view is None:
            raise SystemExit("仮の基準画像が無いときは --view pos1 / pos2 を指定してください")
        view = a.view
        h, w = first.shape
        H0 = np.diag([SIZE[0] / w, SIZE[1] / h, 1.0])
        print(f"{os.path.basename(files[0])}: 仮の基準画像なし -> 先頭のOK画像を {SIZE[0]}x{SIZE[1]} に縮小して基準にします")
    # manual correction: move / scale the inspection lines (green/red) as seen in check_<view>.jpg
    dx, dy = a.move_geom
    c = np.array([SIZE[0] / 2, SIZE[1] / 2])
    sc = a.scale_geom
    T = np.array([[1 / sc, 0, c[0] - c[0] / sc - dx / sc], [0, 1 / sc, c[1] - c[1] / sc - dy / sc], [0, 0, 1]])
    H0 = T @ H0
    a.view = view
    m = build_masks(view)
    cas = cv2.dilate((m["plate"] | m["opening"]).astype(np.uint8), np.ones((41, 41), np.uint8))

    # first OK image becomes the anchor; others are matched against it at full quality
    anchor = Reference("anchor", fs)
    stack = [cv2.warpPerspective(first, H0, SIZE)]
    for f in files[1:]:
        g = load_gray(f)
        gs, Sg = small_of(g)
        H, n = register(gs, anchor, feats=features(gs))
        if H is None or n < 30:
            print(f"{os.path.basename(f)}: skipped (inliers={n})"); continue
        Hf = H0 @ np.linalg.inv(S) @ H @ Sg                 # g -> first(small) -> first(full) -> ref
        Hf, cc = refine_ecc(g, stack[0], Hf, cas)             # polish against the first warped OK
        stack.append(cv2.warpPerspective(g, Hf, SIZE))
        print(f"{os.path.basename(f)}: ok (inliers={n}, ecc={cc:.3f})")
    st = np.stack(stack).astype(np.float32)
    med = np.median(st, axis=0)
    mad = np.median(np.abs(st - med), axis=0) * 1.4826
    ref = np.clip(med, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(a.out, f"ref_{a.view}.png"), ref)
    cv2.imwrite(os.path.join(a.out, f"ok_median_{a.view}.png"), ref)
    cv2.imwrite(os.path.join(a.out, f"ok_mad_{a.view}.png"), np.clip(mad * 4, 0, 255).astype(np.uint8))
    np.save(os.path.join(a.out, f"ok_mad_{a.view}.npy"), mad)

    vis = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    over = vis.copy(); over[m["band"]] = (200, 80, 0)
    vis = cv2.addWeighted(over, 0.25, vis, 0.75, 0)
    vis[m["inner"]] = (0, 220, 0); vis[m["outer"]] = (0, 0, 255)
    cv2.imwrite(os.path.join(a.out, f"check_{a.view}.jpg"), vis)
    print(f"\n{len(stack)} 枚から作成 -> {a.out}/ref_{a.view}.png")
    print(f"{a.out}/check_{a.view}.jpg を見て、緑が U字内縁の縁取りの外側、赤が縦リブ/下端に乗っているか確認してください")
    print(f"良ければ --ref-dir {a.out} で cli を実行（もう片方の view の ref も同じフォルダに置くこと）")

if __name__ == "__main__":
    main()
'''
FILES['crack_v14/register.py'] = r'''"""Reference registration: SIFT + RANSAC homography (production: small shift only, still works)."""
import cv2, numpy as np

_sift = cv2.SIFT_create(nfeatures=4000)

def _prep(gray):
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)

class Reference:
    def __init__(self, name, gray, mask=None):
        """mask: only use keypoints on the cassette itself (background fixtures are 3-D and mislead)."""
        self.name, self.gray = name, gray
        m = None if mask is None else (mask.astype(np.uint8) * 255)
        self.kp, self.des = _sift.detectAndCompute(_prep(gray), m)

def features(gray):
    return _sift.detectAndCompute(_prep(gray), None)

def register(gray, ref, ratio=0.75, feats=None):
    """returns (H img->ref, n_inliers); pass feats=features(gray) to reuse keypoints"""
    kp, des = feats if feats is not None else features(gray)
    if des is None or len(kp) < 10:
        return None, 0
    m = cv2.BFMatcher().knnMatch(des, ref.des, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < ratio * b.distance]
    if len(good) < 12:
        return None, len(good)
    src = np.float32([kp[g.queryIdx].pt for g in good])
    dst = np.float32([ref.kp[g.trainIdx].pt for g in good])
    cv2.setRNGSeed(0)                                     # same input -> same answer
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)       # robust with few matches (OpenCV >= 4.5)
    H, inl = cv2.findHomography(src, dst, method, 4.0, maxIters=10000, confidence=0.999)
    return H, int(inl.sum()) if inl is not None else 0


def refine_ecc(gray, ref_gray, H, mask, iters=80):
    """Refine an initial homography with ECC on the cassette region (ref frame)."""
    h, w = ref_gray.shape
    warped = cv2.warpPerspective(gray, H, (w, h))
    a = cv2.GaussianBlur(_prep(ref_gray), (0, 0), 2).astype(np.float32)
    b = cv2.GaussianBlur(_prep(warped), (0, 0), 2).astype(np.float32)
    W = np.eye(3, dtype=np.float32)
    try:
        cc, W = cv2.findTransformECC(a, b, W, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5),
                                     mask.astype(np.uint8), 5)
    except cv2.error:
        return H, 0.0
    # W maps ref -> warped coords; compose: img -> warped (H) -> ref (W^-1)
    return np.linalg.inv(W).astype(np.float64) @ H, float(cc)
'''
FILES['crack_v14/screen_crop.py'] = r'''"""Cut the monitor area out of a phone photo of the screen (only for the screen-photo test set)."""
import cv2, numpy as np

def crop_screen(im, W=960, H=660):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = g.shape[0]
    g2 = g.copy(); g2[: int(h * 0.31)] = 0; g2[int(h * 0.81):] = 0   # status bar / thumbnails
    m = (cv2.GaussianBlur(g2, (0, 0), 5) > 45).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    cnt, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnt, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    ap = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(ap) == 4:
        p = ap.reshape(4, 2).astype(np.float32)
        s = p.sum(1); d = np.diff(p, axis=1).ravel()
        src = np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)
    else:
        x, y, w, hh = cv2.boundingRect(c)
        src = np.float32([[x, y], [x + w, y], [x + w, y + hh], [x, y + hh]])
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(im, M, (W, H))
'''
FILES['requirements.txt'] = r'''# 動作確認済み: Python 3.12 / opencv 4.9.0 / numpy 1.26.4 / numba 0.60.0
opencv-python>=4.4      # SIFT が本体に入ったのは 4.4 から（headless 版でも可）
numpy>=1.24
numba>=0.57             # 無くても動く（最短経路が純Pythonになり遅くなる）
'''

if __name__ == "__main__":
    for path, src in FILES.items():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        print("wrote", path)
