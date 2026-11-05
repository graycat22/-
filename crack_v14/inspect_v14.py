"""V14: geodesic crossing-path crack inspection.

crack = a path from the U rim (inner) to the outer edge (ribs / bottom) that runs
almost entirely over crack-like pixels.  score = fraction of *intact* material the
best path still has to traverse (0 = cut through end to end, ~1 = healthy).
"""
import cv2, numpy as np
from skimage.graph import MCP_Geometric
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
    def __init__(self, kind, eps=0.03, min_thick=60, edge_margin=None):
        from .geometry import GEOM
        self.kind, self.eps = kind, eps
        self.m = build_masks(kind)
        y0 = GEOM[kind]["y_eval"]
        inner = self.m["inner"].copy(); inner[:y0 - 60] = False
        self.starts = np.argwhere(inner)
        walk = self.m["band"] | self.m["inner"] | self.m["outer"]
        walk[:y0 - 80] = False
        self.walk = walk
        base = np.where(walk, 1.0, np.inf)
        cum0, _ = MCP_Geometric(base, fully_connected=True).find_costs(self.starts)
        ends = np.argwhere(self.m["outer"])
        th = cum0[ends[:, 0], ends[:, 1]]
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

    def run(self, prob):
        prob = np.where(self.edge_zone, 0.0, prob)
        cost = np.where(self.walk, self.eps + (1.0 - prob), np.inf)
        mcp = MCP_Geometric(cost, fully_connected=True)
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
        return dict(ratio=float(ratio[k]), max_gap=float(gap), length=float(length), path=path)

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
        """refs: {"B": gray_ref, "D": gray_ref} already in reference frame (960x660)."""
        from .register import Reference
        self.cross = {k: Crossing(k) for k in refs}
        self.refs = {k: Reference(k, g) for k, g in refs.items()}
        self.cas = {}
        for k in refs:
            m = self.cross[k].m
            self.cas[k] = cv2.dilate((m["plate"] | m["opening"]).astype(np.uint8), np.ones((41, 41), np.uint8))
        self.ratio_th, self.gap_th = ratio_th, gap_th
        self.d_lo, self.d_hi = 0.10, 0.30
        self.d_ratio_on_b = 0.15      # D contrast (fraction of plate range) -> prob

    @staticmethod
    def classify(gray):
        """B = back-light visible through the U opening (large saturated area in the centre)."""
        h, w = gray.shape
        c = gray[int(h * .1):int(h * .6), int(w * .3):int(w * .7)]
        p10, p90 = np.percentile(c, [10, 90])
        return "B" if p10 > 0.5 * p90 else "D"     # B: opening almost uniformly bright

    def classify_and_align(self, gray):
        from .register import register, refine_ecc
        k = self.classify(gray)
        H, n = register(gray, self.refs[k])
        if H is None:
            return k, None, n
        H, cc = refine_ecc(gray, self.refs[k].gray, H, self.cas[k])
        return k, H, n

    def inspect(self, gray):
        kind, H, n = self.classify_and_align(gray)
        if H is None:
            return dict(kind=None, judgment="ERROR", reason="registration failed")
        warped = cv2.warpPerspective(gray, H, SIZE)
        cr = self.cross[kind]
        band = cr.m["band"]
        res = {}
        # D-path (dark line) on every image, B-path (bright gap) only where backlight exists
        pd = z_to_prob(dark_line_map(warped, band), self.d_lo, self.d_hi)
        if kind == "B":
            pb = z_to_prob(bright_gap_map(warped, band), 2.5, 6.0)
            res["B"] = cr.run_multi(pb, ratio_th=self.ratio_th, gap_th=self.gap_th)
            # the dark rim of a B opening is not a separate D crack
            near_b = cv2.dilate((pb > 0.3).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            pd = np.where(near_b, 0.0, pd)
        # D on a back-lit (B-kind) image: the body is in silhouette and full of dark mould
        # lines, so a D candidate there needs much stronger evidence (TODO: replace by OK-MAD)
        d_ratio = self.ratio_th if kind == "D" else self.d_ratio_on_b
        res["D"] = cr.run_multi(pd, ratio_th=d_ratio, gap_th=self.gap_th)
        types = [t for t in ("B", "D") if t in res and res[t][0]["ng"]]
        judgment = "NG[" + "+".join(types) + "]" if types else "OK"
        return dict(kind=kind, inliers=n, H=H, warped=warped, res=res, judgment=judgment)
