"""V14: geodesic crossing-path crack inspection.

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
        """refs: {"type1": gray_ref, "type2": gray_ref}  (product types, in the 960x660 reference frame).

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
        return "type1" if p10 > 0.5 * p90 else "type2"   # (unused for the decision; rough hint only)

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
