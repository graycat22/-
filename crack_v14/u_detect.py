"""1枚の画像から、U字の開口部・縁取り・縦リブを自動で見つける（製品の種類を問わない）。

考え方:
  どの製品でも共通しているのは「U字の縁取りは白く光っていて、そのすぐ外側は黒い樹脂のプレート」
  ということ。そこで、U字の内側のある点（中心）から放射状に線を伸ばし、各方向で
      内側が明るく -> 外側が暗い（縁取りの外側の境目）
      その外側はなめらかで模様が少ない（樹脂のプレート）
  となる距離を探す。方向ごとにバラバラに選ぶとガタガタになるので、
  「隣の方向との距離の差は小さく」という条件で、ぐるっと一周分を動的計画法でまとめて選ぶ。
  中心の位置は分からないので、いくつか試して一番はっきりU字になったものを使う。

  そのあと:
    縁取りの線をもう一度細かく境目に吸い付け、縁取りの太さを測る（snap_rim）
    U字の脚の外側で一番はっきりした縦の線を縦リブとする（find_ribs）
    U字の一番下（6時の位置）をゲートの切り欠きとして、縁取りの途切れ判定から外す
"""
import cv2
import numpy as np

from .geometry import SIZE, build_masks

ANGLES = np.deg2rad(np.arange(0, 182, 2))     # 真右 -> 真下 -> 真左（画像は下向きが +y）。脚はそこから真上に伸ばす
R_STEP = 2.0
OUT = 50          # 外側を何 px 先までなめらかか調べるか
W_OUT = 0.8       # 外側の模様の多さの減点の重み
R_BONUS = 0.0     # 1px 外側ほど得する点数


def _smooth(a, k):
    """点の並びを前後 k 点の平均でなめらかにする（両端はそのまま伸ばす）。"""
    if k <= 1 or len(a) < k:
        return a
    pad = np.concatenate([np.repeat(a[:1], k // 2, 0), a, np.repeat(a[-1:], k // 2, 0)])
    ker = np.ones(k) / k
    return np.stack([np.convolve(pad[:, i], ker, mode="valid") for i in range(a.shape[1])], 1)


def _dp_path(score, max_jump, penalty):
    """score[角度, 距離] の表で、隣の角度との距離の差が max_jump 以内の、合計が一番大きい道。"""
    A, R = score.shape
    acc = score[0].copy()
    back = np.zeros((A, R), np.int64)
    idx = np.arange(R)
    for i in range(1, A):
        best = np.full(R, -1e18)
        arg = np.zeros(R, np.int64)
        for sh in range(-max_jump, max_jump + 1):
            prev = np.full(R, -1e18)
            if sh >= 0:
                prev[sh:] = acc[: R - sh]
            else:
                prev[:sh] = acc[-sh:]
            prev = prev - penalty * abs(sh)
            m = prev > best
            best[m], arg[m] = prev[m], idx[m] - sh       # 前の角度での距離の番号
        acc = best + score[i]
        back[i] = arg
    path = np.zeros(A, np.int64)
    path[-1] = int(np.argmax(acc))
    for i in range(A - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path, float(acc.max()) / A


def _polar_score(img, edge, cx, cy, radii):
    """中心 (cx, cy) から見た「縁取りの外側の境目らしさ」の表 [角度, 距離]。"""
    H, W = img.shape
    rr = np.arange(radii[0] - 30, radii[-1] + OUT + 30, R_STEP)          # 前後に余裕を持たせて一度に取る
    X = (cx + np.outer(np.cos(ANGLES), rr)).astype(np.float32)
    Y = (cy + np.outer(np.sin(ANGLES), rr)).astype(np.float32)
    inside = (X >= 0) & (X < W - 1) & (Y >= 0) & (Y < H - 1)
    I = cv2.remap(img, X, Y, cv2.INTER_LINEAR)
    E = cv2.remap(edge, X, Y, cv2.INTER_LINEAR)
    o = int(30 / R_STEP)                                              # radii[0] が rr の何番目か
    n = len(radii)
    k3 = int(round(3 / R_STEP))
    step = I[:, o - k3:o - k3 + n] - I[:, o + k3:o + k3 + n]          # 内側 3px - 外側 3px（内が明るいと +）
    # 外側 6〜OUT px の輪郭の強さの平均（樹脂のプレートなら小さい）。
    # ネジ筒などの輪郭の外側には、すぐ U字の縁取りなど別の線があるので大きくなる
    cs = np.cumsum(np.pad(E, ((0, 0), (1, 0))), axis=1)
    a, b = int(6 / R_STEP), int(OUT / R_STEP)
    out_edge = (cs[:, o + b + 1:o + b + 1 + n] - cs[:, o + a:o + a + n]) / (b - a + 1)
    # 外側ほど少し得をさせる（同じくらいはっきりした境目なら、一番外の縁取りを選ぶ）
    s = np.clip(step, 0, 60) - W_OUT * out_edge + R_BONUS * radii[None, :]
    ok = inside[:, o:o + n] & inside[:, o + b:o + b + n]
    return np.where(ok, s, -5.0)


def find_rim(frame):
    """U字の縁取りの外側の線を探す。戻り値: (点の並び 左の脚 -> 底 -> 右の脚, 中心, はっきり度)"""
    H, W = frame.shape
    img = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 1.5)
    g = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 1.0)
    edge = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    edge = np.clip(edge / (np.percentile(edge, 99) + 1e-6) * 60, 0, 60).astype(np.float32)
    radii = np.arange(20, 0.75 * W, R_STEP)
    best = None
    for cx in np.linspace(0.3 * W, 0.7 * W, 9):
        for cy in np.linspace(0.05 * H, 0.55 * H, 6):
            sc = _polar_score(img, edge, cx, cy, radii)
            path, total = _dp_path(sc, max_jump=8, penalty=0.6)
            if best is None or total > best[0]:
                best = (total, cx, cy, path)
    total, cx, cy, path = best
    # 中心を細かく寄せる
    for d in (12, 4):
        for dx, dy in ((d, 0), (-d, 0), (0, d), (0, -d)):
            sc = _polar_score(img, edge, cx + dx, cy + dy, radii)
            p2, t2 = _dp_path(sc, max_jump=8, penalty=0.6)
            if t2 > total:
                total, cx, cy, path = t2, cx + dx, cy + dy, p2
    r = radii[path]
    arc = _smooth(np.c_[cx + r * np.cos(ANGLES), cy + r * np.sin(ANGLES)][::-1], 5)   # 左 -> 底 -> 右
    # 真横に近い方向は縦リブなど外側の線を拾いやすいので使わず（両端 6 点 = 12°ぶん捨てる）、
    # そこから上は、U字の脚を1行ずつ上へ追いかける（脚がまっすぐでも、広がっていても合う）
    arc = arc[6:-6]
    left = trace_leg(img, edge, arc[0], side=-1)
    right = trace_leg(img, edge, arc[-1], side=+1)[::-1]
    return np.vstack([left, arc, right]), (cx, cy), total


def trace_leg(img, edge, start, side, row_step=4, search=40, max_shift=3):
    """U字の脚の縁取りの外側の線を、start から画像の上端まで1行ずつ追いかける。

    各行で、start の x から ±search px の範囲について「内側（U字の中心側）が明るく、外側が暗く、
    その外側がなめらか」な強さを測り、隣の行とのずれが max_shift px 以内でなめらかにつながる道を
    動的計画法で選ぶ。side = -1 は左の脚（外側は左）、+1 は右の脚（外側は右）。
    """
    H, W = img.shape
    ys = np.arange(start[1], -1, -row_step)                   # start から上へ
    xs = np.arange(start[0] - search, start[0] + search + 1, 1.0)
    X, Y = np.meshgrid(xs, ys)

    def at(M, dx):
        xx = np.clip(X + dx, 0, W - 1).astype(np.float32)
        return cv2.remap(M, xx, Y.astype(np.float32), cv2.INTER_LINEAR)
    step = at(img, -3 * side) - at(img, 3 * side)                 # 内側 - 外側（内が明るいと +）
    out_edge = np.mean([at(edge, d * side) for d in range(6, OUT + 1, 6)], axis=0)
    sc = np.clip(step, 0, 60) - W_OUT * out_edge
    sc[0] -= 0.5 * np.abs(xs - start[0])                          # 1行目は start の近くから始める
    idx, _ = _dp_path(sc, max_jump=max_shift, penalty=1.0)
    pts = np.c_[xs[idx], ys]
    return pts[::-1][:-1]                                         # 上端 -> start の手前（start は arc 側にある）


def regularize(rim, cx, cy, deg=6, iters=3):
    """見つけた縁取りの線を「綺麗なU字」に直す（ガタつき・部分的な外れを取る）。

      脚（中心より上）: 左右それぞれ1本の直線に当てはめる（外れた点は除いて当てはめ直す）
      底（中心より下）: 中心から見た距離を、角度のなめらかな関数（多項式）に当てはめる
    """
    rim = np.asarray(rim, np.float64)
    legs_l = rim[(rim[:, 1] < cy) & (rim[:, 0] < cx)]
    legs_r = rim[(rim[:, 1] < cy) & (rim[:, 0] >= cx)]
    arc = rim[rim[:, 1] >= cy]

    def robust_line(pts):                      # x = a + b*y
        if len(pts) < 4:
            return None
        keep = np.ones(len(pts), bool)
        for _ in range(iters):
            b_, a_ = np.polyfit(pts[keep, 1], pts[keep, 0], 1)
            res = np.abs(pts[:, 0] - (a_ + b_ * pts[:, 1]))
            keep = res < max(3.0, 2.5 * np.median(res[keep]))
        return a_, b_

    th = np.arctan2(arc[:, 1] - cy, arc[:, 0] - cx)
    r = np.hypot(arc[:, 0] - cx, arc[:, 1] - cy)
    keep = np.ones(len(th), bool)
    for _ in range(iters):
        coef = np.polyfit(th[keep], r[keep], deg)
        res = np.abs(r - np.polyval(coef, th))
        keep = res < max(3.0, 2.5 * np.median(res[keep]))
    t2 = np.linspace(th.max(), th.min(), 120)  # 左（大きい角度）-> 底 -> 右
    a2 = np.c_[cx + np.polyval(coef, t2) * np.cos(t2), cy + np.polyval(coef, t2) * np.sin(t2)]

    out = []
    L, R = robust_line(legs_l), robust_line(legs_r)
    # U字は左右対称なので、脚の傾きは左右で鏡写し（左の傾き = -右の傾き）にそろえ、大きく傾けない
    if L is not None and R is not None:
        bm = float(np.clip((L[1] - R[1]) / 2, -0.08, 0.08))
        L, R = (L[0], bm), (R[0], -bm)
    ys = np.arange(0, a2[0][1], 6.0)
    if L is not None:
        xl = L[0] + L[1] * ys
        xl += a2[0][0] - (L[0] + L[1] * a2[0][1])     # 底とつながるように端をそろえる
        out.append(np.c_[xl, ys])
    out.append(a2)
    ys = np.arange(0, a2[-1][1], 6.0)[::-1]
    if R is not None:
        xr = R[0] + R[1] * ys
        xr += a2[-1][0] - (R[0] + R[1] * a2[-1][1])
        out.append(np.c_[xr, ys])
    return np.vstack(out)


def snap_rim(ref, rim, search=10, step=4.0, smooth_penalty=2.0):
    """縁取りの線を、近くの本当の境目（外側の暗い樹脂 -> 内側の明るい縁取り）に細かく吸い付け、
    縁取りの太さも測る。戻り値: (点の並び, 縁取りの太さ[px])"""
    pts = np.array(rim, np.float64)
    seg = [np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / step)), endpoint=False) for a, b in zip(pts[:-1], pts[1:])]
    c = _smooth(np.vstack(seg + [pts[-1:]]), 9)
    t = np.gradient(c, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    poly = np.array([(c[0][0], 0)] + c.tolist() + [(c[-1][0], 0)], np.float32).reshape(-1, 1, 2)
    flip = np.array([cv2.pointPolygonTest(poly, (float(x + 6 * u), float(y + 6 * v)), False) < 0
                     for (x, y), (u, v) in zip(c, n)])
    n[flip] *= -1                                         # 法線を開口部の内側へ向ける
    img = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), 1.5)

    def val(q):
        x = np.clip(q[:, 0], 0, ref.shape[1] - 1).astype(np.float32)
        y = np.clip(q[:, 1], 0, ref.shape[0] - 1).astype(np.float32)
        return cv2.remap(img, x.reshape(1, -1), y.reshape(1, -1), cv2.INTER_LINEAR).ravel()

    ds = np.arange(-search, search + 1)
    gain = np.clip(np.stack([val(c + (d + 2) * n) - val(c + (d - 2) * n) for d in ds], 1), -40, 80)
    d_idx, _ = _dp_path(gain, max_jump=1, penalty=smooth_penalty)
    out = _smooth(c + ds[d_idx][:, None] * n, 5)

    # 縁取りの太さ: 外側の境目から内側へ進み、明るい縁取りが終わって暗くなる所までの距離
    ws = np.arange(4, 46)
    fall = np.stack([val(out + (w + 2) * n) - val(out + (w - 2) * n) for w in ws], 1)
    strong = fall.min(1) < -15
    width = float(np.median(ws[np.argmin(fall, 1)][strong])) if strong.any() else 16.0
    return [(float(x), float(y)) for x, y in out], width


def find_ribs(frame, left_leg, right_leg, y_from):
    """U字の脚のすぐ外側（gap〜reach px）で、一番はっきりした縦の線を縦リブとする。"""
    H, W = frame.shape
    g = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 2)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0))
    prof = gx[int(min(H - 20, y_from)):, :].mean(0)
    prof = np.convolve(prof, np.ones(5) / 5, mode="same")
    gap, reach = 6, max(50.0, 0.3 * (right_leg - left_leg) / 2)
    lo, hi = int(max(4, left_leg - reach)), int(max(5, left_leg - gap))
    lx = lo + int(np.argmax(prof[lo:hi])) if hi > lo else int(max(2, left_leg - gap))
    lo, hi = int(min(W - 5, right_leg + gap)), int(min(W - 4, right_leg + reach))
    rx = lo + int(np.argmax(prof[lo:hi])) if hi > lo else int(min(W - 3, right_leg + gap))
    return float(lx), float(rx)


def detect_geometry(frame, edge_margin=10, min_thick=30):
    """1枚の画像（基準座標 960x660）から検査範囲を作る。戻り値: (検査範囲, はっきり度)"""
    W, H = SIZE
    rim0, (cx, cy), total = find_rim(frame)
    rim0 = regularize(rim0, cx, cy)                            # 綺麗なU字に直す
    rim, width = snap_rim(frame, rim0.tolist(), search=5, smooth_penalty=4.0)
    rim = regularize(np.array(rim), cx, cy).tolist()          # 吸い付けたあとも、もう一度U字にそろえる
    rim = np.array(rim)
    bottom = rim[np.argmax(rim[:, 1])]
    top_y = float(rim[:, 1].min())
    mid = rim[(rim[:, 1] > top_y + 0.2 * (bottom[1] - top_y)) & (rim[:, 1] < bottom[1] - 0.2 * (bottom[1] - top_y))]
    mid = mid if len(mid) else rim
    lx, rx = find_ribs(frame, mid[:, 0].min(), mid[:, 0].max(), bottom[1] - 0.3 * (bottom[1] - top_y))
    half = (rim[:, 0].max() - rim[:, 0].min()) / 2
    gw = max(18.0, 0.12 * half)                               # ゲートの切り欠きの範囲（U字の大きさに比例）
    g = dict(
        y_eval=int(max(top_y + 10, cy - 0.15 * H)),        # 脚の一番上（ふちの近く）は検査しない edge_margin=edge_margin, min_thick=min_thick, rim_width=width,
        opening=[(float(rim[0][0]), 0.0)] + [tuple(map(float, p)) for p in rim] + [(float(rim[-1][0]), 0.0)],
        plate=[(lx, 0.0), (rx, 0.0), (rx, H - 1.0), (lx, H - 1.0)],
        rim_ignore=[(float(bottom[0] - gw), float(bottom[1] - 15), float(bottom[0] + gw), float(bottom[1] + 60))],
        outer=[[(lx + 2, 0.0), (lx + 2, H - 1.0)], [(rx - 2, 0.0), (rx - 2, H - 1.0)], [(lx, H - 3.0), (rx, H - 3.0)]],
        center=(float(cx), float(cy)), score=float(total),
    )
    return g, float(total)


def draw_geometry(frame, g):
    """検査範囲を画像に重ねた確認用の画像（緑 = 縁取りの外側、赤 = 縦リブと下端、青 = 探す範囲）。"""
    m = build_masks(g)
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    over = vis.copy()
    over[m["band"]] = (200, 80, 0)
    vis = cv2.addWeighted(over, 0.25, vis, 0.75, 0)
    vis[m["inner"]] = (0, 220, 0)
    vis[m["outer"]] = (0, 0, 255)
    for x0, y0, x1, y1 in g.get("rim_ignore", []):
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 255), 1)
    return vis
