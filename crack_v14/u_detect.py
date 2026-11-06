"""1枚の画像から、U字の開口部・縁取り・縦リブを自動で見つける（製品の種類を問わない）。

  1. find_opening : 開口部（縁取りの内側の穴）を領域として切り出す（ウォーターシェッド法）
                    真ん中の上の方 = 必ず U字の中、下端と左右の端 = 必ず外、として、
                    輪郭の一番強い尾根（= 縁取り）で2つに分ける。ネジ筒は U字の中にあるので混ざらない
  2. fit_u_to_mask: その輪郭に「左右対称のまっすぐな脚 + なめらかな底」の綺麗なU字を当てはめる
                    （画角が固定で、カセットは必ず綺麗なU字に写るため）
  3. snap_rim     : 縁取りの太さを測る
  4. find_ribs    : U字の脚のすぐ外側で一番はっきりした縦の線を縦リブとする
  ゲートの切り欠きは U字の一番下（6時の位置）として、縁取りの途切れ判定から外す
"""
import cv2
import numpy as np

from .geometry import SIZE, build_masks



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


def u_model(cx, w, cy, ry, sq, n_arc=120, leg_step=6.0):
    """綺麗なU字の線（左右対称）。左の脚の上 -> 底 -> 右の脚の上 の順の点の並び。

      cx : 左右の中心   w : 脚の間の半分の幅   cy : 脚が終わって底のカーブが始まる高さ
      ry : カーブの深さ   sq : カーブの角ばり具合（2 = 楕円、大きいほど底が平らで角ばる）
    """
    a = np.linspace(np.pi, 0, n_arc)
    co, si = np.cos(a), np.sin(a)
    e = 2.0 / sq
    arc = np.c_[cx + w * np.sign(co) * np.abs(co) ** e, cy + ry * np.abs(si) ** e]
    ys = np.arange(0, cy, leg_step)
    left = np.c_[np.full(len(ys), cx - w), ys]
    right = np.c_[np.full(len(ys), cx + w), ys[::-1]]
    return np.vstack([left, arc, right])


def find_opening(frame):
    """U字の開口部（縁取りの内側の穴）を、領域として切り出す（ウォーターシェッド法）。

    考え方:
      画像の真ん中の上の方（ここは必ず U字の中）を「中」の種、
      画像の下端と左右の端（ここは必ず U字の外）を「外」の種にして、
      輪郭の強さ（明るさの変化）を地形とみなし、低い所から水を満たすように両方の領域を広げる。
      2つの領域がぶつかるのは一番高い尾根 = U字の縁取りになる。
      ネジ筒などは U字の中にあり、縁取りの尾根を越えないと外から届かないので、必ず「中」になる。
    そのあと、明るい割れに沿って細く漏れた部分を、太い円で削って取り除く。
    """
    H, W = frame.shape
    f = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 2)
    gm = cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1))
    gm = np.clip(gm / (np.percentile(gm, 99) + 1e-6) * 255, 0, 255).astype(np.uint8)
    mk = np.zeros((H, W), np.int32)
    mk[int(0.12 * H):int(0.30 * H), int(0.49 * W):int(0.51 * W)] = 1          # 中の種
    mk[int(0.97 * H):, :] = 2; mk[:, :int(0.03 * W)] = 2; mk[:, int(0.97 * W):] = 2   # 外の種
    ws = cv2.watershed(cv2.cvtColor(gm, cv2.COLOR_GRAY2BGR), mk)
    op = (ws == 1).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    op = cv2.morphologyEx(op, cv2.MORPH_OPEN, k)                  # 細い漏れ（割れに沿った部分）を削る
    n, lab = cv2.connectedComponents(op)
    if n > 2:                                                     # 種を含む一番大きい塊だけ残す
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        op = (lab == sizes.argmax()).astype(np.uint8)
    return op.astype(bool)


def fit_u_to_mask(op):
    """開口部の領域の輪郭に、綺麗なU字の形（u_model）を当てはめる。

    輪郭の点（画像の上端の部分は除く）から U字までの距離が小さくなるように形を動かす。
    外れた点（割れの漏れ残りなど）に引っ張られないよう、距離の大きい 15% の点は数えない。
    """
    H, W = op.shape
    cnts, _ = cv2.findContours(op.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    pts = pts[pts[:, 1] > 0.06 * H][::2]                          # 上端（開口部が画像の外へ続く所）は除く

    def cost(p):
        c = u_model(*p, n_arc=120, leg_step=4.0)
        d = np.min(np.hypot(pts[:, None, 0] - c[None, :, 0], pts[:, None, 1] - c[None, :, 1]), axis=1)
        d = np.sort(d)[: int(0.85 * len(d))]
        return float(d.mean())

    xl, xr = np.percentile(pts[:, 0], 2), np.percentile(pts[:, 0], 98)
    yb = pts[:, 1].max()
    cx, w = (xl + xr) / 2, (xr - xl) / 2
    best = None
    for kr in (0.4, 0.55, 0.7, 0.85, 1.0, 1.2):                   # 底のカーブの深さの候補
        for sq in (2.0, 2.6, 3.4):
            p = np.array([cx, w, yb - kr * w, kr * w, sq])
            c_ = cost(p)
            if best is None or c_ < best[0]:
                best = (c_, p)
    cur, p = best
    for st in (6.0, 2.0, 0.5):                                    # 細かく寄せる
        improved = True
        while improved:
            improved = False
            for i, d in ((0, st), (1, st), (2, st), (3, st), (4, st * 0.08)):
                for sgn in (1, -1):
                    q = p.copy(); q[i] += sgn * d
                    if q[4] < 1.6 or q[4] > 6 or q[3] < 10:
                        continue
                    c_ = cost(q)
                    if c_ < cur - 1e-6:
                        p, cur, improved = q, c_, True
    return p, cur


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


def find_plate_edges(frame, rim, y_from):
    """プレートの左右の端（割れの終点）と、カセットの外側の縁（赤い枠）を探す。

    U字の脚から外へ向かって明るさを見ると、どの型でも
        プレート（明るめ） -> [段差で暗く] 溝 -> [段差で明るく] リブのハイライト -> [段差で暗く] リブの外側
    の順に並んでいる。
      プレートの端 = 脚とリブのハイライトの間で、一番はっきり外へ向かって暗くなる所
                     （割れがここまで届けば「外縁まで割れた」とみなす。溝の中を這う線は数えない）
      外側の縁     = リブのハイライトの外側の、外へ向かって暗くなる所（確認画像の赤い枠）
    見つからないときは None を返す。
    """
    H, W = frame.shape
    f = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 2)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=5) / 16
    m = np.median(gx[int(min(H - 40, y_from)):H - 20], 0)          # 列ごとの、横方向の明るさの変化
    out = {}
    for side, leg in ((-1, rim[:, 0].min()), (1, rim[:, 0].max())):
        xs = np.arange(int(leg) + 12 * side, int(leg) + 140 * side, side)
        xs = xs[(xs > 2) & (xs < W - 3)]
        if len(xs) < 20:
            return None
        v = side * m[xs]                         # 外へ向かって明るくなると +、暗くなると -
        rise = int(np.argmax(v))                 # リブのハイライト
        if v[rise] < 8 or rise < 5:
            return None
        edge = int(np.argmin(v[:rise - 2]))      # プレートの端（ハイライトより内側で一番暗くなる所）
        after = v[rise:rise + 35]
        drop = rise + int(np.argmin(after)) if len(after) and after.min() < -8 else min(rise + 10, len(xs) - 1)
        out[side] = (float(xs[edge]), float(xs[drop]))
    return out[-1][0], out[1][0], out[-1][1], out[1][1]


def find_plate_bottom(frame, lx, rx, y_start):
    """プレートの下の端の高さ。プレートの幅いっぱいに、横一直線のはっきりした明るさの段差がある所。
    （カセットが画像の途中で終わっている撮り方用。見つからなければ画像の下端）"""
    H, W = frame.shape
    a, b = int(lx + 0.15 * (rx - lx)), int(rx - 0.15 * (rx - lx))
    if b - a < 20:
        return H - 1.0
    f = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 2)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=5) / 16
    strong = (np.abs(gy[:, a:b]) > 12).mean(1)                    # その行で、段差がある列の割合
    strong = np.convolve(strong, np.ones(5) / 5, mode="same")
    for y in range(int(y_start), H - 15):
        if strong[y] > 0.7:
            return float(y)
    return H - 1.0


def detect_geometry(frame, edge_margin=3, min_thick=40):
    """1枚の画像（基準座標 960x660）から検査範囲を作る。戻り値: (検査範囲, はっきり度)"""
    W, H = SIZE
    op = find_opening(frame)                                    # U字の開口部を領域として切り出す
    up, fit_err = fit_u_to_mask(op)                             # その輪郭に綺麗なU字を当てはめる
    cx, cy = up[0], up[2]
    total = 60.0 - fit_err                                      # はっきり度: 当てはまりが良いほど大きい
    rim = u_model(*up)
    # 開口部が実際に始まる高さ（カセットの上端が写っている型では、その上は背景なので検査しない）
    rows = np.flatnonzero(op[:, int(up[0] - 0.5 * up[1]):int(up[0] + 0.5 * up[1])].any(1))
    open_top = float(rows[0]) if len(rows) else 0.0
    _, width = snap_rim(frame, rim.tolist(), search=3, smooth_penalty=4.0)   # 縁取りの太さだけ測る
    rim = rim.tolist()
    rim = np.array(rim)
    bottom = rim[np.argmax(rim[:, 1])]
    top_y = float(rim[:, 1].min())
    mid = rim[(rim[:, 1] > top_y + 0.2 * (bottom[1] - top_y)) & (rim[:, 1] < bottom[1] - 0.2 * (bottom[1] - top_y))]
    mid = mid if len(mid) else rim
    y_from = bottom[1] - 0.3 * (bottom[1] - top_y)
    pe = find_plate_edges(frame, rim, y_from)
    if pe is not None:
        lx, rx, ox_l, ox_r = pe                                  # プレートの端（終点）と外側の縁（赤い枠）
    else:
        lx, rx = find_ribs(frame, mid[:, 0].min(), mid[:, 0].max(), y_from)
        ox_l, ox_r = lx, rx
    yb = find_plate_bottom(frame, lx, rx, bottom[1] + 60)          # プレートの下の端
    half = (rim[:, 0].max() - rim[:, 0].min()) / 2
    gw = max(18.0, 0.12 * half)                               # ゲートの切り欠きの範囲（U字の大きさに比例）
    g = dict(
        y_eval=int(open_top + 70),                         # 起点は y_eval-60 から = 開口部の一番上から 10px 下まで検査する
        edge_margin=edge_margin, min_thick=min_thick, rim_width=width,
        opening=[(float(rim[0][0]), 0.0)] + [tuple(map(float, p)) for p in rim] + [(float(rim[-1][0]), 0.0)],
        plate=[(lx, 0.0), (rx, 0.0), (rx, yb), (lx, yb)],
        rim_ignore=[(float(bottom[0] - gw), float(bottom[1] - 15), float(bottom[0] + gw), float(bottom[1] + 60))],
        outer=[[(lx + 2, 0.0), (lx + 2, yb)], [(rx - 2, 0.0), (rx - 2, yb)], [(lx, yb - 2.0), (rx, yb - 2.0)]],
        center=(float(cx), float(cy)), score=float(total),
        frame_box=(float(ox_l), float(ox_r)),
    )
    return g, float(total)


def draw_geometry(frame, g):
    """検査範囲を画像に重ねた確認用の画像（緑 = 起点、青 = 探す範囲、オレンジ = 終点、赤 = カセットの外側の縁）。"""
    m = build_masks(g)
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    over = vis.copy()
    over[m["band"]] = (200, 80, 0)
    vis = cv2.addWeighted(over, 0.25, vis, 0.75, 0)
    vis[m["inner"]] = (0, 220, 0)
    vis[m["outer"]] = (0, 165, 255)                  # オレンジ: 割れの終点（プレートの端）
    if "frame_box" in g:                             # 赤: カセットの外側の縁
        a_, b_ = g["frame_box"]
        yb_ = int(g["plate"][2][1])
        cv2.rectangle(vis, (int(a_), 1), (int(b_), yb_), (0, 0, 255), 2)
    for x0, y0, x1, y1 in g.get("rim_ignore", []):
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 255), 1)
    return vis
