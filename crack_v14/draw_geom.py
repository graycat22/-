"""基準画像の上でクリックして、検査範囲の線を引くツール。種類ごとに1回だけやる。

使い方:
  python -m crack_v14.draw_geom --ref-dir refs              # 全部の種類を順番に
  python -m crack_v14.draw_geom --ref-dir refs --kind type1  # 1つだけ

画面の操作（どの手順でも共通）:
  左クリック = 点を打つ / 右クリック = 1つ戻す / Enter = 次の手順へ / Esc = やめる

手順:
  1. U字の縁取りの「外側の線」を、左の脚の上 -> 底 -> 右の脚の上 の順にたどってクリック（10〜20点くらい）
     大まかでよい。終わったあと、近くの縁取りの境目に自動で吸い付ける（±18px の範囲）
  2. 左の縦リブの線を、上と下の2点クリック
  3. 右の縦リブの線を、上と下の2点クリック
  4. ゲートの切り欠き（6時の位置の、縁取りが最初から切れている所）を囲む四角の、左上と右下をクリック
     （無ければそのまま Enter）
  5. 検査を始める高さを1点クリック（これより上の脚の部分は検査しない）

終わると geom_<種類>.json と check_<種類>.jpg（線を重ねた確認画像）が保存される。
"""
import argparse
import glob
import os

import cv2
import numpy as np

from .geometry import SIZE, build_masks, save_geom

STEPS = [
    ("rim", None, "1. U字の縁取りの外側を、左の脚の上 -> 底 -> 右の脚の上 の順にクリック", "1: rim outer edge, left top -> bottom -> right top"),
    ("lrib", 2, "2. 左の縦リブの線を、上と下の2点クリック", "2: left rib line (top, bottom)"),
    ("rrib", 2, "3. 右の縦リブの線を、上と下の2点クリック", "3: right rib line (top, bottom)"),
    ("gate", 2, "4. ゲートの切り欠きを囲む四角の左上と右下をクリック（無ければ Enter）", "4: gate notch box (2 clicks or Enter)"),
    ("top", 1, "5. 検査を始める高さを1点クリック", "5: inspection top (1 click)"),
]


def geom_from_clicks(rim, lrib, rrib, gate, top, edge_margin=10, min_thick=30):
    """クリックした点から検査範囲の定義を作る。"""
    W, H = SIZE

    def line_x_at(p, q, y):          # 2点を通る直線の、高さ y での x
        (x1, y1), (x2, y2) = p, q
        return x1 if y2 == y1 else x1 + (x2 - x1) * (y - y1) / (y2 - y1)

    # 縦リブは画像の上から下まで伸ばす
    l_top, l_bot = (line_x_at(*lrib, 0), 0), (line_x_at(*lrib, H - 1), H - 1)
    r_top, r_bot = (line_x_at(*rrib, 0), 0), (line_x_at(*rrib, H - 1), H - 1)
    # 開口部の多角形: クリックした縁取りの線を、両端とも画像の上端まで伸ばして閉じる
    opening = [(rim[0][0], 0)] + [tuple(p) for p in rim] + [(rim[-1][0], 0)]
    g = dict(
        y_eval=int(top[0][1]), edge_margin=edge_margin, min_thick=min_thick,
        opening=[(float(x), float(y)) for x, y in opening],
        plate=[l_top, r_top, r_bot, l_bot],
        rim_ignore=[(min(gate[0][0], gate[1][0]), min(gate[0][1], gate[1][1]),
                     max(gate[0][0], gate[1][0]), max(gate[0][1], gate[1][1]))] if len(gate) == 2 else [],
        outer=[[(l_top[0] + 2, 0), (l_bot[0] + 2, H - 1)],          # 左リブの少し内側
               [(r_top[0] - 2, 0), (r_bot[0] - 2, H - 1)],          # 右リブの少し内側
               [(l_bot[0], H - 3), (r_bot[0], H - 3)]],             # 画像の下端
    )
    return g


def snap_rim(ref, rim, search=18, step=4.0):
    """クリックした縁取りの線を、近くの「本当の縁取りの外側の境目」に吸い付ける。

    線の各点で、線に垂直な方向に ±search px の範囲を調べ、
    「外側（樹脂、暗い）から内側（縁取り、明るい）へ一番強く明るくなる所」に点を動かす。
    大まかになぞっただけでも、縁取りの境目にぴったり合うようにするため。
    """
    pts = np.array(rim, np.float64)
    # 点を step px おきに並べ直す
    seg = [np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / step)), endpoint=False) for a, b in zip(pts[:-1], pts[1:])]
    c = np.vstack(seg + [pts[-1:]])
    t = np.gradient(c, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    # 法線を開口部の内側に向ける（左の脚の上 -> 底 -> 右の脚の上 の順なら、内側は進行方向の左）
    poly = np.array([(c[0][0], 0)] + c.tolist() + [(c[-1][0], 0)], np.float32).reshape(-1, 1, 2)
    flip = np.array([cv2.pointPolygonTest(poly, (float(x + 6 * u), float(y + 6 * v)), False) < 0
                     for (x, y), (u, v) in zip(c, n)])
    n[flip] *= -1
    img = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), 1.5)

    def val(q):
        x = np.clip(q[:, 0], 0, ref.shape[1] - 1).astype(np.float32)
        y = np.clip(q[:, 1], 0, ref.shape[0] - 1).astype(np.float32)
        return cv2.remap(img, x.reshape(1, -1), y.reshape(1, -1), cv2.INTER_LINEAR).ravel()

    ds = np.arange(-search, search + 1)
    gain = np.stack([val(c + (d + 2) * n) - val(c + (d - 2) * n) for d in ds], 1)   # 内側へ向かう明るさの増え方
    best = ds[np.argmax(gain, axis=1)].astype(np.float64)
    best[gain.max(axis=1) < 8] = 0                      # はっきりした境目が無い所は動かさない
    k = 11                                              # 飛び跳ねを抑える（前後11点の中央値）
    pad = np.r_[[best[0]] * (k // 2), best, [best[-1]] * (k // 2)]
    best = np.array([np.median(pad[i:i + k]) for i in range(len(best))])
    out = c + best[:, None] * n
    out[0], out[-1] = c[0], c[-1]                       # 両端（脚の上）はクリックした所のまま
    return [(float(x), float(y)) for x, y in out]


def draw_check(ref, g):
    """検査範囲を基準画像に重ねた確認画像（緑 = 起点、赤 = 終点、青 = 探す範囲）。"""
    m = build_masks(g)
    vis = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    over = vis.copy()
    over[m["band"]] = (200, 80, 0)
    vis = cv2.addWeighted(over, 0.25, vis, 0.75, 0)
    vis[m["inner"]] = (0, 220, 0)
    vis[m["outer"]] = (0, 0, 255)
    y = g["y_eval"]
    cv2.line(vis, (0, y), (SIZE[0] - 1, y), (0, 200, 255), 1)      # 検査を始める高さ
    for x0, y0, x1, y1 in g.get("rim_ignore", []):
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 255), 1)
    return vis


def collect_clicks(ref, kind):
    """画面に基準画像を出して、手順ごとにクリックを集める。"""
    pts = {k: [] for k, *_ in STEPS}
    state = {"i": 0}
    win = f"draw_geom {kind}"

    def on_mouse(ev, x, y, *_):
        key, n, *_ = STEPS[state["i"]]
        if ev == cv2.EVENT_LBUTTONDOWN and (n is None or len(pts[key]) < n):
            pts[key].append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN and pts[key]:
            pts[key].pop()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(f"\n=== {kind} ===")
    print("左クリック = 点を打つ / 右クリック = 1つ戻す / Enter = 次へ / Esc = やめる")
    print(STEPS[0][2])
    while True:
        key, n, jp, en = STEPS[state["i"]]
        vis = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        colors = {"rim": (0, 220, 0), "lrib": (0, 0, 255), "rrib": (0, 0, 255), "gate": (0, 200, 255), "top": (255, 200, 0)}
        for k, ps in pts.items():
            for p in ps:
                cv2.circle(vis, p, 3, colors[k], -1)
            if len(ps) >= 2 and k != "gate":
                cv2.polylines(vis, [np.array(ps, np.int32)], False, colors[k], 1)
        cv2.rectangle(vis, (0, 0), (SIZE[0], 24), (0, 0, 0), -1)
        cv2.putText(vis, en, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:                                   # Esc
            cv2.destroyWindow(win)
            return None
        if k in (13, 10):                             # Enter
            ok = (len(pts[key]) >= 5) if n is None else (len(pts[key]) == n or key == "gate")
            if not ok:
                print("  点が足りません")
                continue
            state["i"] += 1
            if state["i"] == len(STEPS):
                cv2.destroyWindow(win)
                return pts
            print(STEPS[state["i"]][2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default="refs")
    ap.add_argument("--kind", default=None, help="線を引く種類（省略すると全部）")
    ap.add_argument("--no-snap", action="store_true", help="縁取りへの吸い付けをしない（クリックした所そのまま）")
    a = ap.parse_args()
    kinds = [a.kind] if a.kind else sorted(os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(a.ref_dir, "ref_*.png")))
    if not kinds:
        raise SystemExit(f"基準画像がありません: {a.ref_dir}（先に make_ref を実行してください）")
    for kind in kinds:
        ref = cv2.imread(os.path.join(a.ref_dir, f"ref_{kind}.png"), 0)
        pts = collect_clicks(ref, kind)
        if pts is None:
            print("中止しました"); return
        rim = pts["rim"] if a.no_snap else snap_rim(ref, pts["rim"])
        g = geom_from_clicks(rim, pts["lrib"], pts["rrib"], pts["gate"], pts["top"])
        save_geom(a.ref_dir, kind, g)
        cv2.imwrite(os.path.join(a.ref_dir, f"check_{kind}.jpg"), draw_check(ref, g))
        print(f"保存しました: {a.ref_dir}/geom_{kind}.json, check_{kind}.jpg")


if __name__ == "__main__":
    main()
