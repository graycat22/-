"""OK画像のフォルダから基準画像を作る（製品の種類は1枚ずつ自動で振り分ける）。

python -m crack_v14.make_ref --ok-dir pos1ok --out refs

  1. 各OK画像を仮の基準画像（data/refs/ref_type1.png, ref_type2.png）に位置合わせして、種類を判定
  2. 種類ごとに、先頭の1枚を軸にして残りをその1枚に位置合わせ（実画像どうしなので正確）
  3. 基準の座標（geometry.py と同じ 960x660）に揃えて画素ごとの中央値 -> refs/ref_type1.png など
  4. refs/check_type1.jpg などに検査範囲を重ねて出力（緑=内縁の外側、赤=縦リブ/下端、青=探す範囲）
  OKの中央値・ばらつき（ok_median_*, ok_mad_*）も保存する。

仮の基準画像が無いときは --kind type1 を付けると、先頭のOK画像を縮小してそのまま基準にする
（全部その種類とみなす。線がずれたら --move-geom / --scale-geom で調整）。
"""
import argparse, glob, os
from collections import defaultdict
import cv2, numpy as np
from .geometry import SIZE, GEOM, build_masks
from .register import Reference, register, refine_ecc, features


def load_gray(p):
    g = cv2.imread(p, 0)
    if g is None:
        raise SystemExit(f"読めない画像: {p}")
    return g


def small_of(g):
    f = min(1.0, 1.3 * max(SIZE) / max(g.shape))
    return (cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else g), np.diag([f, f, 1.0])


def build_one(kind, items, out, adjust):
    """items: [(path, gray, H_to_ref)]  先頭が軸"""
    m = build_masks(kind)
    cas = cv2.dilate((m["plate"] | m["opening"]).astype(np.uint8), np.ones((41, 41), np.uint8))
    p0, g0, H0 = items[0]
    H0 = adjust @ H0
    gs0, S0 = small_of(g0)
    anchor = Reference("anchor", gs0)
    stack = [cv2.warpPerspective(g0, H0, SIZE)]
    print(f"[{kind}] 軸: {os.path.basename(p0)}")
    for p, g, _ in items[1:]:
        gs, Sg = small_of(g)
        H, n = register(gs, anchor, feats=features(gs))
        if H is None or n < 30:
            print(f"[{kind}] {os.path.basename(p)}: 除外 (inliers={n})"); continue
        Hf = H0 @ np.linalg.inv(S0) @ H @ Sg
        Hf, cc = refine_ecc(g, stack[0], Hf, cas)
        stack.append(cv2.warpPerspective(g, Hf, SIZE))
    st = np.stack(stack).astype(np.float32)
    med = np.median(st, axis=0)
    mad = np.median(np.abs(st - med), axis=0) * 1.4826
    ref = np.clip(med, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out, f"ref_{kind}.png"), ref)
    cv2.imwrite(os.path.join(out, f"ok_median_{kind}.png"), ref)
    np.save(os.path.join(out, f"ok_mad_{kind}.npy"), mad)
    vis = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    over = vis.copy(); over[m["band"]] = (200, 80, 0)
    vis = cv2.addWeighted(over, 0.25, vis, 0.75, 0)
    vis[m["inner"]] = (0, 220, 0); vis[m["outer"]] = (0, 0, 255)
    cv2.imwrite(os.path.join(out, f"check_{kind}.jpg"), vis)
    print(f"[{kind}] {len(stack)} 枚から作成 -> {out}/ref_{kind}.png  （{out}/check_{kind}.jpg で線を確認）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ok-dir", required=True)
    ap.add_argument("--out", default="refs")
    ap.add_argument("--old-ref-dir", default="data/refs", help="仮の基準画像の場所")
    ap.add_argument("--kind", default=None, choices=sorted(GEOM), help="仮の基準画像なしで作るときの種類")
    ap.add_argument("--max", type=int, default=40, help="種類ごとに使う最大枚数")
    ap.add_argument("--move-geom", type=float, nargs=2, default=(0, 0), metavar=("DX", "DY"),
                    help="check画像の線を右にDX、下にDY px動かす（960x660座標）")
    ap.add_argument("--scale-geom", type=float, default=1.0, help="線を中心から何倍に広げるか（例 1.03）")
    a = ap.parse_args()
    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(a.ok_dir, f"*.{e}")))
    if not files:
        raise SystemExit(f"画像がありません: {a.ok_dir}")
    os.makedirs(a.out, exist_ok=True)

    dx, dy = a.move_geom; sc = a.scale_geom; cx, cy = SIZE[0] / 2, SIZE[1] / 2
    adjust = np.array([[1 / sc, 0, cx - cx / sc - dx / sc], [0, 1 / sc, cy - cy / sc - dy / sc], [0, 0, 1]])

    groups = defaultdict(list)
    old_paths = {k: os.path.join(a.old_ref_dir, f"ref_{k}.png") for k in GEOM}
    if a.kind is None:
        missing = [p for p in old_paths.values() if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"仮の基準画像がありません: {missing}\n"
                             f"  --old-ref-dir で場所を指定するか、--kind type1 などで先頭のOK画像から作ってください")
        from .inspect_v14 import Inspector
        ins = Inspector({k: load_gray(p) for k, p in old_paths.items()})
        for f in files:
            g = load_gray(f)
            k, H, n = ins.classify_and_align(g)
            if H is None:
                print(f"{os.path.basename(f)}: 位置合わせ失敗のため除外"); continue
            if len(groups[k]) < a.max:
                groups[k].append((f, g, H))
        print("振り分け: " + ", ".join(f"{k}={len(v)}枚" for k, v in sorted(groups.items())))
    else:
        for f in files[: a.max]:
            g = load_gray(f); h, w = g.shape
            groups[a.kind].append((f, g, np.diag([SIZE[0] / w, SIZE[1] / h, 1.0])))
        print(f"仮の基準画像なし: 全部 {a.kind} とみなし、先頭のOK画像を {SIZE[0]}x{SIZE[1]} に縮小して基準にします")
    for k, items in sorted(groups.items()):
        build_one(k, items, a.out, adjust)
    for k in GEOM:
        if k not in groups:
            print(f"注意: {k} のOK画像がありませんでした（cli には ref_{k}.png も必要。仮の基準画像をコピーしてもよい）")


if __name__ == "__main__":
    main()
