"""Build a reference image from your own OK images (e.g. pos1ok) in the existing reference frame.

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
