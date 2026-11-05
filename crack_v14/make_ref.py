"""Build a reference image from your own OK images (e.g. pos1ok) in the existing reference frame.

python -m crack_v14.make_ref --ok-dir pos1ok --view B --out refs_new
python -m crack_v14.make_ref --ok-dir pos2ok --view D --out refs_new

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
    ap.add_argument("--view", required=True, choices=["B", "D"])
    ap.add_argument("--old-ref-dir", default="data/refs")
    ap.add_argument("--out", default="refs_new")
    ap.add_argument("--max", type=int, default=40, help="use at most this many OK images")
    a = ap.parse_args()
    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(a.ok_dir, f"*.{e}")))[: a.max]
    if not files:
        raise SystemExit(f"画像がありません: {a.ok_dir}")
    os.makedirs(a.out, exist_ok=True)
    m = build_masks(a.view)
    cas = cv2.dilate((m["plate"] | m["opening"]).astype(np.uint8), np.ones((41, 41), np.uint8))

    old = load_gray(os.path.join(a.old_ref_dir, f"ref_{a.view}.png"))
    first = load_gray(files[0])
    fs, S = small_of(first)
    H0, n0 = register(fs, Reference("old", old))
    if H0 is None:
        raise SystemExit("1枚目のOK画像を今の基準画像に位置合わせできませんでした（別の画像を先頭にしてください）")
    H0, _ = refine_ecc(first, old, H0 @ S, cas)
    print(f"{os.path.basename(files[0])}: aligned to old reference (inliers={n0})")

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
