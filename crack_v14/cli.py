"""python -m crack_v14.cli --input DIR --out DIR [--screen]

Writes per-image overlay, a grid of all results, and results.csv
(img, kind[製品の種類 type1/type2], type[B=明るい割れ/D=暗い割れ], rank, ratio, max_gap, rim_break, ng, bbox in input coords)."""
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
    ap.add_argument("--ref-dir", default="data/refs", help="ref_type1.png / ref_type2.png")
    ap.add_argument("--screen", action="store_true", help="inputs are phone photos of the monitor")
    ap.add_argument("--aligned", action="store_true", help="draw on the image warped to the reference (debug)")
    a = ap.parse_args()
    refs = {}
    for k in GEOM:
        rp = os.path.join(a.ref_dir, f"ref_{k}.png")
        g = cv2.imread(rp, 0) if os.path.exists(rp) else None
        if g is None:
            print(f"注意: {rp} が無いので、種類 {k} は検査しません")
        else:
            refs[k] = g
    if not refs:
        raise SystemExit(f"基準画像がありません: --ref-dir {a.ref_dir} に ref_type1.png / ref_type2.png を置いてください")
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
        msg = (f"{name:24s} kind={r['kind']} backlit={'Y' if r['backlit'] else 'N'} "
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
        w.writerow(["img", "kind", "type", "rank", "ratio", "max_gap", "rim_break", "ng", "x1", "y1", "x2", "y2"])   # bbox in input-image coords
        w.writerows(rows)
    with open(os.path.join(a.out, "judgments.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["img", "judgment", "evidence", "n_defects"]); w.writerows(summary)

if __name__ == "__main__":
    main()
