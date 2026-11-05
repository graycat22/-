"""python -m crack_v14.cli --input DIR --out DIR [--screen]

Writes per-image overlay, a grid of all results, and results.csv
(img, kind, mode, rank, ratio, max_gap, ng, bbox in reference frame)."""
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
    ap.add_argument("--ref-dir", default="data/refs", help="ref_B.png / ref_D.png (reference frame)")
    ap.add_argument("--screen", action="store_true", help="inputs are phone photos of the monitor")
    a = ap.parse_args()
    refs = {k: cv2.imread(os.path.join(a.ref_dir, f"ref_{k}.png"), 0) for k in GEOM}
    ins = Inspector(refs)
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(a.input, f"*.{e}")))
    rows, tiles = [], []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        r = ins.inspect(cv2.cvtColor(load(f, a.screen), cv2.COLOR_BGR2GRAY))
        if r["judgment"] == "ERROR":
            print(name, "ERROR", r.get("reason")); continue
        vis = cv2.cvtColor(r["warped"], cv2.COLOR_GRAY2BGR)
        msg = f"{name:24s} {r['kind']} {r['judgment']:9s}"
        for t, col in (("D", (0, 0, 255)), ("B", (255, 200, 0))):
            for i, q in enumerate(r["res"].get(t, [])):
                p = q["path"][:, ::-1].astype(np.int32)
                (x0, y0), (x1, y1) = p.min(0), p.max(0)
                if q["ng"]:
                    cv2.polylines(vis, [p], False, col, 3)
                    cv2.rectangle(vis, (x0 - 8, y0 - 8), (x1 + 8, y1 + 8), col, 1)
                rb = q.get("rim_break")
                msg += (f" | {t}{i} r={q['ratio']:.2f} gap={q['max_gap']:.0f}"
                        + (f" rim={rb:.2f}" if rb is not None else "") + (" NG" if q["ng"] else ""))
                rows.append([name, r["kind"], t, i, round(q["ratio"], 3), round(q["max_gap"], 1),
                             "" if rb is None else round(rb, 3), q["ng"], x0, y0, x1, y1])
        print(msg)
        cv2.putText(vis, f"{name} {r['judgment']}", (10, 40), 0, 1.2, (0, 255, 255), 3)
        cv2.imwrite(os.path.join(a.out, f"{name}.jpg"), vis)
        tiles.append(cv2.resize(vis, (480, 330)))
    if tiles:
        while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(a.out, "result_grid.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img", "kind", "mode", "rank", "ratio", "max_gap", "rim_break", "ng", "x1", "y1", "x2", "y2"])
        w.writerows(rows)

if __name__ == "__main__":
    main()
