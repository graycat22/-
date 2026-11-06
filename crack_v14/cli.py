"""検査の実行。

  python -m crack_v14.cli --input <画像フォルダ> --template template_table1.png --ref-dir refs --out out

出力（out/）:
  <画像名>.jpg     入力画像に、見つけた割れの経路を描いたもの（水色 = B 明るい割れ、赤 = D 暗い割れ）
  result_grid.jpg  全画像の一覧
  judgments.csv    画像ごとの判定（img, judgment, type, n_defects）
  results.csv      割れ候補ごとの数値（img, kind, type, rank, ratio, max_gap, inner_gap, outer_gap,
                   rim_break, reach, ng, x1, y1, x2, y2）  座標は入力画像の座標
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

from .inspect_v14 import Inspector
from .screen_crop import crop_screen

COLORS = {"B": (255, 200, 0), "D": (0, 0, 255)}   # BGR: B は水色、D は赤


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="検査する画像のフォルダ")
    ap.add_argument("--template", required=True, help="template_table1.png")
    ap.add_argument("--ref-dir", default="refs", help="make_ref で作った基準フォルダ")
    ap.add_argument("--out", default="out")
    ap.add_argument("--screen", action="store_true", help="（テスト用）入力が画面をスマホで撮った写真のとき")
    a = ap.parse_args()

    template = cv2.imread(a.template, 0)
    if template is None:
        raise SystemExit(f"テンプレート画像が読めません: {a.template}")
    ins = Inspector(a.ref_dir, template)
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(a.input, f"*.{e}")))

    rows, summary, tiles = [], [], []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        img = cv2.imread(f)
        if a.screen:
            img = crop_screen(img)
        r = ins.inspect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if r["judgment"] == "ERROR":
            print(f"{name:24s} ERROR {r['reason']}")
            summary.append([name, "ERROR", "", 0])
            continue

        # 1行の表示: 種類、背面光の有無、テンプレート一致度、判定、候補ごとの数値
        msg = (f"{name:24s} kind={r['kind']} backlit={'Y' if r['backlit'] else 'N'} "
               f"tpl={r['template_score']:.2f} {r['judgment']:2s} {r['detail']:4s}")
        vis = img.copy()
        lw = max(2, round(vis.shape[1] / 320))           # 画像の大きさに合わせた線の太さ
        for t in ("D", "B"):
            for i, q in enumerate(r["res"].get(t, [])):
                if q["ng"]:
                    cv2.polylines(vis, [q["path_img"].astype(np.int32)], False, COLORS[t], lw)
                rb, rc = q.get("rim_break"), q.get("reach")
                msg += (f" | {t}{i} r={q['ratio']:.2f} gap={q['max_gap']:.0f}"
                        f" in={q['inner_gap']:.0f} out={q['outer_gap']:.0f}"
                        + (f" rim={rb:.2f}" if rb is not None else "")
                        + (f" reach={rc:.2f}" if rc is not None else "")
                        + (" NG" if q["ng"] else ""))
                (x1, y1), (x2, y2) = q["path_img"].min(0).astype(int), q["path_img"].max(0).astype(int)
                rows.append([name, r["kind"], t, i, round(q["ratio"], 3), round(q["max_gap"], 1),
                             round(q["inner_gap"], 1), round(q["outer_gap"], 1),
                             "" if rb is None else round(rb, 3), "" if rc is None else round(rc, 3),
                             q["ng"], x1, y1, x2, y2])
        print(msg)
        summary.append([name, r["judgment"], r["detail"], len(r["defects"])])

        cv2.putText(vis, f"{name} {r['judgment']}", (10, 14 * lw), 0, 0.4 * lw, (0, 255, 255), lw)
        cv2.imwrite(os.path.join(a.out, f"{name}.jpg"), vis)
        t_ = cv2.resize(vis, (480, round(480 * vis.shape[0] / vis.shape[1])))
        tile = np.zeros((360, 480, 3), np.uint8)
        tile[: min(360, t_.shape[0])] = t_[:360]
        tiles.append(tile)

    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(a.out, "result_grid.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img", "kind", "type", "rank", "ratio", "max_gap", "inner_gap", "outer_gap",
                    "rim_break", "reach", "ng", "x1", "y1", "x2", "y2"])
        w.writerows(rows)
    with open(os.path.join(a.out, "judgments.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img", "judgment", "type", "n_defects"])
        w.writerows(summary)


if __name__ == "__main__":
    main()
