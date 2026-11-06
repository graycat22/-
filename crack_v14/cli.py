"""検査の実行。

  python -m crack_v14.cli --input <画像フォルダ> --out out
  python -m crack_v14.cli --input <動画ファイル> --out out --every 1     # 動画なら1秒ごとに1コマ検査

基準画像・テンプレート・設定ファイルは不要（U字は1枚ごとに自動で見つける）。

出力（out/）:
  <画像名>.jpg     入力画像に、見つけた割れの経路を描いたもの（水色 = B 明るい割れ、赤 = D 暗い割れ）
                   細い緑の線 = 見つけた U字の縁取り、細い黄色の線 = 見つけた縦リブ（確認用）
  result_grid.jpg  全画像の一覧
  judgments.csv    画像ごとの判定（img, judgment, type, n_defects）
  results.csv      割れ候補ごとの数値（img, type, rank, ratio, max_gap, inner_gap, outer_gap,
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


VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v")


def iter_inputs(path, every):
    """検査する画像を (名前, 画像) の順に返す。

    path が画像フォルダなら中の画像を全部、動画ファイルなら every 秒ごとに1コマを取り出す
    （動画のコマの名前は「動画名_t012.5s」のように、何秒目かが分かるようにする）。
    """
    if os.path.isfile(path) and path.lower().endswith(VIDEO_EXT):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit(f"動画が開けません: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * every)))
        base = os.path.splitext(os.path.basename(path))[0]
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield f"{base}_t{i / fps:06.1f}s", img
            i += 1
        cap.release()
        return
    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(path, f"*.{e}")))
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            yield os.path.splitext(os.path.basename(f))[0], img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="検査する画像のフォルダ、または動画ファイル")
    ap.add_argument("--every", type=float, default=1.0, help="動画のとき、何秒ごとに1コマ検査するか")
    ap.add_argument("--out", default="out")
    ap.add_argument("--template", default=None, help="（互換のため受け付けるが、使わない）")
    ap.add_argument("--ref-dir", default=None, help="（互換のため受け付けるが、使わない）")
    ap.add_argument("--screen", action="store_true", help="（テスト用）入力が画面をスマホで撮った写真のとき")
    a = ap.parse_args()

    ins = Inspector()
    os.makedirs(a.out, exist_ok=True)
    items = list(iter_inputs(a.input, a.every))

    rows, summary, tiles = [], [], []
    for name, img in items:
        if a.screen:
            img = crop_screen(img)
        r = ins.inspect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if r["judgment"] == "ERROR":
            print(f"{name:24s} ERROR {r['reason']}")
            summary.append([name, "ERROR", "", 0])
            continue

        # 1行の表示: 種類、背面光の有無、テンプレート一致度、判定、候補ごとの数値
        msg = (f"{name:24s} backlit={'Y' if r['backlit'] else 'N'} "
               f"u={r['u_score']:.0f} {r['judgment']:2s} {r['detail']:4s}")
        vis = img.copy()
        lw = max(2, round(vis.shape[1] / 320))           # 画像の大きさに合わせた線の太さ
        # 見つけた U字と縦リブ（細い線、確認用）
        cv2.polylines(vis, [r["u_img"]["rim"].astype(np.int32)], False, (0, 200, 0), max(1, lw // 2))
        for line in r["u_img"]["ribs"]:
            cv2.polylines(vis, [line.astype(np.int32)], False, (0, 220, 255), max(1, lw // 2))
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
                rows.append([name, t, i, round(q["ratio"], 3), round(q["max_gap"], 1),
                             round(q["inner_gap"], 1), round(q["outer_gap"], 1),
                             "" if rb is None else round(rb, 3), "" if rc is None else round(rc, 3),
                             q["ng"], x1, y1, x2, y2])
        print(msg)
        summary.append([name, r["judgment"], r["detail"], len(r["defects"])])

        cv2.putText(vis, f"{name} {r['judgment']}", (10, 14 * lw), 0, 0.4 * lw, (0, 255, 255), lw)
        cv2.imwrite(os.path.join(a.out, f"{name}.jpg"), vis)
        if a.screen and len(summary) == 1:
            print(f"  （確認用）1枚目の切り出し結果: {os.path.join(a.out, name + '.jpg')} を見て、画面が正しく切り出せているか確認してください")
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
        w.writerow(["img", "type", "rank", "ratio", "max_gap", "inner_gap", "outer_gap",
                    "rim_break", "reach", "ng", "x1", "y1", "x2", "y2"])
        w.writerows(rows)
    n_ng = sum(1 for r in summary if r[1] == "NG")
    n_err = sum(1 for r in summary if r[1] == "ERROR")
    print(f"\n合計 {len(summary)} 枚: NG {n_ng} / OK {len(summary) - n_ng - n_err} / ERROR {n_err}")
    if n_ng:
        print("NG になった画像: " + " ".join(r[0] for r in summary if r[1] == "NG"))
    with open(os.path.join(a.out, "judgments.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img", "judgment", "type", "n_defects"])
        w.writerows(summary)


if __name__ == "__main__":
    main()
