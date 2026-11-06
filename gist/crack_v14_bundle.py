"""crack_v14 一式（1ファイル版）。 python crack_v14_bundle.py で crack_v14/ フォルダを展開する。

展開したあと（基準画像・テンプレート・線引きは不要）:
  pip install -r requirements.txt
  python -m crack_v14.cli --input <画像フォルダ> --out out
  python -m crack_v14.evaluate --labels labels/defect_annotations.csv --pred out
詳しくは crack_v14/README.md
"""
import os
FILES = {}
FILES['crack_v14/__init__.py'] = r''''''
FILES['crack_v14/cli.py'] = r'''"""検査の実行。

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
'''
FILES['crack_v14/crossing.py'] = r'''"""「内縁から外縁まで、割れっぽい所だけを通って抜けられるか」を調べる部分。

考え方:
  割れらしさマップを「通りやすさ」にして（割れっぽい所ほど通りやすい）、
  U字の内縁（inner）から外縁（outer）までの一番通りやすい道を最短経路探索で求める。
  その道の上で「割れていない所」をどれだけ通らされたかで判定する。

  ratio     : 道の上で健全部を通った長さ ÷ その場所の健全な肉厚（0 = 端から端まで割れ、1 = 健全）
  max_gap   : 道の上で割れていない区間が一番長く続いた長さ [px]
  inner_gap : 割れが内側の端からどれだけ手前で止まっているか [px]
  outer_gap : 割れが外側の端からどれだけ手前で止まっているか [px]
"""
import cv2
import numpy as np

from .geodesic import MCP
from .geometry import build_masks

ON = 0.15   # 割れらしさがこれより大きい画素を「割れの上」とみなす


class Crossing:
    def __init__(self, geom, eps=0.03):
        """geom: 検査範囲の定義（u_detect.detect_geometry が作ったもの）。"""
        self.g, self.eps = geom, eps
        self.m = build_masks(geom)
        # 縁取りの太さ（u_detect が測ったもの）の分だけ内側まで調べる（それより内側は開口部の背景なので見ない）。
        # 割れが縁取りを内側の端まで切っているか（= U字の内側まで割れているか）を確かめるため
        rw = geom.get("rim_width")
        self._build_rim(geom["opening"], depth=16 if rw is None else max(8, int(round(rw))))
        y0 = geom["y_eval"]

        # 起点: 開口部の縁のすぐ外側の帯（U字の脚の上の方は除く）
        inner = self.m["inner"].copy()
        inner[: max(0, y0 - 60)] = False
        self.starts = np.argwhere(inner)

        # 歩ける範囲: 樹脂部分＋起点＋終点
        walk = self.m["band"] | self.m["inner"] | self.m["outer"] | self.m["rim"]
        walk[: max(0, y0 - 80)] = False
        self.walk = walk

        # 健全なときの肉厚（全部コスト1で歩いたときの距離）を終点ごとに測っておく
        cum0, _ = MCP(np.where(walk, 1.0, np.inf)).find_costs(self.starts)
        ends = np.argwhere(self.m["outer"])
        th = cum0[ends[:, 0], ends[:, 1]]
        # 起点を縁取りの内側に置いたときは、縁取りの太さぶん厚く測れるので、その分を足して比べる
        extra = (geom.get("rim_width") or 0) + 6 if self.m["rim"].any() else 0
        keep = (ends[:, 0] >= y0 - 60) & np.isfinite(th) & (th >= geom.get("min_thick", 60) + extra)
        self.ends, self.thick = ends[keep], th[keep]

        # プレートの端（縦リブの縁）は黒い線が必ずあるので、割れの証拠として数えない帯にする
        em = geom.get("edge_margin", 0)
        if em > 0:
            plate = self.m["plate"].astype(np.uint8)
            ez = ~cv2.erode(plate, np.ones((2 * em + 1,) * 2, np.uint8)).astype(bool)
            ez[-(em + 4):] = False          # 画像の下端はリブではないので除外しない
            self.edge_zone = ez
        else:
            self.edge_zone = np.zeros_like(self.m["plate"])

    # -------------------------------------------------------------- U字の縁取り
    def _build_rim(self, opening_poly, step=2.0, depth=16):
        """開口部の輪郭を 2px おきの点にして、各点で「開口部の内側へ向かう向き」を求めておく。"""
        p = np.array(opening_poly, np.float32)
        seg = [np.linspace(p[i], p[i + 1], max(2, int(np.hypot(*(p[i + 1] - p[i])) / step)), endpoint=False)
               for i in range(len(p) - 1)]
        c = np.vstack(seg)
        t = np.gradient(c, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)            # 輪郭に垂直な向き
        probe = (c + 4 * nrm).astype(int)
        probe[:, 0] = probe[:, 0].clip(0, self.m["opening"].shape[1] - 1)
        probe[:, 1] = probe[:, 1].clip(0, self.m["opening"].shape[0] - 1)
        inside = self.m["opening"][probe[:, 1], probe[:, 0]]
        nrm[~inside] *= -1                                # 開口部の内側を向くようにそろえる
        self.rim_pts, self.rim_nrm, self.rim_depth = c, nrm, depth

    def rim_profile(self, gray):
        """輪郭の各点で、縁取りを横切る方向（線の前後 8px）に明るさを測り、縁取りの明るさ（上位15%の値）を返す。"""
        d = np.arange(-8, 9)          # 縁取りの線（縁取りの上にある）の前後 8px
        xy = self.rim_pts[:, None, :] + d[None, :, None] * self.rim_nrm[:, None, :]
        x = xy[..., 0].clip(0, gray.shape[1] - 1).astype(np.float32)
        y = xy[..., 1].clip(0, gray.shape[0] - 1).astype(np.float32)
        v = cv2.remap(cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32), x, y, cv2.INTER_LINEAR)
        return np.percentile(v, 85, axis=1)

    def nearest_rim(self, xy):
        """点 xy に一番近い輪郭点の番号。"""
        return int(np.argmin(np.hypot(*(self.rim_pts - xy).T)))

    def rim_break(self, gray, start_xy, win=20, ctx=100):
        """縁取りの途切れ度（0 = 途切れていない、1 = 完全に切れている）。

        割れの起点付近（±win 点）の縁取りの明るさを、その前後（ctx 点の範囲）の明るさと比べる。
        本物の割れは樹脂が割れて開いているので、縁取りの白い光も切れる。傷や影では切れない。
        """
        R = self.rim_profile(gray)
        i = self.nearest_rim(start_xy)
        lo, hi = max(0, i - win), min(len(R), i + win + 1)
        a, b = slice(max(0, i - ctx), max(0, i - 2 * win)), slice(i + 2 * win, i + ctx)
        ref = np.r_[R[a], R[b]]
        if len(ref) < 5 or len(R[a]) < 3 or len(R[b]) < 3:
            return 0.0, i
        return float(1 - R[lo:hi].min() / (np.median(ref) + 1e-6)), i

    def rim_ignored(self, i):
        """輪郭点 i が、最初から縁取りが切れている場所（ゲートの切り欠き）にあるか。"""
        x, y = self.rim_pts[i]
        return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in self.g.get("rim_ignore", []))

    def reach_inside(self, prob, start_xy, half=3):
        """（B用）割れが縁取りを越えて開口部までつながっているか（0〜1、1 = 全部つながっている）。"""
        i = self.nearest_rim(start_xy)
        d = np.arange(0, self.rim_depth + 1)
        vals = []
        for j in range(max(0, i - half), min(len(self.rim_pts), i + half + 1)):
            xy = self.rim_pts[j][None, :] + d[:, None] * self.rim_nrm[j][None, :]
            x = xy[:, 0].clip(0, prob.shape[1] - 1).astype(int)
            y = xy[:, 1].clip(0, prob.shape[0] - 1).astype(int)
            vals.append((prob[y, x] > ON).mean())
        return (float(max(vals)) if vals else 0.0), i

    # -------------------------------------------------------------- 経路探索
    def run(self, prob):
        """割れらしさマップ prob から、内縁 -> 外縁の一番割れっぽい道を1本求める。"""
        prob = np.where(self.edge_zone, 0.0, prob)
        cost = np.where(self.walk, self.eps + (1.0 - prob), np.inf)   # 割れっぽい所ほど安い
        mcp = MCP(cost)
        cum, _ = mcp.find_costs(self.starts)
        vals = cum[self.ends[:, 0], self.ends[:, 1]]
        ratio = vals / np.maximum(self.thick, 1)
        k = int(np.argmin(ratio))                                     # 一番割れっぽく抜けられる終点
        path = np.array(mcp.traceback(tuple(self.ends[k])))           # (y, x) の並び、起点 -> 終点

        on = prob[path[:, 0], path[:, 1]] > ON
        step = np.r_[0, np.hypot(*np.diff(path, axis=0).T)]
        gap, cur = 0.0, 0.0
        for o, s in zip(on, step):                                   # 割れていない区間の最長
            cur = 0.0 if o else cur + s
            gap = max(gap, cur)
        length = step.sum()
        cum_len = np.cumsum(step)
        idx = np.flatnonzero(on)
        inner_gap = float(cum_len[idx[0]]) if len(idx) else float(length)
        outer_gap = float(length - cum_len[idx[-1]]) if len(idx) else float(length)
        return dict(ratio=float(ratio[k]), max_gap=float(gap), length=float(length), path=path,
                    inner_gap=inner_gap, outer_gap=outer_gap, prob=prob)

    def run_multi(self, prob, k=3, ratio_th=0.35, gap_th=25.0, sep=35, reject=None, max_iter=10):
        """割れが複数あるときのために、見つけた道の周り（sep px）を消して最大 k 本まで探す。

        reject(道) が True の道（U字の縁や縦リブに沿って走る溝・影）は割れの候補にせず、
        その道の線だけを細く（5px）消して探し直す。太く消すと、縁から始まる本物の割れの根元まで消えるため。
        """
        out, p = [], prob.copy()
        for _ in range(max_iter):
            r = self.run(p)
            r["ng"] = r["ratio"] <= ratio_th and r["max_gap"] <= gap_th
            line = [r["path"][:, ::-1].astype(np.int32)]
            if r["ng"] and reject is not None and reject(r):
                m = np.zeros(p.shape, np.uint8)
                cv2.polylines(m, line, False, 1, 5)
                p[m.astype(bool)] = 0.0
                continue
            out.append(r)
            if not r["ng"] or len(out) >= k:
                break
            m = np.zeros(p.shape, np.uint8)
            cv2.polylines(m, line, False, 1, 2 * sep + 1)
            p[m.astype(bool)] = 0.0
        if not out:
            out.append(r)
        return out
'''
FILES['crack_v14/evaluate.py'] = r'''"""検査結果を正解CSVと比べて、見逃し・過検出・位置の一致を数える。

  python -m crack_v14.evaluate --labels labels/defect_annotations.csv --pred out [--role VALIDATION]

正解CSVの列: record_type,path,dataset_role,label_status,type,judgment,severity,x1,y1,x2,y2,note
  - label_status が CONFIRMED の行だけ使う
  - その画像のどれかの行が judgment=NG なら、その画像は NG
  - x1〜y2 が入っている行は、正解の割れの位置（入力画像の座標）
予測: cli.py が出力した judgments.csv と results.csv
位置の一致: 予測した割れの範囲が、正解の範囲（--pad px 広げたもの）と重なれば一致
"""
import argparse, csv, os
from collections import defaultdict

def key(p):
    return os.path.splitext(os.path.basename(p))[0]

def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def overlap(a, b, pad):
    return not (a[2] < b[0] - pad or a[0] > b[2] + pad or a[3] < b[1] - pad or a[1] > b[3] + pad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pred", default="out")
    ap.add_argument("--role", default=None, help="TRAIN / VALIDATION / TEST (default: all)")
    ap.add_argument("--pad", type=float, default=20)
    a = ap.parse_args()

    gt_ng, gt_box, role = {}, defaultdict(list), {}
    for r in csv.DictReader(open(a.labels, encoding="utf-8-sig")):
        if r.get("label_status", "CONFIRMED").strip().upper() != "CONFIRMED":
            continue
        if a.role and r.get("dataset_role", "").strip().upper() != a.role.upper():
            continue
        k = key(r["path"]); role[k] = r.get("dataset_role", "")
        j = r.get("judgment", "").strip().upper()
        gt_ng[k] = gt_ng.get(k, False) or j == "NG"
        box = [num(r.get(c)) for c in ("x1", "y1", "x2", "y2")]
        if j == "NG" and None not in box:
            gt_box[k].append(box)

    pred = {r["img"]: r["judgment"] == "NG" for r in csv.DictReader(open(os.path.join(a.pred, "judgments.csv")))}
    pbox = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(a.pred, "results.csv"))):
        if r["ng"] == "True":
            pbox[r["img"]].append([float(r[c]) for c in ("x1", "y1", "x2", "y2")])

    tp = fp = fn = tn = located = loc_total = 0
    misses, false_alarms, mislocated = [], [], []
    for k, ng in sorted(gt_ng.items()):
        if k not in pred:
            print("not predicted:", k); continue
        p = pred[k]
        if ng and p:
            tp += 1
            if gt_box[k]:
                loc_total += 1
                if any(overlap(b, g, a.pad) for b in pbox[k] for g in gt_box[k]):
                    located += 1
                else:
                    mislocated.append(k)
        elif ng:
            fn += 1; misses.append(k)
        elif p:
            fp += 1; false_alarms.append(k)
        else:
            tn += 1
    n = tp + fp + fn + tn
    print(f"images={n}  TP={tp} FN={fn} FP={fp} TN={tn}")
    if tp + fn: print(f"NG recall (見逃し率の逆) = {tp / (tp + fn):.3f}")
    if fp + tn: print(f"OK false-alarm rate (過検出率) = {fp / (fp + tn):.3f}")
    if loc_total: print(f"location hit = {located}/{loc_total}")
    for name, lst in (("見逃し", misses), ("過検出", false_alarms), ("位置ずれ", mislocated)):
        if lst: print(f"{name}: {' '.join(lst)}")

if __name__ == "__main__":
    main()
'''
FILES['crack_v14/extract_frames.py'] = r'''"""動画から一定の間隔でコマを取り出して、JPEG 画像として保存する（チャットに貼る用）。

  python -m crack_v14.extract_frames IMG_8074.MOV frames            # 2秒ごと
  python -m crack_v14.extract_frames IMG_8074.MOV frames --every 1  # 1秒ごと

1枚あたり 100〜200KB くらいの、横 1080px の画像になる。
"""
import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--every", type=float, default=2.0, help="何秒ごとに1枚取り出すか")
    ap.add_argument("--width", type=int, default=1080, help="保存する画像の横幅 [px]")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.src)
    if not cap.isOpened():
        raise SystemExit(f"動画が開けません: {a.src}")
    os.makedirs(a.out_dir, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * a.every)))
    i = n = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            if img.shape[1] > a.width:
                img = cv2.resize(img, (a.width, int(img.shape[0] * a.width / img.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(a.out_dir, f"f{n:03d}_t{i / fps:05.1f}s.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            n += 1
        i += 1
    cap.release()
    print(f"{n} 枚 -> {a.out_dir}/")


if __name__ == "__main__":
    main()
'''
FILES['crack_v14/features.py'] = r'''"""割れらしさマップ（1画素ごとに「割れっぽさ」を 0〜1 の数値にする）。

  D（暗い割れ）: 周りより暗く、細長い所ほど高い
  B（明るい割れ）: 周りの樹脂より明るい所（割れ目から背面光が漏れている所）ほど高い
"""
import cv2
import numpy as np


def _robust_z(x, mask):
    """mask 内の典型的な値（中央値）から何倍ずれているかを測る（外れ値に強い標準化）。"""
    v = x[mask]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return (x - med) / mad


def dark_line_map(gray, band, scales=(1.5, 3.0, 5.0, 8.0), se_sizes=(13, 25, 37)):
    """D: 暗い割れ線らしさ。

    1) 相対的な暗さ: クロージング（暗い細い所を周りの明るさで埋めた画像）と元画像の差 ÷ 周りの明るさ。
       割れの幅は 3〜20px くらいまでばらつくので、大きさを変えて3回やって一番大きい値を使う。
    2) 細長さ: ヘッセ行列（明るさの曲がり方）の固有値の比。線なら1に近く、点やザラザラなら0に近い。
    1) × 2) が大きい所 = 周りより暗くて細長い所 = 割れ線っぽい所。
    """
    f = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    floor = 0.35 * np.percentile(f[band], 90)          # プレート下の暗い所でノイズを増幅しないための下限
    rel = np.zeros_like(f)
    for k in se_sizes:
        clo = cv2.morphologyEx(f, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        rel = np.maximum(rel, (clo - f) / np.maximum(clo, floor))
    elong = np.zeros_like(f)
    for s in scales:
        g = cv2.GaussianBlur(f, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)   # 横方向の2回微分
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)   # 縦方向の2回微分
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        t = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        l1 = 0.5 * (dxx + dyy + t)                       # 大きい方の固有値（線を横切る向きの曲がり）
        l2 = 0.5 * (dxx + dyy - t)                       # 小さい方の固有値（線に沿う向きの曲がり）
        e = np.where(l1 > 0, 1 - np.clip(np.abs(l2) / (l1 + 1e-6), 0, 1), 0)
        elong = np.maximum(elong, e)
    return rel * (0.3 + 0.7 * elong)


def bright_gap_map(gray, band):
    """B: 明るい割れらしさ。周りの樹脂の明るさより、どれだけ明るいか（中央値からのずれ）。"""
    f = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    bg = cv2.medianBlur(gray, 41).astype(np.float32)          # その周りの樹脂の明るさ
    th = f - np.minimum(bg, np.percentile(f[band], 60))
    return _robust_z(th, band)


def to_prob(x, lo, hi):
    """値 x を 0〜1 に直す（lo 以下は0、hi 以上は1、その間は直線）。"""
    return np.clip((x - lo) / (hi - lo), 0, 1)
'''
FILES['crack_v14/geodesic.py'] = r'''"""最短経路探索（ダイクストラ法）。画像の各画素を「通るのにかかる手間（コスト）」として、
起点の集まりから各画素まで、一番手間の少ない道とその手間の合計を求める。

  隣の画素への移動: 上下左右と斜めの8方向
  移動の手間     : (今の画素のコスト + 隣の画素のコスト) / 2 × 移動距離（斜めは √2）
  コストが inf   : 壁（通れない）
numba があれば高速版、無ければ純Python版（遅いが同じ結果）で動く。
"""
import heapq
import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

_DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int64)
_DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int64)
_DL = np.sqrt(_DY.astype(np.float64) ** 2 + _DX.astype(np.float64) ** 2)


def _dijkstra_py(cost, starts):
    h, w = cost.shape
    cum = np.full((h, w), np.inf); prev = np.full((h, w), -1, np.int64)
    pq = []
    for y, x in starts:
        if np.isfinite(cost[y, x]):
            cum[y, x] = 0.0; heapq.heappush(pq, (0.0, int(y), int(x)))
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > cum[y, x]:
            continue
        c0 = cost[y, x]
        for k in range(8):
            ny, nx = y + _DY[k], x + _DX[k]
            if 0 <= ny < h and 0 <= nx < w:
                c1 = cost[ny, nx]
                if not np.isfinite(c1):
                    continue
                nd = d + 0.5 * (c0 + c1) * _DL[k]
                if nd < cum[ny, nx]:
                    cum[ny, nx] = nd; prev[ny, nx] = y * w + x
                    heapq.heappush(pq, (nd, int(ny), int(nx)))
    return cum, prev


if _HAS_NUMBA:
    @numba.njit(cache=True)
    def _dijkstra_nb(cost, sy, sx, DY, DX, DL):
        h, w = cost.shape
        cum = np.full((h, w), np.inf)
        prev = np.full((h, w), -1, np.int64)
        # 「まだ調べていない画素」を手間の小さい順に取り出すための入れ物（二分ヒープ）
        cap = 1 << 16
        hd = np.empty(cap, np.float64); hi = np.empty(cap, np.int64); n = 0
        for i in range(sy.shape[0]):
            y, x = sy[i], sx[i]
            if np.isfinite(cost[y, x]) and cum[y, x] > 0.0:
                cum[y, x] = 0.0
                if n == cap:
                    cap *= 2; hd2 = np.empty(cap, np.float64); hi2 = np.empty(cap, np.int64)
                    hd2[:n] = hd[:n]; hi2[:n] = hi[:n]; hd = hd2; hi = hi2
                j = n; n += 1
                while j > 0 and hd[(j - 1) >> 1] > 0.0:
                    p = (j - 1) >> 1; hd[j] = hd[p]; hi[j] = hi[p]; j = p
                hd[j] = 0.0; hi[j] = y * w + x
        while n > 0:
            d = hd[0]; v = hi[0]
            n -= 1
            ld = hd[n]; li = hi[n]; j = 0
            while True:
                c = 2 * j + 1
                if c >= n:
                    break
                if c + 1 < n and hd[c + 1] < hd[c]:
                    c += 1
                if hd[c] >= ld:
                    break
                hd[j] = hd[c]; hi[j] = hi[c]; j = c
            if n > 0:
                hd[j] = ld; hi[j] = li
            y = v // w; x = v - y * w
            if d > cum[y, x]:
                continue
            c0 = cost[y, x]
            for k in range(8):
                ny = y + DY[k]; nx = x + DX[k]
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                c1 = cost[ny, nx]
                if not np.isfinite(c1):
                    continue
                nd = d + 0.5 * (c0 + c1) * DL[k]
                if nd < cum[ny, nx]:
                    cum[ny, nx] = nd; prev[ny, nx] = y * w + x
                    if n == cap:
                        cap *= 2; hd2 = np.empty(cap, np.float64); hi2 = np.empty(cap, np.int64)
                        hd2[:n] = hd[:n]; hi2[:n] = hi[:n]; hd = hd2; hi = hi2
                    j = n; n += 1
                    while j > 0 and hd[(j - 1) >> 1] > nd:
                        p = (j - 1) >> 1; hd[j] = hd[p]; hi[j] = hi[p]; j = p
                    hd[j] = nd; hi[j] = ny * w + nx
        return cum, prev


class MCP:
    """使い方: mcp = MCP(コスト画像); cum, _ = mcp.find_costs(起点の一覧); 道 = mcp.traceback(終点)
    cum[y, x] = 起点からその画素までの最小の手間。traceback は起点 -> 終点の (y, x) の並びを返す。"""

    def __init__(self, cost):
        self.cost = np.ascontiguousarray(cost, np.float64)
        self.prev = None

    def find_costs(self, starts):
        starts = np.asarray(starts, np.int64)
        if _HAS_NUMBA:
            cum, self.prev = _dijkstra_nb(self.cost, starts[:, 0].copy(), starts[:, 1].copy(), _DY, _DX, _DL)
        else:
            cum, self.prev = _dijkstra_py(self.cost, starts)
        return cum, None

    def traceback(self, end):
        w = self.cost.shape[1]
        y, x = int(end[0]), int(end[1]); out = [(y, x)]
        v = self.prev[y, x]
        while v >= 0:
            y, x = divmod(int(v), w); out.append((y, x)); v = self.prev[y, x]
        return out[::-1]
'''
FILES['crack_v14/geometry.py'] = r'''"""検査範囲（U字の内縁・外縁・プレート）の定義。

座標はすべて「検査の座標」で書く。入力画像を 960x660 に縮小した画像の座標のこと（SIZE）。

  opening    : U字の開口部（中の背景が見える穴）。縁取りの外側の線までを含む多角形
  plate      : 割れが横切る樹脂のプレート部分（この中から opening を除いた所が検査範囲）
  outer      : 割れの「もう一方の端」。左右の縦リブと画像の下端の線
  rim_ignore : U字の縁取りが最初から切れている場所（ゲートの切り欠き）。縁取りの途切れ判定から外す
  y_eval     : これより上（U字の脚の上の方）は検査しない
  edge_margin: プレートの端から何px以内を「割れの証拠にしない」か（リブの縁の黒い線を拾わないため）
  min_thick  : 健全な肉厚がこれより薄い所は検査しない

検査範囲は u_detect.py が1枚ごとに画像から自動で作る。
"""
import cv2
import numpy as np

SIZE = (960, 660)  # 検査の座標の大きさ (幅, 高さ)

# ------------------------------------------------------------------ マスク作成
def _poly(pts, shape, closed=True, thick=None):
    """多角形を塗りつぶした（thick 指定なら線で描いた）True/False の画像を作る。"""
    m = np.zeros(shape, np.uint8)
    p = np.round(np.array(pts, np.float64)).astype(np.int32)
    if thick is None:
        cv2.fillPoly(m, [p], 1)
    else:
        cv2.polylines(m, [p], closed, 1, thick)
    return m.astype(bool)


def build_masks(g, rim_band=10):
    """検査範囲の定義 g から、検査で使うマスク画像（基準座標）を作る。

    戻り値:
      opening : U字の開口部
      plate   : プレート全体
      band    : 割れが横切る樹脂部分（plate から opening を除いた所）
      inner   : 割れの起点を探す帯（縁取りの内側の端。rim_width が無いときは開口部の縁のすぐ外側）
      rim     : 縁取りそのもの（経路が横切る所）
      outer   : 割れの終点（縦リブと下端の線）
    """
    shape = (SIZE[1], SIZE[0])
    opening = _poly(g["opening"], shape)
    plate = _poly(g["plate"], shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_band + 1,) * 2)
    open_dil = cv2.dilate(opening.astype(np.uint8), k).astype(bool)   # 開口部を rim_band だけ太らせる
    inner = open_dil & ~opening & plate
    band = plate & ~opening
    rim = np.zeros(shape, bool)
    rw = g.get("rim_width")
    if g.get("start_inside", True) and rw:
        # 割れの起点を「縁取りの内側の端」に置く。経路は縁取りそのものを横切らないと外へ出られないので、
        # U字の内側（開口部）まで割れている割れだけが抜けられる（傷や影は明るい縁取りで止まる）
        d = int(round(rw))
        e1 = cv2.erode(opening.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1,) * 2))
        e2 = cv2.erode(opening.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 13,) * 2))
        rim = opening & ~e2.astype(bool) & plate            # 縁取り（緑の線から内側へ 太さ + 6px）
        inner = e1.astype(bool) & ~e2.astype(bool) & plate  # 起点: 縁取りの内側の端の帯（幅 6px）
    outer = np.zeros(shape, bool)
    for line in g["outer"]:
        outer |= _poly(line, shape, closed=False, thick=5)
    outer &= band & ~open_dil
    return dict(opening=opening, plate=plate, band=band, inner=inner, outer=outer, rim=rim)
'''
FILES['crack_v14/inspect_v14.py'] = r'''"""カセットワレ検査 V14 の本体。1枚の画像を受け取って OK / NG を返す。

製品の種類（U字の太さ・形・刻印）が違っても、基準画像や設定なしで動く。

流れ:
  1. 縮小      : 入力画像を 960x660 にする（この座標で検査する）
  2. U字を探す : その画像の中から、U字の縁取り・縦リブ・ゲートの切り欠きを自動で見つける（u_detect.py）
  3. 背面光    : U字の開口部の中が明るいか（明るければ B も検査する）
  4. 割れらしさ: D（暗い割れ線）は全画像、B（明るい割れ）は背面光ありの画像で計算（features.py）
  5. 経路探索  : 内縁から外縁まで、割れっぽい所だけを通って抜けられるか（crossing.py）
  6. 判定      : 割れの定義「U字の内側から外縁まで、端から端まで割れている」を満たせば NG
"""
import cv2
import numpy as np

from .crossing import Crossing
from .features import bright_gap_map, dark_line_map, to_prob
from .geometry import SIZE
from .u_detect import detect_geometry

# ---------------------------------------------------------------- 判定の閾値（調整するのはここ）
PARAMS = dict(
    ratio_th=0.35,        # 道の上で健全部を通った割合がこれ以下なら「抜けている」（D: 暗い割れ）
    b_ratio_th=0.20,      # B（明るい割れ）は光が漏れてはっきり明るいので厳しくする（本物は 0.03〜0.06、リブの反射は 0.3 前後）
    gap_th=25.0,          # 割れていない区間が最長でもこれ以下 [px]
    end_gap_inner=10.0,   # 割れが内側の端からこれ以上手前で止まっていたら半割れ（OK） [px]
    end_gap_outer=25.0,   # 割れが外側の端からこれ以上手前で止まっていたら半割れ（OK） [px]
    rim_break_th=0.30,    # D: 起点で縁取りがこれ以上切れていること（上げると過検出が減り、見逃しが増える）
    solid_ratio_th=0.12,  # D: 縁取りの途切れが弱くても、ここまできれいに端から端までつながった割れ線なら NG
    solid_gap_th=10.0,    #    （本物の割れ 0.05〜0.09 / 5px、OK品の傷・溝は 0.17 以上 / 29px 以上）
    reach_th=0.90,        # B: 漏れた光が開口部までつながっている割合
    d_lo=0.10, d_hi=0.30, # D: 相対的な暗さ（×細長さ）がこの範囲で割れらしさ 0 -> 1
    b_lo=2.5, b_hi=6.0,   # B: 周りより明るい度合いがこの範囲で割れらしさ 0 -> 1
    min_u_score=20.0,     # U字のはっきり度がこれ未満なら「U字が見つからない」としてエラー
    hug_max=0.6,          # 経路のうち U字の縁 / 縦リブにぴったり沿っている割合がこれより多ければ、溝や影（割れではない）
    up_max=40.0,          # 経路の終点が起点よりこれ以上 [px] 上なら割れとみなさない（脚の溝を上へ這う線を除く）
)


def is_backlit(frame, opening):
    """背面光があるか: U字の開口部の中が、画像の一番明るい所に近い明るさか。"""
    op = cv2.erode(opening.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    v = frame[op]
    return bool(len(v) and np.percentile(v, 25) > 0.6 * np.percentile(frame, 99))


class Inspector:
    def __init__(self, params=None):
        self.p = dict(PARAMS, **(params or {}))

    def inspect(self, gray):
        p = self.p
        # 1. 縮小（M: 入力画像 -> 検査の座標 の変換）
        h, w = gray.shape
        M = np.array([[SIZE[0] / w, 0, 0], [0, SIZE[1] / h, 0], [0, 0, 1]], np.float64)
        frame = cv2.resize(gray, SIZE, interpolation=cv2.INTER_AREA)

        # 2. U字を探す
        geom, uscore = detect_geometry(frame)
        if uscore < p["min_u_score"]:
            return dict(judgment="ERROR", reason=f"U字が見つからない（はっきり度 {uscore:.1f}）")
        cr = Crossing(geom)
        band = cr.m["band"]
        res = {}

        # 縁・縦リブからの距離（縁や縦リブに沿って走る溝・影を見分けるため）
        dist = cv2.distanceTransform((~cr.m["opening"]).astype(np.uint8), cv2.DIST_L2, 5)   # U字の縁からの距離
        dist_rib = cv2.distanceTransform(cr.m["plate"].astype(np.uint8), cv2.DIST_L2, 5)     # 縦リブからの距離
        hug_d = 1.5 * geom.get("rim_width", 16) + 10

        def hug(q):
            """経路のうち、縁や縦リブにぴったり沿っている割合"""
            P = q["path"]
            return float(max((dist[P[:, 0], P[:, 1]] < hug_d).mean(), (dist_rib[P[:, 0], P[:, 1]] < 25).mean()))

        rim_mask = cr.m["rim"]

        def reject(q):
            """割れではない道:
              大部分が縁・縦リブに沿っている / 終点が起点よりかなり上（脚の溝を上へ這う）/
              縁取りの中を縁に沿って長く進んでいる（割れは縁取りをまっすぐ横切る）"""
            P = q["path"]
            in_rim = rim_mask[P[:, 0], P[:, 1]].sum() * 1.0
            along = in_rim > 2.5 * (geom.get("rim_width") or 16) + 12
            return hug(q) > p["hug_max"] or P[0, 0] - P[-1, 0] > p["up_max"] or along

        # 3-5. 背面光 -> 割れらしさ -> 経路探索
        lit = is_backlit(frame, cr.m["opening"])
        pd = to_prob(dark_line_map(frame, band), p["d_lo"], p["d_hi"])
        if lit:
            pb = to_prob(bright_gap_map(frame, band), p["b_lo"], p["b_hi"])
            res["B"] = cr.run_multi(pb, ratio_th=p["b_ratio_th"], gap_th=p["gap_th"], reject=reject)
            # 明るい割れ（NG と判定した B の経路）のふちの暗い所を、別の暗い割れとして数えない。
            # 明るい所全部を消すと、縁取りのハイライトの近くから始まる暗い割れまで消えてしまうので、
            # 本当に見つかった明るい割れの周りだけを消す
            near_b = np.zeros(pd.shape, np.uint8)
            for q in res["B"]:
                if q["ratio"] <= p["b_ratio_th"] and q["max_gap"] <= p["gap_th"] and not reject(q):
                    cv2.polylines(near_b, [q["path"][:, ::-1].astype(np.int32)], False, 1, 31)
            pd = np.where(near_b.astype(bool), 0.0, pd)
        res["D"] = cr.run_multi(pd, ratio_th=p["ratio_th"], gap_th=p["gap_th"], reject=reject)

        # 6. 割れの定義に合うか
        for t, cands in res.items():
            for q in cands:
                q["hug"] = hug(q)
                if reject(q):
                    q["ng"] = False
                start = q["path"][0][::-1].astype(np.float32)          # 起点 (x, y)
                ends_ok = q["inner_gap"] <= p["end_gap_inner"] and q["outer_gap"] <= p["end_gap_outer"]
                if t == "B":
                    q["reach"], _ = cr.reach_inside(q["prob"], start)
                    q["ng"] = q["ng"] and ends_ok and q["reach"] >= p["reach_th"]
                else:
                    q["rim_break"], ri = cr.rim_break(frame, start)
                    if cr.rim_ignored(ri):
                        q["rim_break"] = 0.0                            # ゲートの切り欠きは数えない
                    solid = q["ratio"] <= p["solid_ratio_th"] and q["max_gap"] <= p["solid_gap_th"]
                    q["ng"] = q["ng"] and ends_ok and (q["rim_break"] >= p["rim_break_th"] or solid)
                q.pop("prob", None)

        # 見つけた経路・U字の線を入力画像の座標に戻す（結果の描画と CSV 用）
        Minv = np.linalg.inv(M)

        def back(pts_xy):
            return cv2.perspectiveTransform(np.asarray(pts_xy, np.float64).reshape(-1, 1, 2), Minv).reshape(-1, 2)

        defects = []
        for t, cands in res.items():
            for q in cands:
                q["path_img"] = back(q["path"][:, ::-1])
                if q["ng"]:
                    (x1, y1), (x2, y2) = q["path_img"].min(0), q["path_img"].max(0)
                    defects.append(dict(type=t, bbox=(int(x1), int(y1), int(x2), int(y2))))
        types = sorted({t for t in res if any(q["ng"] for q in res[t])})
        u_img = dict(rim=back(geom["opening"][1:-1]), ribs=[back(line) for line in geom["outer"][:2]])
        return dict(kind="auto", backlit=lit, u_score=uscore, geom=geom, u_img=u_img, M=M, frame=frame,
                    res=res, judgment="NG" if defects else "OK", defects=defects, detail="+".join(types))
'''
FILES['crack_v14/screen_crop.py'] = r'''"""（テスト用）モニターをスマホで撮った写真から、画面の部分だけを切り出して 960x660 にする。本番では使わない。"""
import cv2, numpy as np

def crop_screen(im, W=960, H=660):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    g2 = g.copy()
    if h > w:   # 縦向きのスマホのスクショ: 上のバーと下のサムネイルを除く
        g2[: int(h * 0.31)] = 0; g2[int(h * 0.81):] = 0
    else:       # 横向き（モニターを撮った動画など）: 端の 3% だけ除く
        g2[: int(h * 0.03)] = 0; g2[int(h * 0.97):] = 0; g2[:, : int(w * 0.03)] = 0; g2[:, int(w * 0.97):] = 0
    m = (cv2.GaussianBlur(g2, (0, 0), 5) > 45).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    cnt, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnt, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    ap = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(ap) == 4:
        p = ap.reshape(4, 2).astype(np.float32)
        s = p.sum(1); d = np.diff(p, axis=1).ravel()
        src = np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)
    else:
        x, y, w, hh = cv2.boundingRect(c)
        src = np.float32([[x, y], [x + w, y], [x + w, y + hh], [x, y + hh]])
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(im, M, (W, H))
'''
FILES['crack_v14/shrink_video.py'] = r'''"""動画を小さくする（共有用）。ffmpeg が無くても OpenCV だけで動く。

  python -m crack_v14.shrink_video IMG_8074.MOV small.mp4            # 横 640px・1秒2コマ
  python -m crack_v14.shrink_video IMG_8074.MOV small.mp4 --width 800 --fps 3

音声は入らない。100MB の動画でも、だいたい数MB になる。
"""
import argparse

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--width", type=int, default=640, help="出力の横幅 [px]")
    ap.add_argument("--fps", type=float, default=2.0, help="1秒あたり何コマ残すか")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.src)
    if not cap.isOpened():
        raise SystemExit(f"動画が開けません: {a.src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / a.fps)))
    out, i, n = None, 0, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            h = int(round(img.shape[0] * a.width / img.shape[1])) // 2 * 2
            img = cv2.resize(img, (a.width, h), interpolation=cv2.INTER_AREA)
            if out is None:
                out = cv2.VideoWriter(a.dst, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.width, h))
            out.write(img)
            n += 1
        i += 1
    cap.release()
    if out is None:
        raise SystemExit("コマが読めませんでした")
    out.release()
    print(f"{n} コマ -> {a.dst}")


if __name__ == "__main__":
    main()
'''
FILES['crack_v14/u_detect.py'] = r'''"""1枚の画像から、U字の開口部・縁取り・縦リブを自動で見つける（製品の種類を問わない）。

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


def detect_geometry(frame, edge_margin=10, min_thick=12):
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
    lx, rx = find_ribs(frame, mid[:, 0].min(), mid[:, 0].max(), bottom[1] - 0.3 * (bottom[1] - top_y))
    half = (rim[:, 0].max() - rim[:, 0].min()) / 2
    gw = max(18.0, 0.12 * half)                               # ゲートの切り欠きの範囲（U字の大きさに比例）
    g = dict(
        y_eval=int(open_top + 70),                         # 起点は y_eval-60 から = 開口部の一番上から 10px 下まで検査する
        edge_margin=edge_margin, min_thick=min_thick, rim_width=width,
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
'''
FILES['crack_v14/README.md'] = r'''# カセットワレ検査 V14

## 特長
**製品の種類（U字の太さ・形・刻印）を問わず、設定なしで動く。**
基準画像・テンプレート・検査範囲の線引きは不要。U字は1枚ごとに、その画像の中から自動で見つける。

## 割れの定義
U字の内側（開口部）から外縁（左右の縦リブ・下端）まで、**端から端まで割れているもの**だけを NG にする。
途中で止まっている半割れ・表面の傷・成形線・影は OK。

- **B** = 明るい割れ（割れ目から背面光が漏れて白く見える）。背面光がある画像だけ検査
- **D** = 暗い割れ（黒い割れ線）。全部の画像で検査

## 処理の流れ（inspect_v14.py）
1. **縮小**: 入力画像を 960x660 にする
2. **U字を探す**（u_detect.py）:
   - 開口部（縁取りの内側の穴）を領域として切り出す（ウォーターシェッド法）。
     画像の真ん中の上の方 = 必ず U字の中、下端と左右の端 = 必ず外、として、輪郭の一番強い尾根（= 縁取り）で分ける。
     ネジ筒は U字の中にあるので混ざらない
   - その輪郭に「左右対称のまっすぐな脚 + なめらかな底」の綺麗なU字を当てはめる（ガタつかない）
   - 縦リブは脚のすぐ外側の一番はっきりした縦の線、ゲートの切り欠きは U字の一番下とする
3. **背面光の判定**: U字の開口部の中が明るいか
4. **割れらしさ**（features.py）: 画素ごとに割れっぽさを 0〜1 で数値化
5. **経路探索**（crossing.py）: **縁取りの内側の端**から外縁まで、割れっぽい所だけを通って抜けられる道を探す。
   起点を縁取りの内側に置くので、経路は縁取りそのものを横切らないと外へ出られない。
   U字の内側まで割れている割れだけが抜けられ、傷や影は明るい縁取りで止まる
6. **判定**: 下の条件を全部満たす道があれば NG

## 判定の条件と閾値（inspect_v14.py の PARAMS）
| 閾値 | 値 | 意味 | 上げると |
|---|---|---|---|
| ratio_th | 0.35 | 道の上で割れていない所を通った割合の上限 | 見逃しが減り、過検出が増える |
| gap_th | 25 px | 割れていない区間の最長の上限 | 見逃しが減り、過検出が増える |
| end_gap_inner | 10 px | 割れが内側の端から手前で止まってよい距離 | 半割れも NG になりやすい |
| end_gap_outer | 25 px | 割れが外側の端から手前で止まってよい距離 | 半割れも NG になりやすい |
| rim_break_th | 0.30 | D: 起点で U字の縁取りが切れている度合いの下限 | 過検出が減り、見逃しが増える |
| b_ratio_th | 0.20 | B: ratio_th の明るい割れ用（リブの反射を除くため厳しく） | 見逃しが減り、過検出が増える |
| solid_ratio_th / solid_gap_th | 0.12 / 10 px | D: 縁取りの途切れが弱くても、ここまでつながった割れ線なら NG | 見逃しが減り、過検出が増える |
| reach_th | 0.90 | B: 漏れた光が開口部までつながっている割合の下限 | 過検出が減り、見逃しが増える |
| hug_max | 0.6 | 経路が縁・縦リブにぴったり沿っている割合の上限（溝や影を除く） | 過検出が増える |
| up_max | 40 px | 終点が起点よりこれ以上上なら割れとみなさない | 過検出が増える |
| d_lo / d_hi | 0.10 / 0.30 | D: 暗さ（×細長さ）を割れらしさ 0〜1 にする範囲 | 薄い線を拾いにくくなる |
| b_lo / b_hi | 2.5 / 6.0 | B: 明るさを割れらしさ 0〜1 にする範囲 | 淡い光漏れを拾いにくくなる |
| min_u_score | 20 | U字のはっきり度の下限（下回ると ERROR） | U字が見つからない扱いが増える |

## 使い方
```
pip install -r requirements.txt
python -m crack_v14.cli --input <画像フォルダ> --out out
python -m crack_v14.cli --input <動画ファイル> --out out --every 1      # 動画は1秒ごとに1コマ検査
python -m crack_v14.evaluate --labels labels/defect_annotations.csv --pred out --role VALIDATION
```
出力画像には、見つけた割れ（水色 = B、赤 = D）に加えて、見つけた U字の縁取り（細い緑）と
縦リブ（細い黄色）を描く。U字の見つけ方がおかしくないか、ここで確認できる。

## ファイル
| ファイル | 中身 |
|---|---|
| cli.py | 検査の実行、結果の描画と CSV 出力 |
| inspect_v14.py | 検査の流れと判定（閾値はここ） |
| u_detect.py | U字・縁取り・縦リブ・ゲートを画像から自動で見つける |
| features.py | 割れらしさマップ（B・D） |
| crossing.py | 内縁 -> 外縁の経路探索、縁取りの途切れ |
| geodesic.py | 最短経路探索（ダイクストラ法） |
| geometry.py | 検査範囲からマスク画像を作る |
| evaluate.py | 正解CSVとの比較 |
| screen_crop.py | （テスト用）画面撮影写真の切り出し |

## 確認済み（画面撮影の画像）
- OK 94枚: すべて OK（過検出 0）
- NG 20枚: 19枚 NG、位置も一致。見逃し 1枚（No.16 の中央の細い線）

## 未確認のこと
- 実画像での動作（テストは画面をスマホで撮った写真で実施）
'''
FILES['requirements.txt'] = r'''# 動作確認済み: Python 3.12 / opencv 4.9.0 / numpy 1.26.4 / numba 0.60.0
opencv-python>=4.4      # headless 版でも可
numpy>=1.24
numba>=0.57             # 無くても動く（最短経路が純Pythonになり遅くなる）
'''

if __name__ == "__main__":
    for path, src in FILES.items():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        print("wrote", path)
