"""crack_v14 一式（1ファイル版）。 python crack_v14_bundle.py で crack_v14/ フォルダを展開する。

展開したあと:
  pip install -r requirements.txt
  # 仮の基準画像 ref_type1.png / ref_type2.png を data/refs/ に置く
  python -m crack_v14.make_ref --ok-dir pos1ok --template template_table1.png --out refs
  python -m crack_v14.cli --input <画像フォルダ> --template template_table1.png --ref-dir refs --out out
  python -m crack_v14.evaluate --labels labels/defect_annotations.csv --pred out
詳しくは crack_v14/README.md
"""
import os
FILES = {}
FILES['crack_v14/__init__.py'] = r''''''
FILES['crack_v14/align.py'] = r'''"""位置合わせ（テンプレートマッチング）。

今の検査システムと同じ考え方:
  検査画像の中から template_table1.png（右側のネジ2つと黒い留め具）に似た所を
  cv2.matchTemplate で探し、基準画像のときの位置との差を X・Y の補正量にする。

カメラは固定なので、ずれは平行移動だけと考える（回転・拡大縮小は補正しない）。
補正したあと、検査の座標（基準座標 960x660）に縮小する。
"""
import cv2
import numpy as np

from .geometry import SIZE


class TemplateAligner:
    def __init__(self, template_gray, blur=1.5):
        """template_gray: template_table1.png をグレースケールで読んだもの。"""
        if template_gray is None:
            raise ValueError("テンプレート画像が読めません")
        self.blur = blur
        self.tpl = self._prep(template_gray)

    def _prep(self, g):
        # 細かいノイズや反射の差で一致度がぶれないよう、少しぼかしてから比べる
        return cv2.GaussianBlur(g, (0, 0), self.blur) if self.blur > 0 else g

    def locate(self, gray):
        """テンプレートが一番よく一致した位置（左上の x, y）と一致度（-1〜1、1が完全一致）を返す。"""
        res = cv2.matchTemplate(self._prep(gray), self.tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(res)   # 最大値とその位置
        return float(x), float(y), float(score)


def frame_matrix(image_shape, offset):
    """入力画像 -> 基準座標 への変換行列（3x3）。

    1. offset (dx, dy) だけ平行移動して、基準画像と同じ位置にそろえる
    2. 960x660 に縮小する
    """
    h, w = image_shape[:2]
    sx, sy = SIZE[0] / w, SIZE[1] / h
    dx, dy = offset
    return np.array([[sx, 0, sx * dx], [0, sy, sy * dy], [0, 0, 1]], np.float64)


def to_frame(gray, M):
    """変換行列 M で入力画像を基準座標の画像（960x660）にする。"""
    return cv2.warpAffine(gray, M[:2], SIZE, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
'''
FILES['crack_v14/cli.py'] = r'''"""検査の実行。

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
        """geom: 検査範囲の定義（geometry.load_geom の戻り値）。"""
        self.g, self.eps = geom, eps
        self.m = build_masks(geom)
        self._build_rim(geom["opening"])
        y0 = geom["y_eval"]

        # 起点: 開口部の縁のすぐ外側の帯（U字の脚の上の方は除く）
        inner = self.m["inner"].copy()
        inner[: max(0, y0 - 60)] = False
        self.starts = np.argwhere(inner)

        # 歩ける範囲: 樹脂部分＋起点＋終点
        walk = self.m["band"] | self.m["inner"] | self.m["outer"]
        walk[: max(0, y0 - 80)] = False
        self.walk = walk

        # 健全なときの肉厚（全部コスト1で歩いたときの距離）を終点ごとに測っておく
        cum0, _ = MCP(np.where(walk, 1.0, np.inf)).find_costs(self.starts)
        ends = np.argwhere(self.m["outer"])
        th = cum0[ends[:, 0], ends[:, 1]]
        keep = (ends[:, 0] >= y0) & np.isfinite(th) & (th >= geom.get("min_thick", 60))
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
        """輪郭の各点で、縁取りを横切る方向に明るさを測り、縁取りの明るさ（上位15%の値）を返す。"""
        d = np.arange(-4, self.rim_depth)
        xy = self.rim_pts[:, None, :] + d[None, :, None] * self.rim_nrm[:, None, :]
        x = xy[..., 0].clip(0, gray.shape[1] - 1).astype(np.float32)
        y = xy[..., 1].clip(0, gray.shape[0] - 1).astype(np.float32)
        v = cv2.remap(cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32), x, y, cv2.INTER_LINEAR)
        return np.percentile(v, 85, axis=1)

    def nearest_rim(self, xy):
        """点 xy に一番近い輪郭点の番号。"""
        return int(np.argmin(np.hypot(*(self.rim_pts - xy).T)))

    def rim_break(self, gray, start_xy, win=10, ctx=60):
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

    def run_multi(self, prob, k=3, ratio_th=0.35, gap_th=25.0, sep=35):
        """割れが複数あるときのために、見つけた道の周り（sep px）を消して最大 k 本まで探す。"""
        out, p = [], prob.copy()
        for _ in range(k):
            r = self.run(p)
            r["ng"] = r["ratio"] <= ratio_th and r["max_gap"] <= gap_th
            out.append(r)
            if not r["ng"]:
                break
            m = np.zeros(p.shape, np.uint8)
            cv2.polylines(m, [r["path"][:, ::-1].astype(np.int32)], False, 1, 2 * sep + 1)
            p[m.astype(bool)] = 0.0
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

座標はすべて「基準座標」で書く。基準座標とは、入力画像を位置補正してから
960x660 に縮小した画像の座標のこと（SIZE）。

  opening    : U字の開口部（中の背景が見える穴）。縁取りの外側の線までを含む多角形
  plate      : 割れが横切る樹脂のプレート部分（この中から opening を除いた所が検査範囲）
  outer      : 割れの「もう一方の端」。左右の縦リブと画像の下端の線
  rim_ignore : U字の縁取りが最初から切れている場所（ゲートの切り欠き）。縁取りの途切れ判定から外す
  y_eval     : これより上（U字の脚の上の方）は検査しない
  edge_margin: プレートの端から何px以内を「割れの証拠にしない」か（リブの縁の黒い線を拾わないため）
  min_thick  : 健全な肉厚がこれより薄い所は検査しない

DEFAULT_GEOM は仮の基準画像（data/refs/ref_type*.png、画面写真から作ったもの）の上で
手で描いた座標。make_ref で実画像の基準を作ると、実画像に合わせて変換した座標が
refs/geom_<種類>.json に保存され、検査ではそちらが使われる。
"""
import json
import os

import cv2
import numpy as np

SIZE = (960, 660)  # 基準座標の大きさ (幅, 高さ)

DEFAULT_GEOM = {
    # 製品の種類1: U字の脚が上まで長く見える
    "type1": dict(
        y_eval=380, edge_margin=14,
        opening=[(262, 60), (265, 350), (285, 420), (330, 470), (400, 497), (450, 502), (520, 497),
                 (590, 470), (630, 420), (652, 350), (652, 60)],
        plate=[(236, 60), (688, 60), (688, 660), (236, 660)],
        rim_ignore=[(440, 470, 510, 530)],
        outer=[[(686, 60), (686, 660)], [(238, 60), (238, 660)], [(236, 657), (688, 657)]],
    ),
    # 製品の種類2: U字の内側にネジ筒が見える（MADE IN CHINA / eDV 刻印）
    "type2": dict(
        y_eval=120, edge_margin=10, min_thick=30,
        opening=[(195, 0), (200, 250), (215, 300), (250, 360), (300, 402), (380, 428), (450, 434),
                 (530, 422), (592, 390), (640, 350), (690, 290), (720, 230), (736, 0)],
        plate=[(152, 0), (758, 0), (758, 660), (152, 660)],
        rim_ignore=[(430, 400, 520, 460)],          # 6時の位置のゲート切り欠き
        outer=[[(756, 0), (756, 660)], [(154, 0), (154, 660)], [(152, 657), (758, 657)]],
    ),
}
KINDS = sorted(DEFAULT_GEOM)


# ------------------------------------------------------------------ 読み書き
def geom_path(ref_dir, kind):
    return os.path.join(ref_dir, f"geom_{kind}.json")


def load_geom(ref_dir, kind):
    """refs/geom_<種類>.json があればそれを、無ければ DEFAULT_GEOM を返す。"""
    p = geom_path(ref_dir, kind) if ref_dir else None
    if p and os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(json.dumps(DEFAULT_GEOM[kind]))   # 深いコピー


def save_geom(ref_dir, kind, g):
    with open(geom_path(ref_dir, kind), "w", encoding="utf-8") as f:
        json.dump(g, f, ensure_ascii=False, indent=1)


def transform_geom(g, M):
    """検査範囲の座標を 3x3 の変換行列 M で写す（仮の基準画像 -> 実画像の基準座標 など）。"""
    def tp(pts):
        a = np.array(pts, np.float64).reshape(-1, 1, 2)
        return [(round(float(x), 1), round(float(y), 1)) for x, y in cv2.perspectiveTransform(a, M).reshape(-1, 2)]
    def clip(pts):   # 画像の外に出た点は端に寄せる（下端の線が画面外に消えないように）
        return [(min(max(x, 2.0), SIZE[0] - 3.0), min(max(y, 2.0), SIZE[1] - 3.0)) for x, y in pts]
    out = dict(g)
    out["opening"] = tp(g["opening"])
    out["plate"] = tp(g["plate"])
    out["outer"] = [clip(tp(line)) for line in g["outer"]]
    boxes = []
    for x0, y0, x1, y1 in g.get("rim_ignore", []):
        (a0, b0), (a1, b1) = tp([(x0, y0), (x1, y1)])
        boxes.append((min(a0, a1), min(b0, b1), max(a0, a1), max(b0, b1)))
    out["rim_ignore"] = boxes
    cx = float(np.mean([p[0] for p in g["plate"]]))
    out["y_eval"] = int(round(tp([(cx, g["y_eval"])])[0][1]))
    return out


def adjust_matrix(move=(0.0, 0.0), scale=1.0):
    """線を中心から scale 倍に広げてから (dx, dy) 動かす行列（check画像を見て手で直すとき用）。"""
    cx, cy = SIZE[0] / 2, SIZE[1] / 2
    dx, dy = move
    return np.array([[scale, 0, cx - scale * cx + dx], [0, scale, cy - scale * cy + dy], [0, 0, 1]], np.float64)


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
      inner   : 割れの起点を探す帯（開口部の縁のすぐ外側、幅 rim_band）
      outer   : 割れの終点（縦リブと下端の線）
    """
    shape = (SIZE[1], SIZE[0])
    opening = _poly(g["opening"], shape)
    plate = _poly(g["plate"], shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_band + 1,) * 2)
    open_dil = cv2.dilate(opening.astype(np.uint8), k).astype(bool)   # 開口部を rim_band だけ太らせる
    inner = open_dil & ~opening & plate
    band = plate & ~opening
    outer = np.zeros(shape, bool)
    for line in g["outer"]:
        outer |= _poly(line, shape, closed=False, thick=5)
    outer &= band & ~open_dil
    return dict(opening=opening, plate=plate, band=band, inner=inner, outer=outer)
'''
FILES['crack_v14/inspect_v14.py'] = r'''"""カセットワレ検査 V14 の本体。1枚の画像を受け取って OK / NG を返す。

流れ:
  1. 位置合わせ   : template_table1.png をテンプレートマッチングで探し、X・Y のずれを補正して 960x660 に縮小
  2. 種類の判定   : 種類1・種類2の基準画像と輪郭を比べて、よく合う方にする（明るさは使わない）
  3. 背面光の判定 : U字の開口部が明るいか（背面光がある画像だけ B を検査する）
  4. 割れらしさ   : D（暗い割れ線）は全画像、B（明るい割れ）は背面光ありの画像で計算
  5. 経路探索     : 内縁から外縁まで、割れっぽい所だけを通って抜けられるか（crossing.py）
  6. 判定         : 割れの定義「U字の内側から外縁まで、端から端まで割れている」を満たせば NG

基準フォルダ（refs/）の中身（make_ref.py が作る）:
  ref_<種類>.png      基準画像（OK画像の中央値、基準座標 960x660）
  geom_<種類>.json    検査範囲の座標
  info_<種類>.json    基準画像を作ったときのテンプレート位置と画像サイズ
"""
import json
import os

import cv2
import numpy as np

from .align import TemplateAligner, frame_matrix, to_frame
from .crossing import Crossing
from .features import bright_gap_map, dark_line_map, to_prob
from .geometry import KINDS, SIZE, load_geom

# ---------------------------------------------------------------- 判定の閾値（調整するのはここ）
PARAMS = dict(
    ratio_th=0.35,        # 道の上で健全部を通った割合がこれ以下なら「抜けている」
    gap_th=25.0,          # 割れていない区間が最長でもこれ以下 [px]
    end_gap_inner=10.0,   # 割れが内側の端からこれ以上手前で止まっていたら半割れ（OK） [px]
    end_gap_outer=25.0,   # 割れが外側の端からこれ以上手前で止まっていたら半割れ（OK） [px]
    rim_break_th=0.30,    # D: 起点で縁取りがこれ以上切れていること（上げると過検出が減り、見逃しが増える）
    reach_th=0.90,        # B: 漏れた光が開口部までつながっている割合
    d_lo=0.10, d_hi=0.30, # D: 相対的な暗さ（×細長さ）がこの範囲で割れらしさ 0 -> 1
    b_lo=2.5, b_hi=6.0,   # B: 周りより明るい度合いがこの範囲で割れらしさ 0 -> 1
    min_template_score=0.5,  # テンプレートの一致度がこれ未満なら位置合わせ失敗とする
)


def _edge_strength(g):
    """輪郭の強さ（明るさの変化の大きさ）。種類の判定に使う。"""
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


class Kind:
    """製品の種類ごとの基準（基準画像・検査範囲・テンプレート位置）。"""

    def __init__(self, name, ref_gray, geom, tpl_xy, src_size):
        self.name, self.ref, self.g = name, ref_gray, geom
        self.tpl_xy = np.array(tpl_xy, np.float64)    # 基準画像を作ったときのテンプレート位置（入力画像の座標）
        self.src_size = tuple(src_size)               # そのときの入力画像サイズ (幅, 高さ)
        self.cross = Crossing(geom)
        m = self.cross.m
        # 種類の判定に使う範囲: 樹脂部分と縁取りの周り（開口部の中は明るさで見た目が変わるので使わない）
        k = np.ones((25, 25), np.uint8)
        ring = cv2.dilate(m["opening"].astype(np.uint8), k).astype(bool) & \
            ~cv2.erode(m["opening"].astype(np.uint8), k).astype(bool)
        self.shape_mask = m["band"] | ring
        self.ref_edge = _edge_strength(ref_gray)

    def shape_score(self, frame):
        """frame（基準座標の画像）がこの種類の基準画像と、輪郭がどれだけ合うか（相関、1が完全一致）。"""
        a, b = _edge_strength(frame)[self.shape_mask], self.ref_edge[self.shape_mask]
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        return float((a * b).mean())

    def backlit(self, frame):
        """背面光があるか: U字の開口部の中が、画像の一番明るい所に近い明るさか。"""
        op = cv2.erode(self.cross.m["opening"].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        op[: int(op.shape[0] * 0.15)] = False
        v = frame[op]
        return bool(len(v) and np.percentile(v, 25) > 0.6 * np.percentile(frame, 99))


def load_kinds(ref_dir):
    """refs/ から使える種類を全部読み込む。"""
    kinds = {}
    for k in KINDS:
        rp = os.path.join(ref_dir, f"ref_{k}.png")
        ip = os.path.join(ref_dir, f"info_{k}.json")
        if not (os.path.exists(rp) and os.path.exists(ip)):
            continue
        with open(ip, encoding="utf-8") as f:
            info = json.load(f)
        kinds[k] = Kind(k, cv2.imread(rp, 0), load_geom(ref_dir, k), info["tpl_xy"], info["src_size"])
    return kinds


class Inspector:
    def __init__(self, ref_dir, template_gray, params=None):
        self.kinds = load_kinds(ref_dir)
        if not self.kinds:
            raise SystemExit(f"基準がありません: {ref_dir} に ref_<種類>.png と info_<種類>.json が必要です"
                             f"（make_ref で作ってください）")
        self.aligner = TemplateAligner(template_gray)
        self.p = dict(PARAMS, **(params or {}))

    # ------------------------------------------------------------ 1〜3
    def align(self, gray):
        """位置合わせして、種類を決める。戻り値: (種類, 変換行列, 基準座標の画像, テンプレート一致度)"""
        x, y, score = self.aligner.locate(gray)
        if score < self.p["min_template_score"]:
            return None, None, None, score
        best = None
        for k in self.kinds.values():
            # 入力画像の大きさが基準を作ったときと違う場合は、テンプレート位置をその比率で読み替える
            sx, sy = gray.shape[1] / k.src_size[0], gray.shape[0] / k.src_size[1]
            offset = (k.tpl_xy[0] * sx - x, k.tpl_xy[1] * sy - y)
            M = frame_matrix(gray.shape, offset)
            frame = to_frame(gray, M)
            s = k.shape_score(frame)
            if best is None or s > best[0]:
                best = (s, k, M, frame)
        _, k, M, frame = best
        return k, M, frame, score

    # ------------------------------------------------------------ 全体
    def inspect(self, gray):
        p = self.p
        kind, M, frame, tscore = self.align(gray)
        if kind is None:
            return dict(judgment="ERROR", reason=f"テンプレートが見つからない（一致度 {tscore:.2f}）")
        cr = kind.cross
        band = cr.m["band"]
        res = {}

        # 4-5. 割れらしさ -> 経路探索
        pd = to_prob(dark_line_map(frame, band), p["d_lo"], p["d_hi"])
        lit = kind.backlit(frame)
        if lit:
            pb = to_prob(bright_gap_map(frame, band), p["b_lo"], p["b_hi"])
            res["B"] = cr.run_multi(pb, ratio_th=p["ratio_th"], gap_th=p["gap_th"])
            # 明るい割れのふちの暗い所を、別の暗い割れとして数えない
            near_b = cv2.dilate((pb > 0.3).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            pd = np.where(near_b, 0.0, pd)
        res["D"] = cr.run_multi(pd, ratio_th=p["ratio_th"], gap_th=p["gap_th"])

        # 6. 割れの定義に合うか
        for t, cands in res.items():
            for q in cands:
                start = q["path"][0][::-1].astype(np.float32)          # 起点 (x, y)
                ends_ok = q["inner_gap"] <= p["end_gap_inner"] and q["outer_gap"] <= p["end_gap_outer"]
                if t == "B":
                    q["reach"], _ = cr.reach_inside(q["prob"], start)
                    q["ng"] = q["ng"] and ends_ok and q["reach"] >= p["reach_th"]
                else:
                    q["rim_break"], ri = cr.rim_break(frame, start)
                    if cr.rim_ignored(ri):
                        q["rim_break"] = 0.0                            # ゲートの切り欠きは数えない
                    q["ng"] = q["ng"] and ends_ok and q["rim_break"] >= p["rim_break_th"]
                q.pop("prob", None)

        # 見つけた経路を入力画像の座標に戻す（結果の描画と CSV 用）
        Minv = np.linalg.inv(M)
        defects = []
        for t, cands in res.items():
            for q in cands:
                pts = q["path"][:, ::-1].reshape(-1, 1, 2).astype(np.float64)
                q["path_img"] = cv2.perspectiveTransform(pts, Minv).reshape(-1, 2)
                if q["ng"]:
                    (x1, y1), (x2, y2) = q["path_img"].min(0), q["path_img"].max(0)
                    defects.append(dict(type=t, bbox=(int(x1), int(y1), int(x2), int(y2))))
        types = sorted({t for t in res if any(q["ng"] for q in res[t])})
        return dict(kind=kind.name, backlit=lit, template_score=tscore, M=M, frame=frame, res=res,
                    judgment="NG" if defects else "OK", defects=defects, detail="+".join(types))
'''
FILES['crack_v14/make_ref.py'] = r'''"""OK画像のフォルダから、検査に使う基準（refs/）を作る。最初の1回だけ実行する。

使い方:
  python -m crack_v14.make_ref --ok-dir pos1ok --template template_table1.png --out refs

やること:
  1. 各OK画像の製品の種類（type1 / type2）を判定する
     -> 仮の基準画像（data/refs/ref_type*.png、画面写真から作ったもの）と特徴点で照合して決める
  2. 種類ごとに、先頭のOK画像を「軸」にする
     -> 仮の基準画像の上で描いてある検査範囲の線を、軸の画像の座標に写して geom_<種類>.json に保存
        （特徴点の照合 = SIFT を使うのはこの1回だけ。検査のときは使わない）
  3. 残りのOK画像を、テンプレートマッチングで軸の画像に位置合わせする（検査のときと同じ方法）
  4. 全部を重ねて、画素ごとの中央値を基準画像 ref_<種類>.png にする
  5. check_<種類>.jpg に検査範囲を重ねて描く -> 目で確認する
       緑 = 割れの起点を探す所（U字内縁の縁取りのすぐ外側）
       赤 = 割れの終点（左右の縦リブと下端）
       青 = 割れを探す範囲（樹脂の部分）

線がずれていたら --move-geom DX DY（右・下に何px動かすか）と --scale-geom（何倍に広げるか）で直す。
仮の基準画像が無いときは --kind type1 のように種類を指定すると、DEFAULT の線をそのまま使う。
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from .align import TemplateAligner, frame_matrix, to_frame
from .geometry import DEFAULT_GEOM, KINDS, SIZE, adjust_matrix, build_masks, save_geom, transform_geom
from .inspect_v14 import Kind


def load_gray(p):
    g = cv2.imread(p, 0)
    if g is None:
        raise SystemExit(f"読めない画像: {p}")
    return g


def classify_with_old_refs(gray, olds):
    """仮の基準画像と特徴点で照合して、種類と「入力画像 -> 仮の基準画像」の変換を返す。"""
    from .register import Reference, features, refine_ecc, register
    f = min(1.0, 1.3 * max(SIZE) / max(gray.shape))         # 大きい画像は縮小してから照合（速さのため）
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    S = np.diag([f, f, 1.0])
    feats = features(small)
    best = None
    for name, (ref_obj, kind_obj) in olds.items():
        H, n = register(small, ref_obj, feats=feats)
        if H is None or n < 8:
            continue
        cas = cv2.dilate((kind_obj.cross.m["plate"] | kind_obj.cross.m["opening"]).astype(np.uint8),
                         np.ones((41, 41), np.uint8))
        Hf, _ = refine_ecc(gray, kind_obj.ref, H @ S, cas)
        s = kind_obj.shape_score(cv2.warpPerspective(gray, Hf, SIZE))
        if best is None or s > best[0]:
            best = (s, name, Hf)
    return (None, None) if best is None else (best[1], best[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ok-dir", required=True, help="OK画像のフォルダ（種類が混ざっていてよい）")
    ap.add_argument("--template", required=True, help="template_table1.png")
    ap.add_argument("--out", default="refs")
    ap.add_argument("--old-ref-dir", default="data/refs", help="仮の基準画像の場所")
    ap.add_argument("--kind", default=None, choices=KINDS, help="仮の基準画像なしで作るときの種類")
    ap.add_argument("--max", type=int, default=40, help="種類ごとに使う最大枚数")
    ap.add_argument("--move-geom", type=float, nargs=2, default=(0, 0), metavar=("DX", "DY"),
                    help="検査範囲の線を右にDX、下にDY px動かす（960x660座標）")
    ap.add_argument("--scale-geom", type=float, default=1.0, help="検査範囲の線を中心から何倍に広げるか")
    a = ap.parse_args()

    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(a.ok_dir, f"*.{e}")))
    if not files:
        raise SystemExit(f"画像がありません: {a.ok_dir}")
    os.makedirs(a.out, exist_ok=True)
    aligner = TemplateAligner(load_gray(a.template))
    adjust = adjust_matrix(a.move_geom, a.scale_geom)

    # ---- 1. 種類ごとに振り分け（軸の画像については、仮の基準画像への変換も覚えておく）
    groups = defaultdict(list)       # 種類 -> [(ファイル名, 画像)]
    to_old = {}                      # 種類 -> 軸の画像 -> 仮の基準画像 の変換
    if a.kind is None:
        from .register import Reference
        olds = {}
        for k in KINDS:
            p = os.path.join(a.old_ref_dir, f"ref_{k}.png")
            if not os.path.exists(p):
                raise SystemExit(f"仮の基準画像がありません: {p}\n"
                                 f"  --old-ref-dir で場所を指定するか、--kind type1 などで作ってください")
            g = load_gray(p)
            olds[k] = (Reference(k, g), Kind(k, g, DEFAULT_GEOM[k], (0, 0), g.shape[::-1]))
        for f in files:
            g = load_gray(f)
            k, H = classify_with_old_refs(g, olds)
            if k is None:
                print(f"{os.path.basename(f)}: 種類を判定できないので除外"); continue
            if len(groups[k]) < a.max:
                if k not in to_old:
                    to_old[k] = H
                groups[k].append((f, g))
        print("振り分け: " + ", ".join(f"{k}={len(v)}枚" for k, v in sorted(groups.items())))
    else:
        groups[a.kind] = [(f, load_gray(f)) for f in files[: a.max]]
        print(f"仮の基準画像なし: 全部 {a.kind} とみなし、検査範囲は DEFAULT の線を使います")

    # ---- 2〜5. 種類ごとに基準を作る
    for k, items in sorted(groups.items()):
        f0, g0 = items[0]
        F0 = frame_matrix(g0.shape, (0, 0))                     # 軸の画像 -> 基準座標（縮小だけ）
        if k in to_old:   # 仮の基準画像の線 -> 軸の画像の基準座標
            geom = transform_geom(DEFAULT_GEOM[k], F0 @ np.linalg.inv(to_old[k]))
        else:
            geom = json.loads(json.dumps(DEFAULT_GEOM[k]))
        geom = transform_geom(geom, adjust)
        tx0, ty0, s0 = aligner.locate(g0)
        print(f"[{k}] 軸: {os.path.basename(f0)}（テンプレート一致度 {s0:.2f}）")

        stack = [to_frame(g0, F0)]
        for f, g in items[1:]:
            x, y, s = aligner.locate(g)
            if s < 0.5:
                print(f"[{k}] {os.path.basename(f)}: テンプレートが見つからないので除外（一致度 {s:.2f}）"); continue
            sx, sy = g.shape[1] / g0.shape[1], g.shape[0] / g0.shape[0]
            stack.append(to_frame(g, frame_matrix(g.shape, (tx0 * sx - x, ty0 * sy - y))))

        st = np.stack(stack).astype(np.float32)
        med = np.median(st, axis=0)
        mad = np.median(np.abs(st - med), axis=0) * 1.4826   # OK品のばらつき（OK差分モデル用に保存）
        ref = np.clip(med, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(a.out, f"ref_{k}.png"), ref)
        np.save(os.path.join(a.out, f"ok_mad_{k}.npy"), mad)
        save_geom(a.out, k, geom)
        with open(os.path.join(a.out, f"info_{k}.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(tpl_xy=[tx0, ty0], src_size=[g0.shape[1], g0.shape[0]], n_images=len(stack),
                           anchor=os.path.basename(f0)), fh, ensure_ascii=False, indent=1)

        m = build_masks(geom)
        vis = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        over = vis.copy(); over[m["band"]] = (200, 80, 0)
        vis = cv2.addWeighted(over, 0.25, vis, 0.75, 0)
        vis[m["inner"]] = (0, 220, 0); vis[m["outer"]] = (0, 0, 255)
        cv2.imwrite(os.path.join(a.out, f"check_{k}.jpg"), vis)
        print(f"[{k}] {len(stack)} 枚から作成 -> {a.out}/ref_{k}.png（{a.out}/check_{k}.jpg で線を確認）")

    for k in KINDS:
        if k not in groups:
            print(f"注意: {k} のOK画像が無かったので、{k} の基準は作っていません（{k} は検査されません）")


if __name__ == "__main__":
    main()
'''
FILES['crack_v14/register.py'] = r'''"""特徴点による照合（SIFT）。make_ref.py で「仮の基準画像の線を実画像に写す」ときだけ使う。
検査のとき（cli.py / inspect_v14.py）は使わない。

  SIFT       : 画像の中から目印になる点（角や模様）を見つけ、その周りの見た目を数値（特徴量）にする
  照合       : 2枚の画像で特徴量が似ている点どうしを対応付ける
  MAGSAC     : 対応の中には間違いが混ざるので、それを捨てながら「どう動かせば重なるか」（ホモグラフィ）を求める
               （RANSAC の改良版）
  ECC        : 求めた重ね方を、画像全体の明るさの一致度が上がるように細かく直す
"""
import cv2
import numpy as np

_sift = cv2.SIFT_create(nfeatures=4000)


def _prep(gray):
    # 明るさのムラを均してから特徴点を探す（暗い所にも目印が見つかるように）
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)


class Reference:
    """照合の相手になる画像（特徴点を先に計算しておく）。"""

    def __init__(self, name, gray):
        self.name, self.gray = name, gray
        self.kp, self.des = _sift.detectAndCompute(_prep(gray), None)


def features(gray):
    """画像の特徴点と特徴量。"""
    return _sift.detectAndCompute(_prep(gray), None)


def register(gray, ref, ratio=0.75, feats=None):
    """gray を ref に重ねる変換（3x3）と、正しく対応した点の数を返す。失敗したら (None, 点の数)。"""
    kp, des = feats if feats is not None else features(gray)
    if des is None or len(kp) < 10:
        return None, 0
    m = cv2.BFMatcher().knnMatch(des, ref.des, k=2)
    # 1番目に似た点が2番目よりはっきり似ているものだけ残す（あいまいな対応を捨てる）
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < ratio * b.distance]
    if len(good) < 12:
        return None, len(good)
    src = np.float32([kp[g.queryIdx].pt for g in good])
    dst = np.float32([ref.kp[g.trainIdx].pt for g in good])
    cv2.setRNGSeed(0)                                       # 毎回同じ結果になるように
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, inl = cv2.findHomography(src, dst, method, 4.0, maxIters=10000, confidence=0.999)
    return H, int(inl.sum()) if inl is not None else 0


def refine_ecc(gray, ref_gray, H, mask, iters=80):
    """変換 H を ECC で微調整する（mask の範囲の明るさが一致するように）。"""
    h, w = ref_gray.shape
    warped = cv2.warpPerspective(gray, H, (w, h))
    a = cv2.GaussianBlur(_prep(ref_gray), (0, 0), 2).astype(np.float32)
    b = cv2.GaussianBlur(_prep(warped), (0, 0), 2).astype(np.float32)
    W = np.eye(3, dtype=np.float32)
    try:
        cc, W = cv2.findTransformECC(a, b, W, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5),
                                     mask.astype(np.uint8), 5)
    except cv2.error:
        return H, 0.0
    return np.linalg.inv(W).astype(np.float64) @ H, float(cc)
'''
FILES['crack_v14/screen_crop.py'] = r'''"""（テスト用）モニターをスマホで撮った写真から、画面の部分だけを切り出して 960x660 にする。本番では使わない。"""
import cv2, numpy as np

def crop_screen(im, W=960, H=660):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = g.shape[0]
    g2 = g.copy(); g2[: int(h * 0.31)] = 0; g2[int(h * 0.81):] = 0   # スマホの上のバーと下のサムネイルを除く
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
FILES['crack_v14/README.md'] = r'''# カセットワレ検査 V14

## 割れの定義
U字の内側（開口部）から外縁（左右の縦リブ・下端）まで、**端から端まで割れているもの**だけを NG にする。
途中で止まっている半割れ・表面の傷・成形線・影は OK。

- **B** = 明るい割れ（割れ目から背面光が漏れて白く見える）。背面光がある画像だけ検査
- **D** = 暗い割れ（黒い割れ線）。全部の画像で検査
- **製品の種類** = type1（U字の脚が上まで長く見える）/ type2（U字の内側にネジ筒が見える）

## 処理の流れ（inspect_v14.py）
1. **位置合わせ**: `template_table1.png`（右側のネジ2つと留め具）を `cv2.matchTemplate` で探し、X・Y のずれを補正して 960x660 に縮小（align.py）
2. **種類の判定**: type1・type2 の基準画像と輪郭を比べ、よく合う方にする（明るさは使わない）
3. **背面光の判定**: U字の開口部が明るいか
4. **割れらしさ**: 画素ごとに割れっぽさを 0〜1 で数値化（features.py）
   - D: 周りより暗く、細長い所
   - B: 周りの樹脂より明るい所
5. **経路探索**: 内縁から外縁まで、割れっぽい所だけを通って抜けられる道を探す（crossing.py, geodesic.py）
6. **判定**: 下の条件を全部満たす道があれば NG

## 判定の条件と閾値（inspect_v14.py の PARAMS）
| 閾値 | 値 | 意味 | 上げると |
|---|---|---|---|
| ratio_th | 0.35 | 道の上で割れていない所を通った割合の上限 | 見逃しが減り、過検出が増える |
| gap_th | 25 px | 割れていない区間の最長の上限 | 見逃しが減り、過検出が増える |
| end_gap_inner | 10 px | 割れが内側の端から手前で止まってよい距離 | 半割れも NG になりやすい |
| end_gap_outer | 25 px | 割れが外側の端から手前で止まってよい距離 | 半割れも NG になりやすい |
| rim_break_th | 0.30 | D: 起点で U字の縁取りが切れている度合いの下限 | 過検出が減り、見逃しが増える |
| reach_th | 0.90 | B: 漏れた光が開口部までつながっている割合の下限 | 過検出が減り、見逃しが増える |
| d_lo / d_hi | 0.10 / 0.30 | D: 暗さ（×細長さ）を割れらしさ 0〜1 にする範囲 | 薄い線を拾いにくくなる |
| b_lo / b_hi | 2.5 / 6.0 | B: 明るさを割れらしさ 0〜1 にする範囲 | 淡い光漏れを拾いにくくなる |

## 使い方
```
pip install -r requirements.txt

# 1. 基準を作る（最初の1回）。OK画像のフォルダは種類が混ざっていてよい
#    仮の基準画像 data/refs/ref_type1.png, ref_type2.png が必要
python -m crack_v14.make_ref --ok-dir pos1ok --template template_table1.png --out refs
#    -> refs/check_type1.jpg, check_type2.jpg で線を確認
#       緑 = U字内縁の縁取りのすぐ外側、赤 = 縦リブと下端、青 = 樹脂の部分 になっていれば OK
#       ずれていたら --move-geom DX DY（右・下に何px）、--scale-geom（何倍）を付けて作り直す

# 2. 検査
python -m crack_v14.cli --input <画像フォルダ> --template template_table1.png --ref-dir refs --out out

# 3. 正解CSVとの比較
python -m crack_v14.evaluate --labels labels/defect_annotations.csv --pred out --role VALIDATION
```

## ファイル
| ファイル | 中身 |
|---|---|
| cli.py | 検査の実行、結果の描画と CSV 出力 |
| inspect_v14.py | 検査の流れと判定（閾値はここ） |
| align.py | テンプレートマッチングでの位置合わせ |
| features.py | 割れらしさマップ（B・D） |
| crossing.py | 内縁 -> 外縁の経路探索、縁取りの途切れ |
| geodesic.py | 最短経路探索（ダイクストラ法） |
| geometry.py | 検査範囲の座標とマスク |
| make_ref.py | OK画像から基準を作る（最初の1回） |
| register.py | 特徴点照合（make_ref だけで使用） |
| evaluate.py | 正解CSVとの比較 |
| screen_crop.py | （テスト用）画面撮影写真の切り出し |

## 未確認のこと
- OK画像での過検出率（OK画像で回していないので不明）
- 本物の template_table1.png・実画像での動作（テストは画面写真を加工したもので実施）
- No.16 の中央の細い線は検出できていない（同じ位置に成形線があり、OK画像との比較が必要）
'''
FILES['requirements.txt'] = r'''# 動作確認済み: Python 3.12 / opencv 4.9.0 / numpy 1.26.4 / numba 0.60.0
opencv-python>=4.4      # SIFT が本体に入ったのは 4.4 から（headless 版でも可）
numpy>=1.24
numba>=0.57             # 無くても動く（最短経路が純Pythonになり遅くなる）
'''

if __name__ == "__main__":
    for path, src in FILES.items():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        print("wrote", path)
