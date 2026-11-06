"""カセットワレ検査 V14 の本体。1枚の画像を受け取って OK / NG を返す。

流れ:
  1. 位置合わせ   : template_table1.png をテンプレートマッチングで探し、X・Y のずれを補正して 960x660 に縮小
  2. 種類の判定   : 種類1・種類2の基準画像と輪郭を比べて、よく合う方にする（明るさは使わない）
  3. 背面光の判定 : U字の開口部が明るいか（背面光がある画像だけ B を検査する）
  4. 割れらしさ   : D（暗い割れ線）は全画像、B（明るい割れ）は背面光ありの画像で計算
  5. 経路探索     : 内縁から外縁まで、割れっぽい所だけを通って抜けられるか（crossing.py）
  6. 判定         : 割れの定義「U字の内側から外縁まで、端から端まで割れている」を満たせば NG

基準フォルダ（refs/）の中身:
  ref_<種類>.png      基準画像（OK画像の中央値、基準座標 960x660）     make_ref.py が作る
  info_<種類>.json    基準画像を作ったときのテンプレート位置と画像サイズ   make_ref.py が作る
  geom_<種類>.json    検査範囲の座標                                       draw_geom.py で線を引いて作る
"""
import glob
import json
import os

import cv2
import numpy as np

from .align import TemplateAligner, frame_matrix, to_frame
from .crossing import Crossing
from .features import bright_gap_map, dark_line_map, to_prob
from .geometry import SIZE, load_geom

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
    """refs/ から、基準画像・info・検査範囲の3つが揃っている種類を全部読み込む。"""
    kinds = {}
    for rp in sorted(glob.glob(os.path.join(ref_dir, "ref_*.png"))):
        k = os.path.basename(rp)[4:-4]
        ip = os.path.join(ref_dir, f"info_{k}.json")
        g = load_geom(ref_dir, k)
        if not os.path.exists(ip):
            print(f"注意: {ip} が無いので {k} は検査しません（make_ref で作り直してください）"); continue
        if g is None:
            print(f"注意: {k} の検査範囲が無いので {k} は検査しません（draw_geom で線を引いてください）"); continue
        with open(ip, encoding="utf-8") as f:
            info = json.load(f)
        kinds[k] = Kind(k, cv2.imread(rp, 0), g, info["tpl_xy"], info["src_size"])
    return kinds


class Inspector:
    def __init__(self, ref_dir, template_gray, params=None):
        self.kinds = load_kinds(ref_dir)
        if not self.kinds:
            raise SystemExit(f"使える基準がありません: {ref_dir}（make_ref と draw_geom を先に実行してください）")
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
