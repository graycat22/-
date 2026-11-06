"""カセットワレ検査 V14 の本体。1枚の画像を受け取って OK / NG を返す。

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
    ratio_th=0.30,        # 道の上で健全部を通った割合がこれ以下なら「抜けている」（D: 暗い割れ。本物は最大 0.25、OK品の傷は 0.31〜）
    b_ratio_th=0.50,      # B（明るい割れ）は光が漏れてはっきり明るいので厳しくする（本物は 0.03〜0.06、リブの反射は 0.3 前後）
    gap_th=25.0,          # 割れていない区間が最長でもこれ以下 [px]
    end_gap_inner=15.0,   # 割れが内側の端からこれ以上手前で止まっていたら半割れ（OK） [px]
    end_gap_outer=25.0,   # 割れが外側の端からこれ以上手前で止まっていたら半割れ（OK） [px]
    rim_break_th=0.30,    # D: 起点で縁取りがこれ以上切れていること（上げると過検出が減り、見逃しが増える）
    solid_ratio_th=0.12,  # D: 縁取りの途切れが弱くても、ここまできれいに端から端までつながった割れ線なら NG
    solid_gap_th=10.0,    #    （本物の割れ 0.05〜0.09 / 5px、OK品の傷・溝は 0.17 以上 / 29px 以上）
    direct_ratio_th=0.0,  # D: まっすぐ外縁へ向かう割れ（遠回り度 direct_detour 以下）なら、ここまで緩める
    direct_detour=2.0,    #    遠回り度 = 経路の長さ / 起点から外縁までの最短距離（溝や成形線を伝う道は大きくなる）
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
        to_outer = cv2.distanceTransform((~cr.m["outer"]).astype(np.uint8), cv2.DIST_L2, 5)   # 外縁までの距離

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
                    at_gate = cr.rim_ignored(ri)
                    if at_gate:
                        q["rim_break"] = 0.0                            # ゲートの切り欠きは数えない
                    P = q["path"]
                    detour = q["length"] / max(float(to_outer[P[0, 0], P[0, 1]]), 1.0)
                    q["detour"] = detour
                    # ゲートから真下の穴へは、OK品にも同じ位置に成形線が通っている。ゲートから始まる線は
                    # 「きれいにつながっている」だけでは NG にしない（縁取りの途切れが必要）
                    solid = (not at_gate) and q["max_gap"] <= p["solid_gap_th"] and (
                        q["ratio"] <= p["solid_ratio_th"]
                        or (q["ratio"] <= p["direct_ratio_th"] and detour <= p["direct_detour"]))
                    q["ng"] = q["ng"] and ends_ok and (q["rim_break"] >= p["rim_break_th"] or solid)
                q.pop("prob", None)

        # 割れの幅: 割れらしい画素の塊のうち、経路の近くにある部分を「割れの領域」として、
        # 経路の各点での太さ（中心から縁までの距離 × 2）を測る
        maps = {"D": pd}
        if lit:
            maps["B"] = pb
        sx_in = gray.shape[1] / SIZE[0]                         # 検査の座標 1px = 入力画像の何px か
        for t, cands in res.items():
            crack = (maps[t] > 0.15).astype(np.uint8)
            dt = cv2.distanceTransform(crack, cv2.DIST_L2, 3)
            for q in cands:
                if not q["ng"]:
                    continue
                P = q["path"]
                near = np.zeros(crack.shape, np.uint8)
                cv2.polylines(near, [P[:, ::-1].astype(np.int32)], False, 1, 15)
                q["region"] = (crack & near).astype(bool)            # 割れの領域（幅のある形）
                w = 2 * dt[P[:, 0], P[:, 1]]
                w = w[w > 0]
                q["width_px"] = float(np.median(w) * sx_in) if len(w) else 0.0      # 入力画像での幅 [px]
                q["width_max_px"] = float(np.percentile(w, 95) * sx_in) if len(w) else 0.0

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
        if "frame_box" in geom:                                  # カセットの外側の縁（確認用の赤い枠）
            a_, b_ = geom["frame_box"]; yb_ = max(y for _, y in geom["plate"])
            u_img["frame"] = back([(a_, 0), (b_, 0), (b_, yb_), (a_, yb_), (a_, 0)])
        return dict(kind="auto", backlit=lit, u_score=uscore, geom=geom, u_img=u_img, M=M, frame=frame,
                    res=res, judgment="NG" if defects else "OK", defects=defects, detail="+".join(types))
