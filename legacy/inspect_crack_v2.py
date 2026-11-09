#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InspectorCrackPos1/2 の単一画像テスト（黒ワレの判定ロジック改善版 v2）

黒ワレ（暗い割れ）の判定を作り直した。明るいワレの判定は元のまま。
元のロジックの問題と、v2 での対応:
  1. 前処理の CLOSE(11) が幅 11px 未満の細い割れ線を埋めて消していた
     -> 3x3 の軽いぼかしだけにして、細い線を残す
  2. 探す範囲が中央 ±250px を除いた左右の細い帯だけだった
     -> 除外はゲート（中央の成形線）の所だけ（±gate_half px）にする
  3. 判定が「一番縦に長い黒い塊の高さ > 20px」だけで、影・汚れ・傷でも NG になり得た
     -> 割れの定義どおり「U字の縁から、ROI の左右の端か下端まで、黒い線がつながっているか」で判定する
  4. 黒さが固定値（平均輝度 18 以下）で、照明の明るさで効き方が変わった
     -> 周りのプレートの明るさに対して何割暗いか（相対値）で見る
"""

import json
import os
import traceback

import click
import cv2
import numpy as np
from loguru import logger

def getker(num):
    """元コードのgetker()相当"""
    num_h = num // 2
    out_ary = np.zeros([num, num], dtype=np.uint8)
    cv2.circle(out_ary, (num_h, num_h), num_h, 1, -1)
    return out_ary


def _contour_mean(ary, contour):
    """輪郭内部の平均輝度"""
    mask = np.zeros_like(ary)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return float(ary[mask > 0].mean())


BLACKHAT_KERNELS = [
    np.ones([1, 51], np.uint8),
    np.ones([51, 1], np.uint8),
    np.eye(35, dtype=np.uint8),
    np.fliplr(np.eye(35, dtype=np.uint8)).copy(),
]

# 黒ワレ領域を広げるときに使う、細い線用のブラックハットのカーネル（元より短くして細かい曲がりにも沿う）
GROW_KERNELS = [
    np.ones([1, 31], np.uint8),
    np.ones([31, 1], np.uint8),
    np.eye(21, dtype=np.uint8),
    np.fliplr(np.eye(21, dtype=np.uint8)).copy(),
]


def grow_black_region(ary_interest, seed_contours, cx_local, radius,
                      bh_th=20, join=7, seed_dilate=9):
    """元の判定で見つかった黒ワレ（種）から、つながっている細い暗い線をたどって割れ全体の領域を作る。

    ary_interest  : 黒ワレ検出に使った切り出し画像（グレー）
    seed_contours : 元の判定で残った黒ワレの輪郭
    cx_local      : 切り出し画像の中での U字の中心 x
    radius        : U字の半径（元コードの radius）
    bh_th         : ブラックハットの値がこれより大きい画素を「暗い線」とみなす（元と同じ 20）
    join          : 途切れた線をつなぐ大きさ [px]
    戻り値: 割れの領域（0/1 の画像、ary_interest と同じ大きさ）
    """
    h, w = ary_interest.shape
    region = np.zeros((h, w), np.uint8)
    if not seed_contours:
        return region

    # 探してよい範囲: 元と同じ U字の外側の帯（ただし中央の除外は無し。種につながる線だけ拾うので安全）
    allow = np.zeros((h, w), np.uint8)
    cv2.circle(allow, (cx_local, 0), int(radius * 1.75), 255, -1)
    cv2.circle(allow, (cx_local, 0), int(radius * 0.97), 0, -1)

    # 細い線（幅 1〜2px）を残すため、CLOSE(11) ではなく 3x3 の軽いぼかしだけにする
    th_clip = int(np.percentile(ary_interest, 50))
    soft = cv2.GaussianBlur(ary_interest.clip(0, th_clip), (3, 3), 0)
    blackhat = np.max([cv2.morphologyEx(soft, cv2.MORPH_BLACKHAT, k) for k in GROW_KERNELS], axis=0)
    line = ((blackhat > bh_th) & (allow > 0)).astype(np.uint8)
    line = cv2.morphologyEx(line, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (join, join)))

    seed = np.zeros((h, w), np.uint8)
    cv2.drawContours(seed, seed_contours, -1, 1, -1)
    seed = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seed_dilate, seed_dilate)))

    # 暗い線のつながった塊のうち、種に触れているものを全部まとめる
    _, lab = cv2.connectedComponents(line | seed)
    keep = np.unique(lab[seed > 0])
    keep = keep[keep > 0]
    region[np.isin(lab, keep)] = 1
    return region


DEFAULT_PARAMS = {
    "width": 1200,
    "height": 800,
    "center_range": [0, 300],
    "center_bias": 200,
    "th_25_35": 530,
    "cst_offset_x_35": -40,
    "cst_top_35": 510,
    "cst_radius_35": 400,
    "cst_offset_x_25": 15,
    "cst_top_25": 630,
    "cst_radius_25": 285,
    "cst_bottom": 800,
    "calib_x": 1022,
    "calib_y": 565,
    "canny1": 20000,
    "canny2": 45000,
    "th_score": 110,
    "th_score_black": 20,
    "th_bright_mean": 135,
    "th_black_level": 18,
    "template_threshold": 0.0,
    # ---- v2 の黒ワレ判定用 ----
    "black_inner_ratio": 1.0,   # 探す範囲の内側の円の半径（radius の何倍か）
    "black_gate_half": 40,      # 中央のゲート（成形線）として除外する幅の半分 [px]
    "black_bh_th": 20,          # ブラックハット（周りより暗い度合い）がこれより大きい画素を黒い線とみなす
    "black_join": 9,            # 途切れた線をつなぐ大きさ [px]
    "black_link": 25,           # 暗いと認めた線どうしがこの距離以内なら同じ割れとしてまとめる [px]
    "black_dark_ratio": 0.55,   # 線の中の暗い所（下位10%）が、プレートの明るさのこの割合以下なら本物の黒い線
    "black_rim_tol": 30,        # 線の上端のまわりこの距離以内（中心寄り）に白い縁取りがあれば「縁に届いた」 [px]
    "black_rim_bright": 0.5,    # 縁取りの白さの目安: プレートの明るさと、U字のまわりの上位1%の明るさの間のこの位置
    "black_follow_len": 120,    # 線の上の端から縁取りまで、暗い所をたどる最大の長さ [px]
    "black_follow_ratio": 0.80, # たどるとき「暗い」とみなす明るさ（プレートの明るさのこの割合以下）
    "black_follow_side": 4,     # たどるとき、横に何 px まで暗い所を探すか
    "black_follow_contrast": 15,  # まわりよりこれだけ暗ければ、割れの続きとみなす
    "black_follow_gap": 12,     # 暗くない所がこれ以上続いたら、つながっていないとみなす [px]
    "black_min_radial": 0.2,    # 線が中心から外へ向かって radius のこの割合以上伸びていること（縁に沿う影を除く）
    "black_outer_bottom": True,   # True なら ROI の下端も「外縁」に数える（黒ワレは左右のリブで終わるので既定は False）
    "black_edge_tol": 12,       # ROI の左右の端・下端からこの距離以内に届いていれば「外縁に届いた」
}

# pos2 は params_cm.json の camera_params["2"] 相当
POS2_PARAMS = dict(
    DEFAULT_PARAMS,
    th_25_35=440,
    cst_top_35=390,
    cst_top_25=490,
    cst_radius_35=330,
    cst_radius_25=235,
    cst_bottom=600,
    calib_x=906,
    calib_y=436,
    th_score=100,
)
POSITION_PARAMS = {"pos1": DEFAULT_PARAMS, "pos2": POS2_PARAMS}


class BaseCrackInspector:
    """InspectorCrackPos1/2のinspect処理をテスト用に切り出したもの"""

    top_key_25 = "cst_top_25"  # 2.5インチ側で使う top の設定名

    def __init__(self, params, template_path):
        self.params = params
        self.template_path = template_path
        self.template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if self.template is None:
            raise FileNotFoundError(
                "テンプレート画像を読み込めません: {}".format(template_path)
            )
        self.frame = None

    def _find_cassette_limits(self, ary_gray):
        p = self.params
        y_start, y_end = p["center_range"]
        xmean = cv2.subtract(
            ary_gray[y_start:y_end],
            p["center_bias"],
        ).mean(axis=0)
        xmean = cv2.morphologyEx(
            xmean,
            cv2.MORPH_CLOSE,
            np.ones([51, 1], np.uint8),
        )[:, 0]

        left_lim = np.argmax(xmean < 5)
        left_lim += np.argmax(xmean[left_lim:] > 10)
        right_lim = len(xmean) - 1 - np.argmax(xmean[::-1] < 5)
        right_lim -= np.argmax(xmean[right_lim::-1] > 10)
        cx = (left_lim + right_lim) // 2
        return left_lim, right_lim, cx

    def _match_template(self, frame):
        p = self.params
        result = cv2.matchTemplate(
            frame,
            self.template,
            cv2.TM_CCOEFF_NORMED,
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val < p["template_threshold"]:
            raise RuntimeError(
                "テンプレートマッチングスコアが低すぎます: {:.5f}".format(
                    max_val
                )
            )

        max_loc_rel = np.array(
            [
                max_loc[0] - p["calib_x"],
                max_loc[1] - p["calib_y"],
            ],
            dtype=np.int32,
        )
        return max_loc, max_loc_rel, max_val

    def _select_roi(self, left_lim, right_lim, max_loc_rel):
        p = self.params
        cassette_width = right_lim - left_lim

        if cassette_width > p["th_25_35"]:
            cassette_type = "3.5inch"
            left_lim -= p["cst_offset_x_35"]
            right_lim += p["cst_offset_x_35"]
            top_lim = p["cst_top_35"] + max_loc_rel[1]
            radius = p["cst_radius_35"]
        else:
            cassette_type = "2.5inch"
            left_lim -= p["cst_offset_x_25"]
            right_lim += p["cst_offset_x_25"]
            top_lim = p[self.top_key_25] + max_loc_rel[1]
            radius = p["cst_radius_25"]

        bottom_lim = p["cst_bottom"] + max_loc_rel[1]
        left_lim = max(0, left_lim)
        right_lim = min(self.frame.shape[1], right_lim)
        top_lim = max(0, top_lim)
        bottom_lim = min(self.frame.shape[0], bottom_lim)

        if left_lim >= right_lim or top_lim >= bottom_lim:
            raise RuntimeError(
                "ROIの範囲が不正です: x={}..{}, y={}..{}".format(
                    left_lim, right_lim, top_lim, bottom_lim
                )
            )

        return {
            "cassette_type": cassette_type,
            "left_lim": int(left_lim),
            "right_lim": int(right_lim),
            "top_lim": int(top_lim),
            "bottom_lim": int(bottom_lim),
            "radius": int(radius),
        }

    def _detect_bright_crack(
        self,
        ary_gray,
        left_lim,
        right_lim,
        top_lim,
        bottom_lim,
        cx,
        radius,
    ):
        p = self.params
        ary_interest = ary_gray[top_lim:bottom_lim, left_lim:right_lim]
        if ary_interest.size == 0:
            raise RuntimeError("明るいワレ検出用ROIが空です")

        clipped = ary_interest.clip(int(np.percentile(ary_interest, 50)), 255)
        closed = cv2.morphologyEx(
            clipped,
            cv2.MORPH_CLOSE,
            getker(11),
        )
        canny = cv2.Canny(
            closed,
            p["canny1"],
            p["canny2"],
            apertureSize=7,
        )
        dilated = cv2.dilate(canny, getker(3))

        contours = cv2.findContours(dilated, 0, 2)[-2]

        score = 0
        selected_contours = []
        for contour in contours:
            if contour[:, 0, 1].min() > 10:
                continue

            vx, vy, x, y = cv2.fitLine(
                contour,
                cv2.DIST_L2,
                0,
                0.01,
                0.01,
            )
            dist = np.abs(
                vx * (-radius - y)
                - vy * (cx - left_lim - x)
            )[0]

            if dist > radius * 1.1:
                dist = -cv2.pointPolygonTest(
                    contour,
                    (int(cx - left_lim), int(-radius)),
                    True,
                )
                if dist > radius * 1.15:
                    continue

            # 擦り傷はグレー、本物の明るいワレは光が透けて白飛びする
            if _contour_mean(ary_interest, contour) < p["th_bright_mean"]:
                continue

            contour_height = (
                contour[:, 0, 1].max() - contour[:, 0, 1].min()
            )
            score = max(score, contour_height)
            selected_contours.append(contour)

        return {
            "score": int(score),
            "is_ng": score > p["th_score"],
            "canny": canny,
            "dilated": dilated,
            "contours": selected_contours,
        }

    def _detect_black_crack(
        self,
        ary_gray,
        left_lim,
        right_lim,
        top_lim,
        bottom_lim,
        cx,
        radius,
    ):
        """黒ワレ判定 v2: U字の縁から外縁まで黒い線がつながっていれば NG。"""
        p = self.params
        y0 = max(0, top_lim - radius)
        ary_interest = ary_gray[y0:bottom_lim, left_lim:right_lim]
        if ary_interest.size == 0:
            raise RuntimeError("黒ワレ検出用ROIが空です")
        h, w = ary_interest.shape
        cxl = cx - left_lim                          # 切り出し画像の中での U字の中心 x
        cy_l = 0                                     # U字の円の中心 y（切り出し画像の上端）
        y_top_l = top_lim - y0                       # 元の ROI の上端（U字の底のあたり）

        # 探す範囲: U字の外側の帯（元と同じ円の帯）。中央はゲートの所だけ除く
        mask = np.zeros_like(ary_interest)
        cv2.circle(mask, (cxl, cy_l), int(radius * 1.75), 255, -1)
        # 内側の円は radius より少し小さくする（pos2 などで U字が円より小さいと、割れの根元が円の中に隠れるため）。
        # U字の開口部の中を拾わないことは、あとの「縁取りに届いているか」の判定で保証する
        cv2.circle(mask, (cxl, cy_l), int(radius * p["black_inner_ratio"]), 0, -1)
        mask[:radius // 2] = 0
        g = p["black_gate_half"]
        mask[y_top_l - 20:, max(0, cxl - g):cxl + g] = 0   # ゲートより下の成形線だけ除外

        # 細い線を残したまま、周りより暗い線を取り出す（元の CLOSE(11) は細い線を消すのでやめた）
        th_clip = int(np.percentile(ary_gray[top_lim:bottom_lim, left_lim:right_lim], 50))
        soft = cv2.GaussianBlur(ary_interest.clip(0, th_clip), (3, 3), 0)
        blackhat = np.max([
            cv2.morphologyEx(soft, cv2.MORPH_BLACKHAT, kern,
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
            for kern in BLACKHAT_KERNELS
        ], axis=0)
        binary = ((blackhat > p["black_bh_th"]) & (mask > 0)).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["black_join"], p["black_join"]))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k) & (mask > 0)

        # 周りのプレートの明るさ（相対的な暗さの基準）
        plate = float(np.median(ary_interest[mask > 0])) if (mask > 0).any() else 255.0

        yy, xx = np.mgrid[0:h, 0:w]
        dist_center = np.hypot(xx - cxl, yy - cy_l)
        # 縁に届いたかは局所的に見る: 線の一番中心寄りの点から中心へ向かって rim_tol px 以内に
        # 白く光る縁取り（プレートより十分明るい画素が 3px 続く所）があれば「縁に届いた」
        ring = (dist_center > 0.8 * radius) & (dist_center < 1.3 * radius) & (yy > radius // 2)
        top_bright = float(np.percentile(ary_interest[ring], 99)) if ring.any() else 255.0   # U字のまわりの一番明るい所
        bright_th = plate + p["black_rim_bright"] * (top_bright - plate)
        # 白く光る所のうち、U字の縁取りのように長く続く帯だけを「縁」とする（白い擦り傷や刻印は短いので除く）
        bn, blab, bstats, _ = cv2.connectedComponentsWithStats((ary_interest > bright_th).astype(np.uint8))
        big = np.zeros(bn, bool)
        big[1:] = np.maximum(bstats[1:, cv2.CC_STAT_WIDTH], bstats[1:, cv2.CC_STAT_HEIGHT]) >= 0.6 * radius
        rim_band = big[blab]

        def follow_to_rim(comp):
            """割れの根元（縁取りの近く）は細く薄くて途切れやすい。線の上の端から、線の向きに沿って
            中心の方へ暗い所をたどり、白い縁取りの帯まで暗いままつながっていれば True。"""
            ys, xs = np.nonzero(comp)
            d = dist_center[ys, xs]
            top = d <= np.percentile(d, 30)                     # 線の中心寄り 3割の点で向きを測る
            if top.sum() < 5:
                return False
            pts = np.c_[xs[top], ys[top]].astype(np.float32)
            vx, vy, x0_, y0_ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            i0 = int(np.argmin(d))
            x, y = float(xs[i0]), float(ys[i0])
            if vx * (cxl - x) + vy * (cy_l - y) < 0:            # 中心の方へ向ける
                vx, vy = -vx, -vy
            dark_th = p["black_follow_ratio"] * plate
            miss = 0
            for _ in range(p["black_follow_len"]):
                x, y = x + vx, y + vy
                # 進む向きに垂直に ±follow_side px 探し、一番暗い所に寄せる（曲がった割れにも沿う）
                best_v, best_xy = 1e9, (x, y)
                for t in range(-p["black_follow_side"], p["black_follow_side"] + 1):
                    qx, qy = int(round(x - t * vy)), int(round(y + t * vx))
                    if 0 <= qx < w and 0 <= qy < h and ary_interest[qy, qx] < best_v:
                        best_v, best_xy = float(ary_interest[qy, qx]), (qx, qy)
                qx, qy = int(round(best_xy[0])), int(round(best_xy[1]))
                if not (0 <= qx < w and 0 <= qy < h):
                    return False
                if rim_band[qy, qx] or (0 <= int(y) < h and 0 <= int(x) < w and rim_band[int(y), int(x)]):
                    return True                                  # 暗い線のまま縁取りの帯に着いた
                # まわり（横に ±follow_side*3 px）の明るさと比べて暗いか。根元は全体に明るく写ることがあるので相対で見る
                side = [ary_interest[int(round(y + t * vx)) % h, int(round(x - t * vy)) % w]
                        for t in range(-3 * p["black_follow_side"], 3 * p["black_follow_side"] + 1)]
                local = float(np.median(side))
                if best_v <= dark_th or best_v <= local - p["black_follow_contrast"]:
                    x, y = float(qx), float(qy)
                    miss = 0
                else:
                    miss += 1
                    if miss > p["black_follow_gap"]:
                        return False
            return False

        def touches_rim(comp):
            ys, xs = np.nonzero(comp)

            # 縁は U字のまわり（ROI の上端 = U字の底より上）にある。それより下の刻印の白い所を縁と間違えない
            if ys.min() > y_top_l + p["black_rim_tol"]:
                return False
            i = int(np.argmin(dist_center[ys, xs]))
            px, py = float(xs[i]), float(ys[i])
            # 割れの所は縁取りが切れて暗いので、真っすぐ中心方向だけでなく、上端のまわり（半径 rim_tol）の
            # 中心寄り半分に、白く光る縁取りが 15 画素以上あれば「縁に届いた」とする
            r = p["black_rim_tol"]
            x0, x1 = int(max(0, px - r)), int(min(w, px + r + 1))
            y0_, y1_ = int(max(0, py - r)), int(min(h, py + r + 1))
            gy, gx = np.mgrid[y0_:y1_, x0:x1]
            disc = np.hypot(gx - px, gy - py) <= r
            inward = dist_center[y0_:y1_, x0:x1] <= dist_center[int(py), int(px)] + 3
            return int((rim_band[y0_:y1_, x0:x1] & disc & inward).sum()) >= 15

        e = p["black_edge_tol"]
        near_outer = np.zeros((h, w), bool)
        near_outer[:, :e] = True
        near_outer[:, w - e:] = True
        if p["black_outer_bottom"]:
            near_outer[h - e:, :] = True                         # 下端も外縁に数える（既定は数えない）

        # 1) 線の一つ一つについて、周りより十分暗いか（汚れ・影は除く）
        n, lab, stats, _ = cv2.connectedComponentsWithStats(binary)
        strong = np.zeros((h, w), np.uint8)
        heights = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 30:
                continue
            comp = lab == i
            if float(np.percentile(ary_interest[comp], 10)) > p["black_dark_ratio"] * plate:
                continue
            strong[comp] = 1
            heights.append(int(stats[i, cv2.CC_STAT_HEIGHT]))
        contours = cv2.findContours(strong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # 2) 暗いと認めた線どうしが近ければ（black_link px 以内）同じ割れとしてまとめ、
        #    まとまりごとに「縁から外縁まで抜けているか」を見る
        L_ = p["black_link"]
        grp_n, grp = cv2.connectedComponents(
            cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L_, L_))))
        crossing = []
        for gi in range(1, grp_n):
            comp = (grp == gi) & (strong > 0)
            if not comp.any():
                continue
            dc = dist_center[comp]
            radial = float(dc.max() - dc.min())                  # 中心から外へ向かう長さ（縁に沿う影は小さい）
            ys_, xs_ = np.nonzero(comp)
            gate_line = np.mean(np.abs(xs_ - cxl) < p["black_gate_half"] * 2) > 0.5   # 大部分がゲートの真下（成形線）
            if (near_outer[comp].any() and radial >= p["black_min_radial"] * radius
                    and not gate_line and (touches_rim(comp) or follow_to_rim(comp))):
                crossing += cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)[-2]

        score_black = 0 if not heights else int(np.max(heights))
        is_ng = len(crossing) > 0
        logger.info(f"score_black:{score_black}, crossing:{len(crossing)}")
        region_contours = crossing if crossing else contours
        region_height = max([int(c[:, 0, 1].ptp()) for c in region_contours], default=0)

        return {
            "score_black": score_black,
            "is_ng": is_ng,
            "blackhat": blackhat,
            "binary": binary,
            "mask": mask,
            "contours": contours,
            "crossing_contours": crossing,         # NG の理由になった（縁から外縁まで抜けた）線
            "region_contours": crossing,           # 描画用（NG の線だけ描く）
            "region_height": region_height,
            "offset": (left_lim, y0),
        }

    def inspect(self, frame, out_frame=None, force=False):
        """画像1枚に対して、元inspectの検査処理を実行する。"""
        if frame is None:
            raise ValueError("入力画像がNoneです")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "入力画像はBGRカラー画像である必要があります: {}".format(
                    frame.shape
                )
            )

        self.frame = frame
        ary_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        left_lim, right_lim, cx = self._find_cassette_limits(ary_gray)
        max_loc, max_loc_rel, template_score = self._match_template(frame)
        roi = self._select_roi(left_lim, right_lim, max_loc_rel)

        bright_result = self._detect_bright_crack(
            ary_gray,
            roi["left_lim"],
            roi["right_lim"],
            roi["top_lim"],
            roi["bottom_lim"],
            cx,
            roi["radius"],
        )
        black_result = self._detect_black_crack(
            ary_gray,
            roi["left_lim"],
            roi["right_lim"],
            roi["top_lim"],
            roi["bottom_lim"],
            cx,
            roi["radius"],
        )

        is_ng = bright_result["is_ng"] or black_result["is_ng"]
        info = {
            "judge": "NG" if is_ng else "OK",
            "is_ng": bool(is_ng),
            "score": bright_result["score"],
            "score_black": black_result["score_black"],
            "black_region_height": black_result["region_height"],
            "template_score": float(template_score),
            "template_location": [int(max_loc[0]), int(max_loc[1])],
            "template_location_relative": [
                int(max_loc_rel[0]),
                int(max_loc_rel[1]),
            ],
            "cassette_type": roi["cassette_type"],
            "cassette_width": int(right_lim - left_lim),
            "center_x": int(cx),
            "roi": {
                "left": roi["left_lim"],
                "right": roi["right_lim"],
                "top": roi["top_lim"],
                "bottom": roi["bottom_lim"],
                "radius": roi["radius"],
            },
            "thresholds": {
                "th_score": int(self.params["th_score"]),
                "th_score_black": int(self.params["th_score_black"]),
                "th_bright_mean": int(self.params["th_bright_mean"]),
                "th_black_level": int(self.params["th_black_level"]),
            },
        }

        if out_frame is not None:
            self._draw_debug_image(
                out_frame,
                frame,
                roi,
                bright_result,
                max_loc,
                is_ng,
                black_result,
            )

        return info

    def _draw_debug_image(
        self,
        out_frame,
        frame,
        roi,
        bright_result,
        max_loc,
        is_ng,
        black_result=None,
    ):
        out_frame[:] = frame
        left = roi["left_lim"]
        right = roi["right_lim"]
        top = roi["top_lim"]
        bottom = roi["bottom_lim"]
        color = (0, 0, 255) if is_ng else (0, 255, 0)

        cv2.rectangle(
            out_frame,
            (left, top),
            (right - 1, bottom - 1),
            color,
            3,
        )
        cv2.drawMarker(
            out_frame,
            max_loc,
            (255, 0, 0),
            cv2.MARKER_CROSS,
            30,
            2,
        )

        for contour in bright_result["contours"]:
            contour_shifted = contour.copy()
            contour_shifted[:, 0, 0] += left
            contour_shifted[:, 0, 1] += top
            cv2.drawContours(
                out_frame,
                [contour_shifted],
                -1,
                (0, 0, 255),
                2,
            )

        # 追加: 黒ワレは、広げた割れ全体の領域をピンクで描く
        if black_result is not None:
            ox, oy = black_result["offset"]
            for contour in black_result["region_contours"]:
                contour_shifted = contour.copy()
                contour_shifted[:, 0, 0] += ox
                contour_shifted[:, 0, 1] += oy
                cv2.drawContours(
                    out_frame,
                    [contour_shifted],
                    -1,
                    (255, 0, 255),
                    2,
                )


class InspectorCrackPos1(BaseCrackInspector):
    """InspectorCrackPos1の単一画像テスト用クラス"""

    pass


class InspectorCrackPos2(BaseCrackInspector):
    """InspectorCrackPos2の単一画像テスト用クラス"""

    # 本番の InspectorCrackPos2 は 2.5インチ側でも cst_top_35 を使う
    top_key_25 = "cst_top_35"


def create_inspector(position, template_path, params):
    if position == "pos1":
        return InspectorCrackPos1(params, template_path)
    if position == "pos2":
        return InspectorCrackPos2(params, template_path)
    raise ValueError("positionはpos1またはpos2を指定してください")


@click.command()
@click.argument(
    "position",
    type=click.Choice(["pos1", "pos2"]),
)
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--template", "template_path", default=None)
@click.option("--save-debug", "save_debug", default=None)
@click.option(
    "--template-threshold",
    type=float,
    default=0.0,
    show_default=True,
)
def main(position, image_path, template_path, save_debug, template_threshold):
    """InspectorCrackPos1/Pos2の画像1枚検査"""
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise click.ClickException(
            "入力画像を読み込めません: {}".format(image_path)
        )

    params = POSITION_PARAMS[position].copy()
    params["template_threshold"] = template_threshold

    if template_path is None:
        template_name = {
            "pos1": "template_table1.png",
            "pos2": "template_table2.png",
        }[position]
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            template_name,
        )

    try:
        inspector = create_inspector(
            position,
            template_path,
            params,
        )
        debug_image = np.zeros_like(frame) if save_debug else None
        info = inspector.inspect(
            frame,
            out_frame=debug_image,
            force=True,
        )
    except Exception as exc:
        traceback.print_exc()
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(info, ensure_ascii=False, indent=2))

    if save_debug and debug_image is not None:
        save_dir = os.path.dirname(os.path.abspath(save_debug))
        os.makedirs(save_dir, exist_ok=True)
        if not cv2.imwrite(save_debug, debug_image):
            raise click.ClickException(
                "デバッグ画像を保存できません: {}".format(save_debug)
            )
        click.echo("debug image: {}".format(save_debug))


if __name__ == "__main__":
    main()
