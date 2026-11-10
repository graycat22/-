#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


# ---- v2 追加: 割れの定義「U字の縁から外縁まで抜けている」で判定するための関数 ----
def _long_bright_lines(ary, th, min_len):
    """周りより th 以上明るい細い線のうち、min_len px 以上続くもの（= U字の縁取り）"""
    tophat = cv2.morphologyEx(ary, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bright = cv2.morphologyEx((tophat > th).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright)
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= min_len
    return keep[lab]


def _link_to_rim(comp, ary, d_out, rim, plate):
    """線の内側の端から縁取りまでつながっていれば、その道筋 [(x, y), ...] を返す。だめなら None"""
    h, w = comp.shape
    ys, xs = np.nonzero(comp)
    d = d_out[ys, xs]
    inner = d >= np.percentile(d, 70)                     # 外縁から遠い 3割 = 内側の端
    if inner.sum() < 5:
        return None
    # (1) 内側の端から、線の向きに暗い所をたどって縁取りまで行けるか（割れの根元は薄く途切れやすい）
    vx, vy, _, _ = cv2.fitLine(np.c_[xs[inner], ys[inner]].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    x, y = float(xs[np.argmax(d)]), float(ys[np.argmax(d)])
    qx, qy = int(np.clip(round(x + 10 * vx), 0, w - 1)), int(np.clip(round(y + 10 * vy), 0, h - 1))
    if d_out[qy, qx] < d_out[int(y), int(x)]:             # 外縁から離れる向きにする
        vx, vy = -vx, -vy
    route, miss = [(int(x), int(y))], 0
    for _ in range(200):
        x, y = x + vx, y + vy
        cand = [(int(round(x - t * vy)), int(round(y + t * vx))) for t in range(-4, 5)]
        cand = [(cx_, cy_) for cx_, cy_ in cand if 0 <= cx_ < w and 0 <= cy_ < h]
        if not cand:
            break
        bx, by = min(cand, key=lambda q: ary[q[1], q[0]])   # 横 ±4px で一番暗い所に寄せる
        route.append((bx, by))
        if rim[by, bx] or (0 <= int(y) < h and 0 <= int(x) < w and rim[int(y), int(x)]):
            return route
        side = [ary[int(round(y + t * vx)) % h, int(round(x - t * vy)) % w] for t in range(-12, 13)]
        if ary[by, bx] <= 0.8 * plate or ary[by, bx] <= np.median(side) - 15:   # 周りより暗い = 割れの続き
            x, y, miss = float(bx), float(by), 0
        else:
            miss += 1
            if miss > 12:
                break
    # (2) たどれなくても、内側の端のまわり 30px に縁取りがあれば届いている
    for i in np.nonzero(inner)[0][::max(1, inner.sum() // 20)]:
        px, py = int(xs[i]), int(ys[i])
        y0, x0 = max(0, py - 30), max(0, px - 30)
        sub = rim[y0:py + 31, x0:px + 31]
        gy, gx = np.nonzero(sub)
        dist = np.hypot(gx + x0 - px, gy + y0 - py)
        if (dist <= 30).sum() >= 15:
            j = int(np.argmin(dist))
            return [(px, py), (int(gx[j] + x0), int(gy[j] + y0))]
    return None


def _find_black_crossing(ary, binary, mask, roi_lr, top_row, p):
    """黒い線のまとまりのうち、縁取りから外縁まで抜けているものの輪郭を返す"""
    h, w = ary.shape
    binary = ((binary > 0) & (mask > 0)).astype(np.uint8)
    # 縦リブの内側の溝: 端の近くで、上から下まで同じ列に続く暗い線は消す
    col = cv2.dilate(binary, np.ones((1, 7), np.uint8))[top_row:].mean(axis=0)
    gd = p["black_groove_dist"]
    groove = (col > 0.5) & ((np.arange(w) < gd) | (np.arange(w) >= w - gd))
    groove = cv2.dilate(groove.astype(np.uint8)[None], np.ones((1, 25), np.uint8))[0] > 0
    binary[:, groove] = 0
    # 外縁 = 溝か赤い ROI の左右の辺（内側の方）と、下端
    gl, gr = np.nonzero(groove[:w // 2])[0], w // 2 + np.nonzero(groove[w // 2:])[0]
    in_l = max(15, gl.max() + 1 if len(gl) else 0, roi_lr[0])
    in_r = min(w - 15, gr.min() if len(gr) else w, roi_lr[1])
    outer = np.zeros((h, w), bool)
    outer[:, :in_l + 12] = outer[:, in_r - 12:] = outer[h - 12:] = True
    d_out = cv2.distanceTransform((~outer).astype(np.uint8), cv2.DIST_L2, 5)

    plate = float(np.median(ary[mask > 0]))
    rim = _long_bright_lines(ary, p["black_rim_tophat"], p["black_rim_len"])        # はっきりした縁取り
    rim_soft = _long_bright_lines(ary, p["black_rim_tophat"] - 10, p["black_rim_len"])   # 弱いものも含む
    near_rim = cv2.dilate(rim_soft.astype(np.uint8), getker(31)) > 0
    d_rim = cv2.distanceTransform((cv2.morphologyEx(rim_soft.astype(np.uint8), cv2.MORPH_CLOSE, getker(31)) == 0)
                                  .astype(np.uint8), cv2.DIST_L2, 5)

    # 小さい点は捨てる。縁取りの 15px 以内は使わない（縁に沿う暗い輪郭を割れと間違えないため）
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary)
    keep = (lab > 0) & (st[lab, cv2.CC_STAT_AREA] >= 30) & ~near_rim
    strong = keep & (np.bincount(lab[keep], minlength=n)[lab] >= 30)
    # 近い線（25px 以内）を 1つの割れとしてまとめて、1つずつ調べる
    _, grp = cv2.connectedComponents(cv2.dilate(strong.astype(np.uint8), getker(25)))
    crossing = []
    for gi in range(1, grp.max() + 1):
        comp = (grp == gi) & strong
        ys, xs = np.nonzero(comp)
        if len(xs) < 30 or np.ptp(xs) < 25:              # 小さい / まっすぐ縦の線（リブの縁）
            continue
        far = comp & (d_rim > 30)                        # 縁取りから離れた部分
        fy, fx = np.nonzero(far)
        if len(fx) < 30 or np.hypot(np.ptp(fx), np.ptp(fy)) < 100 or not outer[far].any():
            continue                                     # 縁に沿っているだけ / 外縁に届かない
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.c_[xs, ys].T.astype(float))))
        cs, hier = cv2.findContours(comp.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        holes = sum(cv2.contourArea(c) for c, hh in zip(cs, hier[0]) if hh[3] >= 0)
        if ev[0] > 0.36 * ev[1] or holes > 0.1 * len(xs):  # 細長くない / 穴がある = 刻印・ロゴ
            continue
        route = _link_to_rim(comp, ary, d_out, rim, plate)
        if route:
            region = comp.astype(np.uint8)
            cv2.polylines(region, [np.array(route, np.int32)], False, 1, 5)   # 縁取りまでつないで描く
            crossing += cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return crossing


def _find_chip(ary):
    """欠け: ROI の下端からつながっている、プレートより明るい所（= 欠けて背景が見えている所）"""
    a = cv2.blur(ary, (7, 7))
    ref = float(np.median(a[:a.shape[0] // 3]))          # プレートの明るさ（ROI の上 1/3）
    bright = (a >= ref + 0.5 * (float(np.percentile(a, 99)) - ref)).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    _, lab = cv2.connectedComponents(bright)
    return np.isin(lab, np.unique(lab[-1])[1:]).astype(np.uint8)   # 下端の行に触れているもの


def _find_bright_crossing(ary, closed, p):
    """光が漏れて白く光る割れ目が、ROI の上端から下端まで続いていればその輪郭を返す"""
    h = ary.shape[0]
    med = float(np.median(ary))
    white = closed >= med + 0.65 * (np.percentile(ary, 99) - med)
    white &= ~cv2.dilate(_long_bright_lines(ary, p["black_rim_tophat"], p["black_rim_len"]).astype(np.uint8),
                         np.ones((7, 7), np.uint8)).astype(bool)    # U字の縁取りは除く
    white = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(white)
    found = []
    for i in range(1, n):
        top, hgt = st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_HEIGHT]
        widths = (lab == i).sum(axis=1)
        if top <= 30 and top + hgt >= h - 30 and np.median(widths[widths > 0]) >= p["bright_min_width"]:
            found += cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return found                                         # 幅が細いもの = 白い擦り傷は除く


BLACKHAT_KERNELS = [
    np.ones([1, 51], np.uint8),
    np.ones([51, 1], np.uint8),
    np.eye(35, dtype=np.uint8),
    np.fliplr(np.eye(35, dtype=np.uint8)).copy(),
]

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
    # v2 追加
    "black_groove_dist": 60,   # 端からこの距離以内の縦の暗い線は溝として消す [px]
    "black_rim_tophat": 25,    # 周りよりこれだけ明るい細い線を縁取りとする
    "black_rim_len": 200,      # 縁取りとみなす白い線の長さ [px]
    "bright_min_width": 25,    # 白い割れ目の幅の下限 [px]（白い擦り傷は細い）
    "th_chip": 0.03,           # 欠け: 下端からつながる明るい所が ROI のこの割合を超えたら NG
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

        crossing = _find_bright_crossing(ary_interest, closed, p)   # v2: 上端から下端まで白い割れ目
        selected_contours += crossing

        return {
            "score": int(score),
            "is_ng": score > p["th_score"] or len(crossing) > 0,
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
        p = self.params
        # v2: 赤い ROI ではなく、カセットの左右の端から端までを見る
        y0 = max(0, top_lim - radius)
        cst_left, cst_right, _ = self._find_cassette_limits(ary_gray)
        ary_interest = ary_gray[
            y0:bottom_lim,
            cst_left:cst_right,
        ]
        if ary_interest.size == 0:
            raise RuntimeError("黒ワレ検出用ROIが空です")

        # v2: 範囲は長方形（U字の円は使わない）。上の方・左右の端・ゲートの下だけ除く
        mask = np.full_like(ary_interest, 255)
        mask[:radius // 5] = 0
        mask[:, :15] = mask[:, -15:] = 0
        gx = cx - cst_left
        mask[top_lim - y0 - 20:, max(0, gx - 40):gx + 40] = 0

        th_clip = int(np.percentile(
            ary_gray[top_lim:bottom_lim, left_lim:right_lim],
            50,
        ))
        clipped = ary_interest.clip(0, th_clip)

        closed = cv2.GaussianBlur(clipped, (3, 3), 0)   # v2: CLOSE(11) は細い割れを消すのでやめた
        # 横長に加えて、縦・斜めの黒い割れも拾う(inspect_cm.py と同じ)
        blackhat = np.max([
            cv2.morphologyEx(
                closed,
                cv2.MORPH_BLACKHAT,
                kern,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for kern in BLACKHAT_KERNELS
        ], axis=0)
        binary = cv2.threshold(
            blackhat,
            20,
            255,
            cv2.THRESH_BINARY,
        )[1]
        cv2.bitwise_and(binary, mask, binary)

        contours = cv2.findContours(binary, 0, 2)[-2]
        # 汚れは濃いグレー、本物の黒ワレは隙間から真っ黒に見える
        contours = [
            contour for contour in contours
            if _contour_mean(ary_interest, contour) <= p["th_black_level"]
        ]
        heights = [
            contour[:, 0, 1].max() - contour[:, 0, 1].min()
            for contour in contours
        ]
        score_black = 0 if not heights else int(np.max(heights))
        logger.info(f"score_black:{score_black}, th:{p['th_score_black']}")

        # v2: 欠け（下が欠けて背景が見えている所）。欠けていたら NG は決まりで、
        #     明るい背景で明るさの基準がずれるので黒ワレは調べない
        chip = _find_chip(ary_gray[top_lim:bottom_lim, left_lim:right_lim])
        # v2: 判定は score_black ではなく「縁から外縁まで抜けた黒い線があるか」
        crossing = [] if chip.mean() > p["th_chip"] else _find_black_crossing(
            ary_interest, binary, mask,
            (left_lim - cst_left, right_lim - cst_left), radius // 5, p,
        )

        return {
            "score_black": score_black,
            "is_ng": len(crossing) > 0,
            "blackhat": blackhat,
            "binary": binary,
            "mask": mask,
            "contours": contours,
            "crossing": crossing,           # v2: 描画用
            "chip": chip,                   # v2: 欠け（赤い ROI の座標）
            "offset": (cst_left, y0),       # v2: 描画用
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

        chip = black_result["chip"]
        chip_ng = chip.mean() > self.params["th_chip"]   # v2: 下が大きく欠けていないか

        is_ng = bright_result["is_ng"] or black_result["is_ng"] or chip_ng
        info = {
            "judge": "NG" if is_ng else "OK",
            "is_ng": bool(is_ng),
            "score": bright_result["score"],
            "score_black": black_result["score_black"],
            "chip_area": round(float(chip.mean()), 3),   # v2: 欠けの面積（ROI に対する割合）
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

        if black_result is not None:   # v2: 黒ワレの範囲を水色、割れをピンクで描く
            ox, oy = black_result["offset"]
            ys, xs = np.nonzero(black_result["mask"])
            cv2.rectangle(out_frame, (ox + xs.min(), oy + ys.min()), (ox + xs.max(), oy + ys.max()), (255, 255, 0), 2)
            cv2.drawContours(out_frame, black_result["crossing"], -1, (255, 0, 255), 2, offset=(ox, oy))
            if black_result["chip"].mean() > self.params["th_chip"]:   # 欠けはオレンジ
                chip_cs = cv2.findContours(black_result["chip"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                cv2.drawContours(out_frame, chip_cs, -1, (0, 165, 255), 3, offset=(left, top))


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
