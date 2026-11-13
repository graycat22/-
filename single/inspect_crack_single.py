#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import traceback

import click
import cv2
import matplotlib.pyplot as plt
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
}


class BaseCrackInspector:
    """InspectorCrackPos1/2のinspect処理をテストする最小基底クラス"""

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
            top_lim = p["cst_top_25"] + max_loc_rel[1]
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

        bg_detected = np.zeros_like(dilated)
        for each_contour in selected_contours:
            cv2.drawContours(bg_detected, [each_contour], -1, 255, -1)

        plt.imshow(bg_detected)
        plt.title("bg_detected")
        plt.show()

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
        p = self.params
        ary_interest = ary_gray[
            max(0, top_lim - radius):bottom_lim,
            left_lim:right_lim,
        ]
        if ary_interest.size == 0:
            raise RuntimeError("黒ワレ検出用ROIが空です")

        mask = np.zeros_like(ary_interest)
        cv2.circle(
            mask,
            (cx - left_lim, 0),
            int(radius * 1.75),
            255,
            -1,
        )
        cv2.circle(
            mask,
            (cx - left_lim, 0),
            radius,
            0,
            -1,
        )
        mask[:radius // 2] = 0
        mask[:, max(0, cx - left_lim - 250):cx - left_lim + 250] = 0

        th_clip = int(np.percentile(
            ary_gray[top_lim:bottom_lim, left_lim:right_lim],
            50,
        ))
        clipped = ary_interest.clip(0, th_clip)

        closed = cv2.morphologyEx(
            clipped,
            cv2.MORPH_CLOSE,
            getker(11),
        )
        blackhat = cv2.morphologyEx(
            closed,
            cv2.MORPH_BLACKHAT,
            np.ones([1, 51], np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
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

        return {
            "score_black": score_black,
            "is_ng": score_black > p["th_score_black"],
            "blackhat": blackhat,
            "binary": binary,
            "mask": mask,
            "contours": contours,
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


class InspectorCrackPos1(BaseCrackInspector):
    """InspectorCrackPos1の単一画像テスト用クラス"""

    pass


class InspectorCrackPos2(BaseCrackInspector):
    """InspectorCrackPos2の単一画像テスト用クラス"""

    pass


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

    params = DEFAULT_PARAMS.copy()
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
