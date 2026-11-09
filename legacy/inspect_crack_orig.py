#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ユーザーから受け取った既存の検査コード（比較用、変更しない）

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
    top_key_25 = "cst_top_25"

    def __init__(self, params, template_path):
        self.params = params
        self.template_path = template_path
        self.template = cv2.imread(template_path, cv2.IMREAD_COLOR) if template_path else None
        self.frame = None

    def _find_cassette_limits(self, ary_gray):
        p = self.params
        y_start, y_end = p["center_range"]
        xmean = cv2.subtract(ary_gray[y_start:y_end], p["center_bias"]).mean(axis=0)
        xmean = cv2.morphologyEx(xmean, cv2.MORPH_CLOSE, np.ones([51, 1], np.uint8))[:, 0]
        left_lim = np.argmax(xmean < 5)
        left_lim += np.argmax(xmean[left_lim:] > 10)
        right_lim = len(xmean) - 1 - np.argmax(xmean[::-1] < 5)
        right_lim -= np.argmax(xmean[right_lim::-1] > 10)
        cx = (left_lim + right_lim) // 2
        return left_lim, right_lim, cx

    def _select_roi(self, left_lim, right_lim, max_loc_rel):
        p = self.params
        cassette_width = right_lim - left_lim
        if cassette_width > p["th_25_35"]:
            cassette_type = "3.5inch"
            left_lim -= p["cst_offset_x_35"]; right_lim += p["cst_offset_x_35"]
            top_lim = p["cst_top_35"] + max_loc_rel[1]; radius = p["cst_radius_35"]
        else:
            cassette_type = "2.5inch"
            left_lim -= p["cst_offset_x_25"]; right_lim += p["cst_offset_x_25"]
            top_lim = p[self.top_key_25] + max_loc_rel[1]; radius = p["cst_radius_25"]
        bottom_lim = p["cst_bottom"] + max_loc_rel[1]
        left_lim = max(0, left_lim); right_lim = min(self.frame.shape[1], right_lim)
        top_lim = max(0, top_lim); bottom_lim = min(self.frame.shape[0], bottom_lim)
        return {"cassette_type": cassette_type, "left_lim": int(left_lim), "right_lim": int(right_lim),
                "top_lim": int(top_lim), "bottom_lim": int(bottom_lim), "radius": int(radius)}

    def _detect_black_crack(self, ary_gray, left_lim, right_lim, top_lim, bottom_lim, cx, radius):
        p = self.params
        ary_interest = ary_gray[max(0, top_lim - radius):bottom_lim, left_lim:right_lim]
        mask = np.zeros_like(ary_interest)
        cv2.circle(mask, (cx - left_lim, 0), int(radius * 1.75), 255, -1)
        cv2.circle(mask, (cx - left_lim, 0), radius, 0, -1)
        mask[:radius // 2] = 0
        mask[:, max(0, cx - left_lim - 250):cx - left_lim + 250] = 0
        th_clip = int(np.percentile(ary_gray[top_lim:bottom_lim, left_lim:right_lim], 50))
        clipped = ary_interest.clip(0, th_clip)
        closed = cv2.morphologyEx(clipped, cv2.MORPH_CLOSE, getker(11))
        blackhat = np.max([cv2.morphologyEx(closed, cv2.MORPH_BLACKHAT, kern,
                                            borderType=cv2.BORDER_CONSTANT, borderValue=0)
                           for kern in BLACKHAT_KERNELS], axis=0)
        binary = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)[1]
        cv2.bitwise_and(binary, mask, binary)
        contours = cv2.findContours(binary, 0, 2)[-2]
        contours = [c for c in contours if _contour_mean(ary_interest, c) <= p["th_black_level"]]
        heights = [c[:, 0, 1].max() - c[:, 0, 1].min() for c in contours]
        score_black = 0 if not heights else int(np.max(heights))
        return {"score_black": score_black, "is_ng": score_black > p["th_score_black"],
                "blackhat": blackhat, "binary": binary, "mask": mask, "contours": contours,
                "y0": max(0, top_lim - radius)}
