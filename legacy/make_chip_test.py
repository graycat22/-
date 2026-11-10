#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""欠けの試験画像を作る: OK 画像の ROI の下側を半楕円にくり抜き、明るい背景で埋める。

使い方:
  python make_chip_test.py pos1 OK画像のフォルダ 出力フォルダ --size 0.5
  -> 出力フォルダの画像を eval_folders.py の NG フォルダとして使う
"""
import glob
import os

import click
import cv2
import numpy as np

import inspect_crack_v2 as V2


@click.command()
@click.argument("position", type=click.Choice(["pos1", "pos2"]))
@click.argument("ok_dir")
@click.argument("out_dir")
@click.option("--size", default=0.5, help="欠けの大きさ（ROI の幅・高さに対する割合）")
@click.option("--level", default=220, help="欠けた所に見える背景の明るさ")
def main(position, ok_dir, out_dir, size, level):
    os.makedirs(out_dir, exist_ok=True)
    params = V2.POSITION_PARAMS[position].copy()
    cls = V2.InspectorCrackPos1 if position == "pos1" else V2.InspectorCrackPos2
    insp = cls.__new__(cls)
    insp.params = params
    rng = np.random.default_rng(0)
    for f in sorted(glob.glob(os.path.join(ok_dir, "*"))):
        im = cv2.imread(f)
        if im is None:
            continue
        im = cv2.resize(im, (params["width"], params["height"]))
        insp.frame = im
        left, right, _ = insp._find_cassette_limits(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        roi = insp._select_roi(left, right, np.array([0, 0]))
        w, h = roi["right_lim"] - roi["left_lim"], roi["bottom_lim"] - roi["top_lim"]
        m = np.zeros(im.shape[:2], np.uint8)
        cv2.ellipse(m, ((roi["left_lim"] + roi["right_lim"]) // 2, roi["bottom_lim"]),
                    (int(w * size / 2), int(h * size)), 0, 180, 360, 255, -1)
        bg = np.clip(level + rng.normal(0, 15, im.shape[:2]), 0, 255).astype(np.uint8)
        im[m > 0] = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)[m > 0]
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), im)


if __name__ == "__main__":
    main()
