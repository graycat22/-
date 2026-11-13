#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_crack_single.py をフォルダの画像全部に回して、画像ごとの score / score_black を CSV に出す。
閾値（th_score / th_score_black など）を決めるためのデータ集め用。判定ロジックには触らない。

使い方（inspect_crack_single.py と同じフォルダに置いて実行）:
  python score_list.py pos1 detectedのフォルダ --out detected_pos1.csv
  python score_list.py pos1 pos1_ngのフォルダ  --out ng_pos1.csv
"""
import csv
import glob
import os

import click
import cv2

import inspect_crack_single as S


@click.command()
@click.argument("position", type=click.Choice(["pos1", "pos2"]))
@click.argument("folder")
@click.option("--out", default="scores.csv")
@click.option("--template", default=None, help="省略時は inspect_crack_single.py と同じ場所の template_table1/2.png")
def main(position, folder, out, template):
    if template is None:
        here = os.path.dirname(os.path.abspath(S.__file__))
        template = os.path.join(here, {"pos1": "template_table1.png", "pos2": "template_table2.png"}[position])
    S.plt.show = lambda *a, **k: None            # 明ワレの途中にある plt.show() で止まらないようにする
    insp = S.create_inspector(position, template, S.DEFAULT_PARAMS.copy())
    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(folder, "*." + e)))
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["画像", "判定", "score(明ワレ)", "score_black(黒ワレ)", "template_score"])
        for f in files:
            try:
                info = insp.inspect(cv2.imread(f), force=True)
                w.writerow([os.path.basename(f), info["judge"], info["score"], info["score_black"],
                            round(info["template_score"], 3)])
            except Exception as exc:                  # 1枚の失敗で止めない
                w.writerow([os.path.basename(f), "ERROR", "", "", str(exc)])
    print("書き出しました:", out, len(files), "枚")


if __name__ == "__main__":
    main()
