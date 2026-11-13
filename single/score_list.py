#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_crack_single.py をフォルダの画像全部に回して、画像ごとの判定と score / score_black を CSV に出す。
閾値（th_score / th_score_black など）を決めるためのデータ集め用。判定ロジックには触らない。

置き方: 次の 4つを同じフォルダに置く
  inspect_crack_single.py / score_list.py / template_table1.png / template_table2.png

使い方（一番最後に画像のフォルダを指定する）:
  python score_list.py C:\\画像\\detected
  python score_list.py --pos pos2 C:\\画像\\pos2_detected
  python score_list.py --pos pos1 --debug C:\\画像\\pos1_ng      （デバッグ画像も保存する）
  python score_list.py --new --debug C:\\画像\\pos1_ng            （新しい方 inspect_crack_single_new.py で検査）

出力: score_list.py と同じフォルダに「<画像フォルダ名>_<pos>.csv」（--debug なら「<画像フォルダ名>_<pos>_debug」フォルダも）
"""
import csv
import glob
import os

import click
import cv2
import numpy as np
from loguru import logger

import inspect_crack_single
import inspect_crack_single_new

HERE = os.path.dirname(os.path.abspath(__file__))


def imread(path):
    """日本語を含むパスでも読めるように読み込む（cv2.imread は Windows で日本語パスを読めない）"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img)
    if ok:
        buf.tofile(path)


@click.command()
@click.option("--pos", "position", type=click.Choice(["pos1", "pos2"]), default="pos1", show_default=True)
@click.option("--out", default=None, help="CSV の保存先（省略時は <画像フォルダ名>_<pos>.csv）")
@click.option("--debug", is_flag=True, help="デバッグ画像（ROI と検出した輪郭）も保存する")
@click.option("--new", "use_new", is_flag=True, help="新しい方（欠け・ひび割れの判定を追加した inspect_crack_single_new.py）で検査する")
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
def main(position, out, debug, use_new, folder):
    S = inspect_crack_single_new if use_new else inspect_crack_single
    tag = position + ("_new" if use_new else "")
    logger.remove()                                   # 1枚ごとのログは出さない
    S.plt.show = lambda *a, **k: None                 # 明ワレの途中にある plt.show() で止まらないようにする
    template = os.path.join(HERE, {"pos1": "template_table1.png", "pos2": "template_table2.png"}[position])
    insp = S.create_inspector(position, template, S.DEFAULT_PARAMS.copy())

    name = os.path.basename(os.path.normpath(folder))
    out = out or os.path.join(HERE, "{}_{}.csv".format(name, tag))
    debug_dir = os.path.join(HERE, "{}_{}_debug".format(name, tag)) if debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp")
                   for f in set(glob.glob(os.path.join(folder, "*." + e)) + glob.glob(os.path.join(folder, "*." + e.upper()))))
    n_ng = 0
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["画像", "判定", "score(明ワレ)", "score_black(黒ワレ)", "template_score", "欠け面積", "ひび割れ高さ"])
        for f in files:
            frame = imread(f)
            if frame is None:
                w.writerow([os.path.basename(f), "ERROR", "", "", "読み込めない"])
                continue
            try:
                dbg = np.zeros_like(frame) if debug_dir else None
                info = insp.inspect(frame, out_frame=dbg, force=True)
            except Exception as exc:                  # 1枚の失敗で止めない
                w.writerow([os.path.basename(f), "ERROR", "", "", str(exc)])
                continue
            n_ng += info["is_ng"]
            w.writerow([os.path.basename(f), info["judge"], info["score"], info["score_black"],
                        round(info["template_score"], 3), info.get("chip_area", ""), info.get("line_height", "")])
            if debug_dir:
                imwrite(os.path.join(debug_dir, os.path.splitext(os.path.basename(f))[0] + ".jpg"), dbg)
    print("{} 枚中 NG {} 枚 -> {}".format(len(files), n_ng, out))


if __name__ == "__main__":
    main()
