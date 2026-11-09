#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""元のロジック（inspect_crack.py）と改善版（inspect_crack_v2.py）を、フォルダの画像全部で比べる。

使い方:
  python eval_folders.py pos1 OK画像のフォルダ  NG画像のフォルダ  --template template_table1.png --out eval_pos1
  python eval_folders.py pos2 OKフォルダ NGフォルダ --template template_table2.png --out eval_pos2
  （テンプレートが無いときは --no-template。位置合わせのずれを 0 として動かす）

出力:
  画面   : OK フォルダで NG になった枚数（過検出）、NG フォルダで NG になった枚数（検出）を新旧で表示
  <out>/results.csv       : 画像ごとの判定・score・score_black（新旧）
  <out>/old/*.jpg, new/*.jpg : デバッグ画像（ピンク = 黒ワレ、赤 = 明るいワレ）
"""
import csv
import glob
import os

import click
import cv2
import numpy as np
from loguru import logger

import inspect_crack as OLD
import inspect_crack_v2 as NEW

logger.remove()       # 1枚ごとのログは出さない


def make(mod, position, template, no_template):
    params = mod.POSITION_PARAMS[position].copy()
    cls = mod.InspectorCrackPos1 if position == "pos1" else mod.InspectorCrackPos2
    if no_template:
        class NoTpl(cls):
            def __init__(self, params, template_path):
                self.params, self.template_path, self.template, self.frame = params, None, None, None

            def _match_template(self, frame):
                return (0, 0), np.array([0, 0], np.int32), 1.0
        return NoTpl(params, None)
    return cls(params, template)


def images(folder):
    return sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(folder, f"*.{e}")))


@click.command()
@click.argument("position", type=click.Choice(["pos1", "pos2"]))
@click.argument("ok_dir")
@click.argument("ng_dir")
@click.option("--template", default=None)
@click.option("--no-template", is_flag=True)
@click.option("--out", default="eval_out")
@click.option("--size", default="1200x800", help="入力をこの大きさにそろえる（実機と同じなら変更不要）")
def main(position, ok_dir, ng_dir, template, no_template, out, size):
    W, H = map(int, size.split("x"))
    ins = {"old": make(OLD, position, template, no_template), "new": make(NEW, position, template, no_template)}
    for k in ins:
        os.makedirs(os.path.join(out, k), exist_ok=True)
    rows = []
    count = {(k, lab): [0, 0] for k in ins for lab in ("OK", "NG")}     # [NG判定数, 枚数]
    for lab, folder in (("OK", ok_dir), ("NG", ng_dir)):
        for f in images(folder):
            frame = cv2.imread(f)
            if frame is None:
                continue
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))
            name = os.path.splitext(os.path.basename(f))[0]
            row = [lab, name]
            for k, insp in ins.items():
                dbg = np.zeros_like(frame)
                try:
                    info = insp.inspect(frame, out_frame=dbg)
                except Exception as exc:                       # 1枚の失敗で止めない
                    row += ["ERROR", "", ""]
                    print(f"{k} {name}: ERROR {exc}")
                    continue
                cv2.putText(dbg, f"{k} {info['judge']} b={info['score']} k={info['score_black']}",
                            (10, 40), 0, 1.2, (0, 255, 255), 3)
                cv2.imwrite(os.path.join(out, k, f"{lab}_{name}.jpg"), dbg)
                count[(k, lab)][0] += info["is_ng"]
                count[(k, lab)][1] += 1
                row += [info["judge"], info["score"], info["score_black"]]
            rows.append(row)
    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["正解", "画像", "旧_判定", "旧_score", "旧_score_black", "新_判定", "新_score", "新_score_black"])
        w.writerows(rows)
    for k in ins:
        ok_ng, ok_n = count[(k, "OK")]
        ng_ng, ng_n = count[(k, "NG")]
        print(f"{'旧' if k == 'old' else '新'}: 過検出 {ok_ng}/{ok_n}  検出 {ng_ng}/{ng_n}")
    changed = [r for r in rows if len(r) >= 8 and r[2] != r[5]]
    if changed:
        print("新旧で判定が変わった画像: " + " ".join(f"{r[0]}:{r[1]}({r[2]}->{r[5]})" for r in changed))


if __name__ == "__main__":
    main()
