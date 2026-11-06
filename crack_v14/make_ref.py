"""OK画像のフォルダから、検査に使う基準画像（refs/）を作る。最初の1回だけ実行する。

使い方:
  python -m crack_v14.make_ref --ok-dir pos1ok --template template_table1.png --out refs
  （このあと python -m crack_v14.draw_geom --ref-dir refs で検査範囲の線を引く）

やること:
  1. 各OK画像で template_table1.png を探し、X・Y のずれを補正して 960x660 に縮小する（検査のときと同じ）
  2. 輪郭の似ている画像どうしを同じ「製品の種類」にまとめる（type1, type2, ... の順に名前を付ける）
     すでに refs/ に基準画像があれば、似ている方の名前を引き継ぐ
  3. 種類ごとに、全部を重ねた画素ごとの中央値を基準画像 ref_<種類>.png にする
  4. テンプレートの位置などを info_<種類>.json に保存する

作り直し: 検査のたびに作る必要はない（cli は refs/ を読むだけ）。
  カメラや治具を動かした、照明を変えた、新しい種類が増えた、OK画像を増やしたいときだけ実行する。
  同じ --out に作り直すと、基準画像と info は上書きされ、検査範囲の線（geom_*.json）はそのまま残る。
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from .align import TemplateAligner, frame_matrix, to_frame

GROUP_TH = 0.25   # 輪郭の相関がこれより大きければ同じ種類とみなす


def load_gray(p):
    g = cv2.imread(p, 0)
    if g is None:
        raise SystemExit(f"読めない画像: {p}")
    return g


def edge_signature(frame):
    """種類の振り分け用: 縮小した画像の輪郭の強さ（平均0・標準偏差1にそろえたもの）。"""
    g = cv2.resize(frame, (240, 165), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return (e - e.mean()) / (e.std() + 1e-6)


def similarity(a, b):
    """2つの輪郭の相関（1 = 同じ、0 = 無関係）。"""
    return float((a * b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ok-dir", required=True, help="OK画像のフォルダ（種類が混ざっていてよい）")
    ap.add_argument("--template", required=True, help="template_table1.png")
    ap.add_argument("--out", default="refs")
    ap.add_argument("--max", type=int, default=40, help="種類ごとに使う最大枚数")
    a = ap.parse_args()

    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(a.ok_dir, f"*.{e}")))
    if not files:
        raise SystemExit(f"画像がありません: {a.ok_dir}")
    os.makedirs(a.out, exist_ok=True)
    aligner = TemplateAligner(load_gray(a.template))

    # ---- 1. 位置補正。先頭の画像のテンプレート位置を基準にする
    #      ただし、すでに refs/ に基準があるときはその位置を引き継ぐ（作り直しても検査範囲の線がずれないように）
    g0 = load_gray(files[0])
    old_info = sorted(glob.glob(os.path.join(a.out, "info_*.json")))
    if old_info:
        with open(old_info[0], encoding="utf-8") as fh:
            info0 = json.load(fh)
        tx0, ty0 = info0["tpl_xy"]
        sx0, sy0 = g0.shape[1] / info0["src_size"][0], g0.shape[0] / info0["src_size"][1]
        tx0, ty0 = tx0 * sx0, ty0 * sy0
        print(f"基準のテンプレート位置: ({tx0:.0f}, {ty0:.0f})  前回の基準から引き継ぎ")
    else:
        tx0, ty0, s0 = aligner.locate(g0)
        print(f"基準のテンプレート位置: ({tx0:.0f}, {ty0:.0f})  {os.path.basename(files[0])}  一致度 {s0:.2f}")
    frames = []
    for f in files:
        g = load_gray(f)
        x, y, s = aligner.locate(g)
        if s < 0.5:
            print(f"{os.path.basename(f)}: テンプレートが見つからないので除外（一致度 {s:.2f}）"); continue
        sx, sy = g.shape[1] / g0.shape[1], g.shape[0] / g0.shape[0]
        fr = to_frame(g, frame_matrix(g.shape, (tx0 * sx - x, ty0 * sy - y)))
        frames.append((f, fr, edge_signature(fr)))

    # ---- 2. 種類ごとにまとめる（すでにある基準画像があれば、その種類から始める）
    groups = {}                              # 名前 -> {"sigs": [...], "items": [...]}
    for p in sorted(glob.glob(os.path.join(a.out, "ref_*.png"))):
        name = os.path.basename(p)[4:-4]
        groups[name] = dict(sigs=[edge_signature(load_gray(p))], items=[])
    for f, fr, sig in frames:
        best, best_s = None, GROUP_TH
        for name, gr in groups.items():
            s = max(similarity(sig, t) for t in gr["sigs"])     # その種類の誰かと似ていれば同じ種類
            if s > best_s:
                best, best_s = name, s
        if best is None:
            best = f"type{len(groups) + 1}"
            groups[best] = dict(sigs=[], items=[])
        groups[best]["sigs"].append(sig)
        if len(groups[best]["items"]) < a.max:
            groups[best]["items"].append((f, fr))
    print("振り分け: " + ", ".join(f"{k}={len(v['items'])}枚" for k, v in sorted(groups.items()) if v["items"]))

    # ---- 3〜4. 種類ごとに中央値をとって保存
    for name, gr in sorted(groups.items()):
        if not gr["items"]:
            continue
        st = np.stack([fr for _, fr in gr["items"]]).astype(np.float32)
        med = np.median(st, axis=0)
        mad = np.median(np.abs(st - med), axis=0) * 1.4826     # OK品のばらつき（OK差分モデル用に保存）
        cv2.imwrite(os.path.join(a.out, f"ref_{name}.png"), np.clip(med, 0, 255).astype(np.uint8))
        np.save(os.path.join(a.out, f"ok_mad_{name}.npy"), mad)
        with open(os.path.join(a.out, f"info_{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(tpl_xy=[tx0, ty0], src_size=[g0.shape[1], g0.shape[0]], n_images=len(gr["items"]),
                           images=[os.path.basename(f) for f, _ in gr["items"]]), fh, ensure_ascii=False, indent=1)
        has_geom = os.path.exists(os.path.join(a.out, f"geom_{name}.json"))
        print(f"[{name}] {len(gr['items'])} 枚から作成 -> {a.out}/ref_{name}.png"
              + ("" if has_geom else f"   ※ 次に draw_geom で {name} の線を引いてください"))


if __name__ == "__main__":
    main()
