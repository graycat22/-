"""OK画像のフォルダから、検査に使う基準（refs/）を作る。最初の1回だけ実行する。

使い方:
  python -m crack_v14.make_ref --ok-dir pos1ok --template template_table1.png --out refs

やること:
  1. 各OK画像の製品の種類（type1 / type2）を判定する
     -> 仮の基準画像（data/refs/ref_type*.png、画面写真から作ったもの）と特徴点で照合して決める
  2. 種類ごとに、先頭のOK画像を「軸」にする
     -> 仮の基準画像の上で描いてある検査範囲の線を、軸の画像の座標に写して geom_<種類>.json に保存
        （特徴点の照合 = SIFT を使うのはこの1回だけ。検査のときは使わない）
  3. 残りのOK画像を、テンプレートマッチングで軸の画像に位置合わせする（検査のときと同じ方法）
  4. 全部を重ねて、画素ごとの中央値を基準画像 ref_<種類>.png にする
  5. check_<種類>.jpg に検査範囲を重ねて描く -> 目で確認する
       緑 = 割れの起点を探す所（U字内縁の縁取りのすぐ外側）
       赤 = 割れの終点（左右の縦リブと下端）
       青 = 割れを探す範囲（樹脂の部分）

線がずれていたら --move-geom DX DY（右・下に何px動かすか）と --scale-geom（何倍に広げるか）で直す。
仮の基準画像が無いときは --kind type1 のように種類を指定すると、DEFAULT の線をそのまま使う。
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from .align import TemplateAligner, frame_matrix, to_frame
from .geometry import DEFAULT_GEOM, KINDS, SIZE, adjust_matrix, build_masks, save_geom, transform_geom
from .inspect_v14 import Kind


def load_gray(p):
    g = cv2.imread(p, 0)
    if g is None:
        raise SystemExit(f"読めない画像: {p}")
    return g


def classify_with_old_refs(gray, olds):
    """仮の基準画像と特徴点で照合して、種類と「入力画像 -> 仮の基準画像」の変換を返す。"""
    from .register import Reference, features, refine_ecc, register
    f = min(1.0, 1.3 * max(SIZE) / max(gray.shape))         # 大きい画像は縮小してから照合（速さのため）
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    S = np.diag([f, f, 1.0])
    feats = features(small)
    best = None
    for name, (ref_obj, kind_obj) in olds.items():
        H, n = register(small, ref_obj, feats=feats)
        if H is None or n < 8:
            continue
        cas = cv2.dilate((kind_obj.cross.m["plate"] | kind_obj.cross.m["opening"]).astype(np.uint8),
                         np.ones((41, 41), np.uint8))
        Hf, _ = refine_ecc(gray, kind_obj.ref, H @ S, cas)
        s = kind_obj.shape_score(cv2.warpPerspective(gray, Hf, SIZE))
        if best is None or s > best[0]:
            best = (s, name, Hf)
    return (None, None) if best is None else (best[1], best[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ok-dir", required=True, help="OK画像のフォルダ（種類が混ざっていてよい）")
    ap.add_argument("--template", required=True, help="template_table1.png")
    ap.add_argument("--out", default="refs")
    ap.add_argument("--old-ref-dir", default="data/refs", help="仮の基準画像の場所")
    ap.add_argument("--kind", default=None, choices=KINDS, help="仮の基準画像なしで作るときの種類")
    ap.add_argument("--max", type=int, default=40, help="種類ごとに使う最大枚数")
    ap.add_argument("--move-geom", type=float, nargs=2, default=(0, 0), metavar=("DX", "DY"),
                    help="検査範囲の線を右にDX、下にDY px動かす（960x660座標）")
    ap.add_argument("--scale-geom", type=float, default=1.0, help="検査範囲の線を中心から何倍に広げるか")
    a = ap.parse_args()

    files = sorted(f for e in ("png", "jpg", "jpeg", "bmp") for f in glob.glob(os.path.join(a.ok_dir, f"*.{e}")))
    if not files:
        raise SystemExit(f"画像がありません: {a.ok_dir}")
    os.makedirs(a.out, exist_ok=True)
    aligner = TemplateAligner(load_gray(a.template))
    adjust = adjust_matrix(a.move_geom, a.scale_geom)

    # ---- 1. 種類ごとに振り分け（軸の画像については、仮の基準画像への変換も覚えておく）
    groups = defaultdict(list)       # 種類 -> [(ファイル名, 画像)]
    to_old = {}                      # 種類 -> 軸の画像 -> 仮の基準画像 の変換
    if a.kind is None:
        from .register import Reference
        olds = {}
        for k in KINDS:
            p = os.path.join(a.old_ref_dir, f"ref_{k}.png")
            if not os.path.exists(p):
                raise SystemExit(f"仮の基準画像がありません: {p}\n"
                                 f"  --old-ref-dir で場所を指定するか、--kind type1 などで作ってください")
            g = load_gray(p)
            olds[k] = (Reference(k, g), Kind(k, g, DEFAULT_GEOM[k], (0, 0), g.shape[::-1]))
        for f in files:
            g = load_gray(f)
            k, H = classify_with_old_refs(g, olds)
            if k is None:
                print(f"{os.path.basename(f)}: 種類を判定できないので除外"); continue
            if len(groups[k]) < a.max:
                if k not in to_old:
                    to_old[k] = H
                groups[k].append((f, g))
        print("振り分け: " + ", ".join(f"{k}={len(v)}枚" for k, v in sorted(groups.items())))
    else:
        groups[a.kind] = [(f, load_gray(f)) for f in files[: a.max]]
        print(f"仮の基準画像なし: 全部 {a.kind} とみなし、検査範囲は DEFAULT の線を使います")

    # ---- 2〜5. 種類ごとに基準を作る
    for k, items in sorted(groups.items()):
        f0, g0 = items[0]
        F0 = frame_matrix(g0.shape, (0, 0))                     # 軸の画像 -> 基準座標（縮小だけ）
        if k in to_old:   # 仮の基準画像の線 -> 軸の画像の基準座標
            geom = transform_geom(DEFAULT_GEOM[k], F0 @ np.linalg.inv(to_old[k]))
        else:
            geom = json.loads(json.dumps(DEFAULT_GEOM[k]))
        geom = transform_geom(geom, adjust)
        tx0, ty0, s0 = aligner.locate(g0)
        print(f"[{k}] 軸: {os.path.basename(f0)}（テンプレート一致度 {s0:.2f}）")

        stack = [to_frame(g0, F0)]
        for f, g in items[1:]:
            x, y, s = aligner.locate(g)
            if s < 0.5:
                print(f"[{k}] {os.path.basename(f)}: テンプレートが見つからないので除外（一致度 {s:.2f}）"); continue
            sx, sy = g.shape[1] / g0.shape[1], g.shape[0] / g0.shape[0]
            stack.append(to_frame(g, frame_matrix(g.shape, (tx0 * sx - x, ty0 * sy - y))))

        st = np.stack(stack).astype(np.float32)
        med = np.median(st, axis=0)
        mad = np.median(np.abs(st - med), axis=0) * 1.4826   # OK品のばらつき（OK差分モデル用に保存）
        ref = np.clip(med, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(a.out, f"ref_{k}.png"), ref)
        np.save(os.path.join(a.out, f"ok_mad_{k}.npy"), mad)
        save_geom(a.out, k, geom)
        with open(os.path.join(a.out, f"info_{k}.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(tpl_xy=[tx0, ty0], src_size=[g0.shape[1], g0.shape[0]], n_images=len(stack),
                           anchor=os.path.basename(f0)), fh, ensure_ascii=False, indent=1)

        m = build_masks(geom)
        vis = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        over = vis.copy(); over[m["band"]] = (200, 80, 0)
        vis = cv2.addWeighted(over, 0.25, vis, 0.75, 0)
        vis[m["inner"]] = (0, 220, 0); vis[m["outer"]] = (0, 0, 255)
        cv2.imwrite(os.path.join(a.out, f"check_{k}.jpg"), vis)
        print(f"[{k}] {len(stack)} 枚から作成 -> {a.out}/ref_{k}.png（{a.out}/check_{k}.jpg で線を確認）")

    for k in KINDS:
        if k not in groups:
            print(f"注意: {k} のOK画像が無かったので、{k} の基準は作っていません（{k} は検査されません）")


if __name__ == "__main__":
    main()
