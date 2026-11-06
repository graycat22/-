"""検査範囲（U字の内縁・外縁・プレート）の定義。

座標はすべて「基準座標」で書く。基準座標とは、入力画像を位置補正してから
960x660 に縮小した画像の座標のこと（SIZE）。

  opening    : U字の開口部（中の背景が見える穴）。縁取りの外側の線までを含む多角形
  plate      : 割れが横切る樹脂のプレート部分（この中から opening を除いた所が検査範囲）
  outer      : 割れの「もう一方の端」。左右の縦リブと画像の下端の線
  rim_ignore : U字の縁取りが最初から切れている場所（ゲートの切り欠き）。縁取りの途切れ判定から外す
  y_eval     : これより上（U字の脚の上の方）は検査しない
  edge_margin: プレートの端から何px以内を「割れの証拠にしない」か（リブの縁の黒い線を拾わないため）
  min_thick  : 健全な肉厚がこれより薄い所は検査しない

検査範囲は draw_geom.py で基準画像の上をクリックして作り、refs/geom_<種類>.json に保存する。
下の DEFAULT_GEOM は、画面写真（テスト用）の上で描いた座標の例。
"""
import json
import os

import cv2
import numpy as np

SIZE = (960, 660)  # 基準座標の大きさ (幅, 高さ)

DEFAULT_GEOM = {
    # 製品の種類1: U字の脚が上まで長く見える
    "type1": dict(
        y_eval=380, edge_margin=14,
        opening=[(262, 60), (265, 350), (285, 420), (330, 470), (400, 497), (450, 502), (520, 497),
                 (590, 470), (630, 420), (652, 350), (652, 60)],
        plate=[(236, 60), (688, 60), (688, 660), (236, 660)],
        rim_ignore=[(440, 470, 510, 530)],
        outer=[[(686, 60), (686, 660)], [(238, 60), (238, 660)], [(236, 657), (688, 657)]],
    ),
    # 製品の種類2: U字の内側にネジ筒が見える（MADE IN CHINA / eDV 刻印）
    "type2": dict(
        y_eval=120, edge_margin=10, min_thick=30,
        opening=[(195, 0), (200, 250), (215, 300), (250, 360), (300, 402), (380, 428), (450, 434),
                 (530, 422), (592, 390), (640, 350), (690, 290), (720, 230), (736, 0)],
        plate=[(152, 0), (758, 0), (758, 660), (152, 660)],
        rim_ignore=[(430, 400, 520, 460)],          # 6時の位置のゲート切り欠き
        outer=[[(756, 0), (756, 660)], [(154, 0), (154, 660)], [(152, 657), (758, 657)]],
    ),
}


# ------------------------------------------------------------------ 読み書き
def geom_path(ref_dir, kind):
    return os.path.join(ref_dir, f"geom_{kind}.json")


def load_geom(ref_dir, kind):
    """refs/geom_<種類>.json を読む。無ければ None。"""
    p = geom_path(ref_dir, kind)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_geom(ref_dir, kind, g):
    with open(geom_path(ref_dir, kind), "w", encoding="utf-8") as f:
        json.dump(g, f, ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ マスク作成
def _poly(pts, shape, closed=True, thick=None):
    """多角形を塗りつぶした（thick 指定なら線で描いた）True/False の画像を作る。"""
    m = np.zeros(shape, np.uint8)
    p = np.round(np.array(pts, np.float64)).astype(np.int32)
    if thick is None:
        cv2.fillPoly(m, [p], 1)
    else:
        cv2.polylines(m, [p], closed, 1, thick)
    return m.astype(bool)


def build_masks(g, rim_band=10):
    """検査範囲の定義 g から、検査で使うマスク画像（基準座標）を作る。

    戻り値:
      opening : U字の開口部
      plate   : プレート全体
      band    : 割れが横切る樹脂部分（plate から opening を除いた所）
      inner   : 割れの起点を探す帯（開口部の縁のすぐ外側、幅 rim_band）
      outer   : 割れの終点（縦リブと下端の線）
    """
    shape = (SIZE[1], SIZE[0])
    opening = _poly(g["opening"], shape)
    plate = _poly(g["plate"], shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_band + 1,) * 2)
    open_dil = cv2.dilate(opening.astype(np.uint8), k).astype(bool)   # 開口部を rim_band だけ太らせる
    inner = open_dil & ~opening & plate
    band = plate & ~opening
    outer = np.zeros(shape, bool)
    for line in g["outer"]:
        outer |= _poly(line, shape, closed=False, thick=5)
    outer &= band & ~open_dil
    return dict(opening=opening, plate=plate, band=band, inner=inner, outer=outer)
