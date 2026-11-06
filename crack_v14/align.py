"""位置合わせ（テンプレートマッチング）。

今の検査システムと同じ考え方:
  検査画像の中から template_table1.png（右側のネジ2つと黒い留め具）に似た所を
  cv2.matchTemplate で探し、基準画像のときの位置との差を X・Y の補正量にする。

カメラは固定なので、ずれは平行移動だけと考える（回転・拡大縮小は補正しない）。
補正したあと、検査の座標（基準座標 960x660）に縮小する。
"""
import cv2
import numpy as np

from .geometry import SIZE


class TemplateAligner:
    def __init__(self, template_gray, blur=1.5):
        """template_gray: template_table1.png をグレースケールで読んだもの。"""
        if template_gray is None:
            raise ValueError("テンプレート画像が読めません")
        self.blur = blur
        self.tpl = self._prep(template_gray)

    def _prep(self, g):
        # 細かいノイズや反射の差で一致度がぶれないよう、少しぼかしてから比べる
        return cv2.GaussianBlur(g, (0, 0), self.blur) if self.blur > 0 else g

    def locate(self, gray):
        """テンプレートが一番よく一致した位置（左上の x, y）と一致度（-1〜1、1が完全一致）を返す。"""
        res = cv2.matchTemplate(self._prep(gray), self.tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(res)   # 最大値とその位置
        return float(x), float(y), float(score)


def frame_matrix(image_shape, offset):
    """入力画像 -> 基準座標 への変換行列（3x3）。

    1. offset (dx, dy) だけ平行移動して、基準画像と同じ位置にそろえる
    2. 960x660 に縮小する
    """
    h, w = image_shape[:2]
    sx, sy = SIZE[0] / w, SIZE[1] / h
    dx, dy = offset
    return np.array([[sx, 0, sx * dx], [0, sy, sy * dy], [0, 0, 1]], np.float64)


def to_frame(gray, M):
    """変換行列 M で入力画像を基準座標の画像（960x660）にする。"""
    return cv2.warpAffine(gray, M[:2], SIZE, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
