"""割れらしさマップ（1画素ごとに「割れっぽさ」を 0〜1 の数値にする）。

  D（暗い割れ）: 周りより暗く、細長い所ほど高い
  B（明るい割れ）: 周りの樹脂より明るい所（割れ目から背面光が漏れている所）ほど高い
"""
import cv2
import numpy as np


def _robust_z(x, mask):
    """mask 内の典型的な値（中央値）から何倍ずれているかを測る（外れ値に強い標準化）。"""
    v = x[mask]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return (x - med) / mad


def dark_line_map(gray, band, scales=(1.5, 3.0, 5.0, 8.0), se_sizes=(13, 25, 37)):
    """D: 暗い割れ線らしさ。

    1) 相対的な暗さ: クロージング（暗い細い所を周りの明るさで埋めた画像）と元画像の差 ÷ 周りの明るさ。
       割れの幅は 3〜20px くらいまでばらつくので、大きさを変えて3回やって一番大きい値を使う。
    2) 細長さ: ヘッセ行列（明るさの曲がり方）の固有値の比。線なら1に近く、点やザラザラなら0に近い。
    1) × 2) が大きい所 = 周りより暗くて細長い所 = 割れ線っぽい所。
    """
    f = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    floor = 0.35 * np.percentile(f[band], 90)          # プレート下の暗い所でノイズを増幅しないための下限
    rel = np.zeros_like(f)
    for k in se_sizes:
        clo = cv2.morphologyEx(f, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        rel = np.maximum(rel, (clo - f) / np.maximum(clo, floor))
    elong = np.zeros_like(f)
    for s in scales:
        g = cv2.GaussianBlur(f, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)   # 横方向の2回微分
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)   # 縦方向の2回微分
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        t = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        l1 = 0.5 * (dxx + dyy + t)                       # 大きい方の固有値（線を横切る向きの曲がり）
        l2 = 0.5 * (dxx + dyy - t)                       # 小さい方の固有値（線に沿う向きの曲がり）
        e = np.where(l1 > 0, 1 - np.clip(np.abs(l2) / (l1 + 1e-6), 0, 1), 0)
        elong = np.maximum(elong, e)
    return rel * (0.3 + 0.7 * elong)


def bright_gap_map(gray, band):
    """B: 明るい割れらしさ。周りの樹脂の明るさより、どれだけ明るいか（中央値からのずれ）。"""
    f = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    bg = cv2.medianBlur(gray, 41).astype(np.float32)          # その周りの樹脂の明るさ
    th = f - np.minimum(bg, np.percentile(f[band], 60))
    return _robust_z(th, band)


def to_prob(x, lo, hi):
    """値 x を 0〜1 に直す（lo 以下は0、hi 以上は1、その間は直線）。"""
    return np.clip((x - lo) / (hi - lo), 0, 1)
