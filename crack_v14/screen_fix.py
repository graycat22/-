"""モニターをカメラで直接撮った写真を、普通の検査画像に近づける補正。

  python -m crack_v14.screen_fix <入力フォルダ> <出力フォルダ> [--ref <見本の画像フォルダ>]

やること:
  1. 切り出し : 写真の中から、モニターに映っている検査画像の四角を探し、まっすぐに直して 1260x840 にする
                （暗いモニターの枠より明るい、一番大きい四角い領域）
  2. モアレ除去: 画面の画素とカメラの画素がずれてできる細かい縞模様を、エッジを残すぼかしで消す
  3. 明るさ合わせ: 見本（--ref、補正済みの画像）の明るさの分布に合わせる（ヒストグラムマッチング）
"""
import argparse
import glob
import os

import cv2
import numpy as np

OUT_SIZE = (1260, 840)


def find_screen_quad(img):
    """写真の中の、表示されている検査画像の四隅（左上, 右上, 右下, 左下）を返す。見つからなければ None。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.GaussianBlur(g, (0, 0), 6)
    _, m = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)       # 明るい所 = 表示中の画像
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * img.shape[0] * img.shape[1]:
        return None
    hull = cv2.convexHull(c)
    for eps in (0.01, 0.02, 0.03, 0.05):
        ap = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(ap) == 4:
            p = ap.reshape(4, 2).astype(np.float32)
            break
    else:
        x, y, w, h = cv2.boundingRect(c)
        p = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.float32([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]])


def demoire(img):
    """細かい縞（モアレ）を消す。エッジ（割れ線・縁取り）は残す。"""
    x = cv2.bilateralFilter(img, 9, 40, 7)
    return cv2.medianBlur(x, 3)


def match_histogram(img, ref_cdf):
    """明るさの分布を見本（ref_cdf: 0〜255 の累積分布）に合わせる。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cdf = np.cumsum(np.bincount(g.ravel(), minlength=256)).astype(np.float64)
    cdf /= cdf[-1]
    lut = np.interp(cdf, ref_cdf, np.arange(256)).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.LUT(g, lut), cv2.COLOR_GRAY2BGR)


def reference_cdf(folder, n=20):
    """見本フォルダの画像の、明るさの累積分布の平均。"""
    fs = sorted(f for e in ("png", "jpg", "jpeg") for f in glob.glob(os.path.join(folder, f"*.{e}")))[:n]
    h = np.zeros(256)
    for f in fs:
        g = cv2.imread(f, 0)
        if g is not None:
            h += np.bincount(g.ravel(), minlength=256) / g.size
    c = np.cumsum(h)
    return c / c[-1] if c[-1] > 0 else None


def fix(img, ref_cdf=None):
    q = find_screen_quad(img)
    if q is None:
        return None
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [OUT_SIZE[0], 0], [OUT_SIZE[0], OUT_SIZE[1]], [0, OUT_SIZE[1]]]))
    out = cv2.warpPerspective(img, M, OUT_SIZE, flags=cv2.INTER_AREA)
    out = demoire(out)
    if ref_cdf is not None:
        out = match_histogram(out, ref_cdf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--ref", default=None, help="明るさを合わせる見本（補正済み画像のフォルダ）")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    ref = reference_cdf(a.ref) if a.ref else None
    for f in sorted(f for e in ("png", "jpg", "jpeg") for f in glob.glob(os.path.join(a.src, f"*.{e}"))):
        out = fix(cv2.imread(f), ref)
        name = os.path.splitext(os.path.basename(f))[0] + ".png"
        if out is None:
            print(f"{name}: 画面が見つからないので飛ばします"); continue
        cv2.imwrite(os.path.join(a.dst, name), out)
        print(f"{name}: 補正しました")


if __name__ == "__main__":
    main()
