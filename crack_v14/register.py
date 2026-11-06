"""特徴点による照合（SIFT）。make_ref.py で「仮の基準画像の線を実画像に写す」ときだけ使う。
検査のとき（cli.py / inspect_v14.py）は使わない。

  SIFT       : 画像の中から目印になる点（角や模様）を見つけ、その周りの見た目を数値（特徴量）にする
  照合       : 2枚の画像で特徴量が似ている点どうしを対応付ける
  MAGSAC     : 対応の中には間違いが混ざるので、それを捨てながら「どう動かせば重なるか」（ホモグラフィ）を求める
               （RANSAC の改良版）
  ECC        : 求めた重ね方を、画像全体の明るさの一致度が上がるように細かく直す
"""
import cv2
import numpy as np

_sift = cv2.SIFT_create(nfeatures=4000)


def _prep(gray):
    # 明るさのムラを均してから特徴点を探す（暗い所にも目印が見つかるように）
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)


class Reference:
    """照合の相手になる画像（特徴点を先に計算しておく）。"""

    def __init__(self, name, gray):
        self.name, self.gray = name, gray
        self.kp, self.des = _sift.detectAndCompute(_prep(gray), None)


def features(gray):
    """画像の特徴点と特徴量。"""
    return _sift.detectAndCompute(_prep(gray), None)


def register(gray, ref, ratio=0.75, feats=None):
    """gray を ref に重ねる変換（3x3）と、正しく対応した点の数を返す。失敗したら (None, 点の数)。"""
    kp, des = feats if feats is not None else features(gray)
    if des is None or len(kp) < 10:
        return None, 0
    m = cv2.BFMatcher().knnMatch(des, ref.des, k=2)
    # 1番目に似た点が2番目よりはっきり似ているものだけ残す（あいまいな対応を捨てる）
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < ratio * b.distance]
    if len(good) < 12:
        return None, len(good)
    src = np.float32([kp[g.queryIdx].pt for g in good])
    dst = np.float32([ref.kp[g.trainIdx].pt for g in good])
    cv2.setRNGSeed(0)                                       # 毎回同じ結果になるように
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, inl = cv2.findHomography(src, dst, method, 4.0, maxIters=10000, confidence=0.999)
    return H, int(inl.sum()) if inl is not None else 0


def refine_ecc(gray, ref_gray, H, mask, iters=80):
    """変換 H を ECC で微調整する（mask の範囲の明るさが一致するように）。"""
    h, w = ref_gray.shape
    warped = cv2.warpPerspective(gray, H, (w, h))
    a = cv2.GaussianBlur(_prep(ref_gray), (0, 0), 2).astype(np.float32)
    b = cv2.GaussianBlur(_prep(warped), (0, 0), 2).astype(np.float32)
    W = np.eye(3, dtype=np.float32)
    try:
        cc, W = cv2.findTransformECC(a, b, W, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5),
                                     mask.astype(np.uint8), 5)
    except cv2.error:
        return H, 0.0
    return np.linalg.inv(W).astype(np.float64) @ H, float(cc)
