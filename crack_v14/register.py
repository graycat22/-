"""Reference registration: SIFT + RANSAC homography (production: small shift only, still works)."""
import cv2, numpy as np

_sift = cv2.SIFT_create(nfeatures=4000)

def _prep(gray):
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)

class Reference:
    def __init__(self, name, gray, mask=None):
        """mask: only use keypoints on the cassette itself (background fixtures are 3-D and mislead)."""
        self.name, self.gray = name, gray
        m = None if mask is None else (mask.astype(np.uint8) * 255)
        self.kp, self.des = _sift.detectAndCompute(_prep(gray), m)

def features(gray):
    return _sift.detectAndCompute(_prep(gray), None)

def register(gray, ref, ratio=0.75, feats=None):
    """returns (H img->ref, n_inliers); pass feats=features(gray) to reuse keypoints"""
    kp, des = feats if feats is not None else features(gray)
    if des is None or len(kp) < 10:
        return None, 0
    m = cv2.BFMatcher().knnMatch(des, ref.des, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < ratio * b.distance]
    if len(good) < 12:
        return None, len(good)
    src = np.float32([kp[g.queryIdx].pt for g in good])
    dst = np.float32([ref.kp[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return H, int(inl.sum()) if inl is not None else 0


def refine_ecc(gray, ref_gray, H, mask, iters=80):
    """Refine an initial homography with ECC on the cassette region (ref frame)."""
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
    # W maps ref -> warped coords; compose: img -> warped (H) -> ref (W^-1)
    return np.linalg.inv(W).astype(np.float64) @ H, float(cc)
