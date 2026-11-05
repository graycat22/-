"""Cut the monitor area out of a phone photo of the screen (only for the screen-photo test set)."""
import cv2, numpy as np

def crop_screen(im, W=960, H=660):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = g.shape[0]
    g2 = g.copy(); g2[: int(h * 0.31)] = 0; g2[int(h * 0.81):] = 0   # status bar / thumbnails
    m = (cv2.GaussianBlur(g2, (0, 0), 5) > 45).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    cnt, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnt, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    ap = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(ap) == 4:
        p = ap.reshape(4, 2).astype(np.float32)
        s = p.sum(1); d = np.diff(p, axis=1).ravel()
        src = np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)
    else:
        x, y, w, hh = cv2.boundingRect(c)
        src = np.float32([[x, y], [x + w, y], [x + w, y + hh], [x, y + hh]])
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(im, M, (W, H))
