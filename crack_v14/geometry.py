"""Hand-defined geometry in reference coordinates (960x660 reference frame).

opening : polygon of the U opening, up to the OUTER edge of the rim
plate   : polygon of the cassette body that a crack has to cross
outer   : polylines where a crack is considered to have reached the "other end"
"""
import cv2, numpy as np

SIZE = (960, 660)  # w, h

GEOM = {
    "D": dict(
        ref="s03", y_eval=120, edge_margin=10, min_thick=30,
        opening=[(195, 0), (200, 250), (215, 300), (250, 360), (300, 402), (380, 428), (450, 434),
                 (530, 422), (592, 390), (640, 350), (690, 290), (720, 230), (736, 0)],
        plate=[(152, 0), (758, 0), (758, 660), (152, 660)],
        rim_ignore=[(430, 400, 520, 460)],          # gate notch at 6 o'clock
        outer=[[(756, 0), (756, 660)], [(154, 0), (154, 660)], [(152, 657), (758, 657)]],
    ),
    "B": dict(
        ref="s12", y_eval=380, edge_margin=14,
        opening=[(262, 60), (265, 350), (285, 420), (330, 470), (400, 497), (450, 502), (520, 497),
                 (590, 470), (630, 420), (652, 350), (652, 60)],
        plate=[(236, 60), (688, 60), (688, 660), (236, 660)],
        rim_ignore=[(440, 470, 510, 530)],
        outer=[[(686, 60), (686, 660)], [(238, 60), (238, 660)], [(236, 657), (688, 657)]],
    ),
}

def _poly(pts, shape, closed=True, thick=None):
    m = np.zeros(shape, np.uint8)
    p = np.array(pts, np.int32)
    if thick is None:
        cv2.fillPoly(m, [p], 1)
    else:
        cv2.polylines(m, [p], closed, 1, thick)
    return m.astype(bool)

def build_masks(kind, rim_band=None):
    g = GEOM[kind]
    rim_band = g.get("rim_band", 10) if rim_band is None else rim_band
    shape = (SIZE[1], SIZE[0])
    opening = _poly(g["opening"], shape)
    plate = _poly(g["plate"], shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_band + 1,) * 2)
    open_dil = cv2.dilate(opening.astype(np.uint8), k).astype(bool)
    inner = open_dil & ~opening & plate                 # thin band just outside the rim
    band = plate & ~opening                             # material the crack must cross
    outer = np.zeros(shape, bool)
    for line in g["outer"]:
        outer |= _poly(line, shape, closed=False, thick=5)
    outer &= band & ~open_dil
    return dict(opening=opening, plate=plate, band=band, inner=inner, outer=outer)
