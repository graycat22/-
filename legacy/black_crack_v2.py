# -*- coding: utf-8 -*-
"""黒ワレ（暗い割れ）の判定 v2。inspect_crack_v2.py の _detect_black_crack から呼ぶ。

割れの定義: 「U字の内側（白く光る縁取り）から、外縁（左右のリブ・下端）まで、端から端まで割れている」。
半割れ・擦り傷・成形線・影は OK。

U字の形（中心・半径・円）は使わない。写真ごとのブレが大きく、当てはめると外れるため。
代わりに、画像の中から「周りより明るい細い線が長く続く所」を縁取りとして拾う。

流れ:
  1. 探す範囲（長方形）を作る ............ make_search_mask
  2. 縦リブの内側の溝を消す .............. remove_grooves
  3. 縁取り（白く光る細い線）を拾う ...... find_rim
  4. 暗い線を、近いものどうしでまとめ、まとまりごとに
     「縁取りに届いている」かつ「外縁に届いている」かを見る ... find_crossing_cracks
"""
import cv2
import numpy as np

# 黒ワレ v2 の設定（DEFAULT_PARAMS に足して使う。pos1 / pos2 共通）
BLACK_V2_PARAMS = {
    # 探す範囲
    "black_top_cut": 0.2,         # 切り出しの上から radius のこの割合の行は見ない（カセットの上の縁）
    "black_edge_margin": 15,      # カセットの左右の端からこの幅は探さない（縦リブの黒い線） [px]
    "black_gate_half": 40,        # 中央のゲート（成形線）として除外する幅の半分 [px]
    # 暗い線
    "black_join": 9,              # 途切れた線をつなぐ大きさ [px]
    "black_dark_ratio": 0.55,     # 線の中の暗い所（下位10%）がプレートの明るさのこの割合以下なら本物の黒い線
    "black_link": 25,             # 線どうしがこの距離以内なら同じ割れとしてまとめる [px]
    # 縦リブの内側の溝
    "black_groove_dist": 60,      # 端からこの距離以内の列を溝の候補にする [px]
    "black_groove_frac": 0.5,     # 列の高さのこの割合以上が暗い列は溝として消す
    "black_groove_pad": 12,       # 溝の列の左右この幅も一緒に消す [px]
    "black_vline_width": 25,      # 左右のぶれがこれ未満のまっすぐ縦の線は、リブの縁の線として除く [px]
    # 縁取り（白く光る細い線）
    "black_rim_tophat": 25,       # 周りよりこれだけ明るい細い線を縁取りとする（「縁に届いた」の判定用）
    "black_rim_tophat_soft": 15,  # 縁沿いの暗い輪郭を見分ける用は、これだけ明るければ縁取りに数える
    "black_rim_len": 200,         # 白い線がこれ以上長ければ縁取りとみなす [px]
    "black_hug_dist": 15,         # 縁取りからこの距離以内の暗い画素は使わない（縁取りの暗い輪郭） [px]
    "black_rim_tol": 30,          # 線の内側の端のまわりこの距離以内に縁取りがあれば「縁に届いた」 [px]
    # 縁取りまでたどる（割れの根元は細く薄くて途切れやすい）
    "black_follow_len": 200,      # たどる最大の長さ [px]
    "black_follow_ratio": 0.80,   # 「暗い」とみなす明るさ（プレートの明るさのこの割合以下）
    "black_follow_side": 4,       # 横に何 px まで暗い所を探すか
    "black_follow_contrast": 15,  # まわりよりこれだけ暗ければ割れの続きとみなす
    "black_follow_gap": 12,       # 暗くない所がこれ以上続いたら、つながっていないとみなす [px]
    # 割れとみなす条件
    "black_min_len": 150,         # 線のまとまりの長さの下限 [px]
    "black_line_ratio": 0.6,      # 細長さ（短い方向の広がり ÷ 長い方向の広がり）がこれより大きければ塊（刻印・ロゴ）
    "black_hole_ratio": 0.1,      # 線が囲む穴の面積が、線の面積のこの割合より大きければ刻印・ロゴ
    "black_far_dist": 30,         # 縁取りからこれより離れた部分を「縁から離れた部分」とする [px]
    "black_far_len": 100,         # 縁から離れた部分・一本の線として続く部分の長さの下限 [px]
    "black_edge_tol": 12,         # 外縁からこの距離以内に来ていれば「外縁に届いた」 [px]
    "black_outer_bottom": True,   # 探す範囲の下端も外縁に数える
}


def make_search_mask(shape, radius, gate_x, gate_y, p):
    """1. 探す範囲: カセットの左右の端から端までの長方形。上の方とゲートの真下だけ除く。"""
    h, w = shape
    mask = np.full((h, w), 255, np.uint8)
    mask[:int(radius * p["black_top_cut"])] = 0                  # カセットの上の縁
    em = p["black_edge_margin"]
    mask[:, :em] = 0                                             # 左右の端（縦リブの黒い線）
    mask[:, max(0, w - em):] = 0
    g = p["black_gate_half"]
    mask[max(0, gate_y - 20):, max(0, gate_x - g):gate_x + g] = 0   # ゲートより下の成形線
    return mask


def remove_grooves(binary, top_row, p):
    """2. 縦リブの内側の溝を消す。溝は上から下まで同じ列に続き、割れは斜めに横切るので、
    端の近くで「列の black_groove_frac 以上が暗い」列を溝とする。
    戻り値: 溝を消した binary、溝の列（True/False）"""
    h, w = binary.shape
    gd = p["black_groove_dist"]
    wide = cv2.dilate(binary, np.ones((1, 7), np.uint8))         # 写真のガタつき（左右 ±3px）を吸収
    col_frac = wide[top_row:].mean(axis=0)
    near_edge = (np.arange(w) < gd) | (np.arange(w) >= w - gd)
    groove = (col_frac > p["black_groove_frac"]) & near_edge
    pad = p["black_groove_pad"]
    groove = cv2.dilate(groove.astype(np.uint8)[None, :], np.ones((1, 2 * pad + 1), np.uint8))[0] > 0
    binary = binary.copy()
    binary[:, groove] = 0
    return binary, groove


def _long_lines(bright, min_len):
    """白い画素のうち、縦か横に min_len 以上続く塊だけを残す。"""
    bright = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright)
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= min_len
    return keep[lab]


def find_rim(ary, p):
    """3. 縁取り = 周りより明るい細い線が長く続く所。明るさそのもので決めると、画面の写真では
    明るく写った面全体が入ってしまうので、トップハット（周り 15px より明るい分）で見る。
    戻り値: rim（はっきりした縁取り）、rim_soft（弱いものも含めた縁取り）"""
    tophat = cv2.morphologyEx(ary, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    rim = _long_lines(tophat > p["black_rim_tophat"], p["black_rim_len"])
    rim_soft = _long_lines(tophat > p["black_rim_tophat_soft"], p["black_rim_len"])
    return rim, rim_soft


def _touches_rim(comp, d_out, rim, p):
    """線の内側の部分（外縁から遠い方 3割）のどこかで、まわり（半径 rim_tol）に縁取りが 15 画素以上あれば True。"""
    h, w = comp.shape
    ys, xs = np.nonzero(comp)
    d = d_out[ys, xs]
    idx = np.nonzero(d >= np.percentile(d, 70))[0]
    r = p["black_rim_tol"]
    for i in idx[::max(1, len(idx) // 20)]:
        px, py = float(xs[i]), float(ys[i])
        x0, x1 = int(max(0, px - r)), int(min(w, px + r + 1))
        y0, y1 = int(max(0, py - r)), int(min(h, py + r + 1))
        gy, gx = np.mgrid[y0:y1, x0:x1]
        disc = np.hypot(gx - px, gy - py) <= r
        if int((rim[y0:y1, x0:x1] & disc).sum()) >= 15:
            return True
    return False


def _follow_to_rim(comp, ary, d_out, rim, plate, p):
    """線の内側の端（外縁から一番遠い所）から、線の向きに沿って外縁と反対の方へ暗い所をたどり、
    暗いまま縁取りに着けば True。割れの根元は細く薄くて、暗い線として途切れやすいため。"""
    h, w = comp.shape
    ys, xs = np.nonzero(comp)
    d = d_out[ys, xs]
    top = d >= np.percentile(d, 70)                              # 線の内側 3割の点で向きを測る
    if top.sum() < 5:
        return False
    vx, vy, _, _ = cv2.fitLine(np.c_[xs[top], ys[top]].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    i0 = int(np.argmax(d))
    x, y = float(xs[i0]), float(ys[i0])
    qx, qy = int(np.clip(round(x + 10 * vx), 0, w - 1)), int(np.clip(round(y + 10 * vy), 0, h - 1))
    if d_out[qy, qx] < d_out[int(y), int(x)]:                   # 外縁から遠ざかる向きにそろえる
        vx, vy = -vx, -vy
    side_n = p["black_follow_side"]
    miss = 0
    for _ in range(p["black_follow_len"]):
        x, y = x + vx, y + vy
        # 進む向きに垂直に ±side_n px 探し、一番暗い所に寄せる（曲がった割れにも沿う）
        best_v, best_xy = 1e9, (x, y)
        for t in range(-side_n, side_n + 1):
            qx, qy = int(round(x - t * vy)), int(round(y + t * vx))
            if 0 <= qx < w and 0 <= qy < h and ary[qy, qx] < best_v:
                best_v, best_xy = float(ary[qy, qx]), (qx, qy)
        qx, qy = int(round(best_xy[0])), int(round(best_xy[1]))
        if not (0 <= qx < w and 0 <= qy < h):
            return False
        if rim[qy, qx] or (0 <= int(y) < h and 0 <= int(x) < w and rim[int(y), int(x)]):
            return True                                          # 暗い線のまま縁取りに着いた
        # まわり（横に ±side_n*3 px）と比べて暗いか。根元は全体に明るく写ることがあるので相対で見る
        side = [ary[int(round(y + t * vx)) % h, int(round(x - t * vy)) % w]
                for t in range(-3 * side_n, 3 * side_n + 1)]
        if best_v <= p["black_follow_ratio"] * plate or best_v <= float(np.median(side)) - p["black_follow_contrast"]:
            x, y = float(qx), float(qy)
            miss = 0
        else:
            miss += 1
            if miss > p["black_follow_gap"]:
                return False
    return False


def find_crossing_cracks(ary, binary, mask, roi_lr, gate_x, top_row, p):
    """4. 縁取りから外縁まで抜けている暗い線のまとまりを探す。

    ary    : 切り出したグレー画像（カセットの左右の端から端まで）
    binary : 周りより暗い画素（ブラックハット > 20、探す範囲の中だけ）
    mask   : 探す範囲（make_search_mask）
    roi_lr : 元の赤い ROI の左右の辺（切り出し画像の x）。縦リブの内側の線の目安
    gate_x : ゲート（カセットの中心）の x
    top_row: これより上の行は見ない
    戻り値 : 割れと判定した線の輪郭のリスト（空なら OK）
    """
    h, w = ary.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["black_join"], p["black_join"]))
    binary = cv2.morphologyEx((binary > 0).astype(np.uint8), cv2.MORPH_CLOSE, k) & (mask > 0)
    binary, groove = remove_grooves(binary, top_row, p)

    # 外縁 = 縦リブの内側の線。溝が見つかればその溝、なければ元の赤い ROI の辺（どちらか内側の方）
    cols = np.nonzero(groove)[0]
    em, e = p["black_edge_margin"], p["black_edge_tol"]
    in_l = max(em, int(cols[cols < w // 2].max()) + 1 if (cols < w // 2).any() else 0, roi_lr[0])
    in_r = min(w - em, int(cols[cols >= w // 2].min()) if (cols >= w // 2).any() else w, roi_lr[1])
    near_outer = np.zeros((h, w), bool)
    near_outer[:, :max(0, in_l) + e] = True
    near_outer[:, max(0, in_r - e):] = True
    if p["black_outer_bottom"]:
        near_outer[h - e:, :] = True
    d_out = cv2.distanceTransform((~near_outer).astype(np.uint8), cv2.DIST_L2, 5)   # 外縁からの距離

    plate = float(np.median(ary[mask > 0])) if (mask > 0).any() else 255.0   # プレートの明るさ（暗さの基準）
    rim, rim_soft = find_rim(ary, p)
    hd = p["black_hug_dist"]
    rim_near = cv2.dilate(rim_soft.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hd + 1, 2 * hd + 1))) > 0
    # 縁取りからの距離（途切れを埋めてから測る）。縁に沿う暗い輪郭はずっと縁のそばにいる
    rim_closed = cv2.morphologyEx(rim_soft.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    d_rim = cv2.distanceTransform((rim_closed == 0).astype(np.uint8), cv2.DIST_L2, 5)

    # 暗い線の一つ一つ: 周りより十分暗いものだけ残す（汚れ・影は除く）。縁取りのすぐそばの画素は使わない
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary)
    strong = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 30:
            continue
        comp = lab == i
        if float(np.percentile(ary[comp], 10)) > p["black_dark_ratio"] * plate:
            continue
        comp &= ~rim_near
        if comp.sum() >= 30:
            strong[comp] = 1

    # 近い線どうしをまとめ、まとまりごとに割れの定義を満たすか見る
    L = p["black_link"]
    gn, grp = cv2.connectedComponents(cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L, L))))
    crossing = []
    for gi in range(1, gn):
        comp = (grp == gi) & (strong > 0)
        if not comp.any():
            continue
        ys, xs = np.nonzero(comp)
        if np.ptp(xs) < p["black_vline_width"]:                  # まっすぐ縦の細い線 = リブの縁の線
            continue
        if np.hypot(np.ptp(xs), np.ptp(ys)) < p["black_min_len"]:   # 短い
            continue
        if np.mean(np.abs(xs - gate_x) < p["black_gate_half"] * 2) > 0.5:   # 大部分がゲートの真下（成形線）
            continue
        # 縁取りから離れた部分だけで見ても、長さがあり外縁に届いていること（縁に沿う輪郭はここで落ちる）
        far = comp & (d_rim > p["black_far_dist"])
        if far.sum() < 30:
            continue
        fy, fx = np.nonzero(far)
        if np.hypot(np.ptp(fx), np.ptp(fy)) < p["black_far_len"] or not near_outer[far].any():
            continue
        # 細かい汚れや刻印が集まっただけのまとまりは割れではない。一本の線として十分長い部分があること
        sn, _, sst, _ = cv2.connectedComponentsWithStats(comp.astype(np.uint8))
        if max(np.hypot(sst[j, cv2.CC_STAT_WIDTH], sst[j, cv2.CC_STAT_HEIGHT]) for j in range(1, sn)) < p["black_far_len"]:
            continue
        # 刻印の文字やロゴを除く。割れは細長い線で、線が閉じて穴を作ることはほとんどない。
        # 文字やロゴは塊（細長くない）か、線が閉じて穴がある（リサイクルマークの三角、A・D・O など）
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.c_[xs, ys].T.astype(float))))
        if np.sqrt(max(ev[0], 0) / max(ev[1], 1e-6)) > p["black_line_ratio"]:   # 細長さ = 短い方向 ÷ 長い方向の広がり
            continue
        cs, hier = cv2.findContours(comp.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        holes = sum(cv2.contourArea(c) for c, hh in zip(cs, hier[0]) if hh[3] >= 0) if hier is not None else 0
        if holes > p["black_hole_ratio"] * comp.sum():
            continue
        # 内側の端が縁取りに届いているか（届いていなければ、暗い所をたどって届くか）
        if _touches_rim(comp, d_out, rim, p) or _follow_to_rim(comp, ary, d_out, rim, plate, p):
            crossing += cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return crossing
