"""動画から一定の間隔でコマを取り出して、JPEG 画像として保存する（チャットに貼る用）。

  python -m crack_v14.extract_frames IMG_8074.MOV frames            # 2秒ごと
  python -m crack_v14.extract_frames IMG_8074.MOV frames --every 1  # 1秒ごと

1枚あたり 100〜200KB くらいの、横 1080px の画像になる。
"""
import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--every", type=float, default=2.0, help="何秒ごとに1枚取り出すか")
    ap.add_argument("--width", type=int, default=1080, help="保存する画像の横幅 [px]")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.src)
    if not cap.isOpened():
        raise SystemExit(f"動画が開けません: {a.src}")
    os.makedirs(a.out_dir, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * a.every)))
    i = n = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            if img.shape[1] > a.width:
                img = cv2.resize(img, (a.width, int(img.shape[0] * a.width / img.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(a.out_dir, f"f{n:03d}_t{i / fps:05.1f}s.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            n += 1
        i += 1
    cap.release()
    print(f"{n} 枚 -> {a.out_dir}/")


if __name__ == "__main__":
    main()
