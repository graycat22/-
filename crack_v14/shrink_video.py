"""動画を小さくする（共有用）。ffmpeg が無くても OpenCV だけで動く。

  python -m crack_v14.shrink_video IMG_8074.MOV small.mp4            # 横 640px・1秒2コマ
  python -m crack_v14.shrink_video IMG_8074.MOV small.mp4 --width 800 --fps 3

音声は入らない。100MB の動画でも、だいたい数MB になる。
"""
import argparse

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--width", type=int, default=640, help="出力の横幅 [px]")
    ap.add_argument("--fps", type=float, default=2.0, help="1秒あたり何コマ残すか")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.src)
    if not cap.isOpened():
        raise SystemExit(f"動画が開けません: {a.src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / a.fps)))
    out, i, n = None, 0, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            h = int(round(img.shape[0] * a.width / img.shape[1])) // 2 * 2
            img = cv2.resize(img, (a.width, h), interpolation=cv2.INTER_AREA)
            if out is None:
                out = cv2.VideoWriter(a.dst, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.width, h))
            out.write(img)
            n += 1
        i += 1
    cap.release()
    if out is None:
        raise SystemExit("コマが読めませんでした")
    out.release()
    print(f"{n} コマ -> {a.dst}")


if __name__ == "__main__":
    main()
