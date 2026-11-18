"""
녹화한 영상에서 학습용 이미지를 뽑는 스크립트

사용법:
    python extract_frames.py 영상폴더 [--out frames] [--every 0.5] [--diff 12]

- 영상폴더 안의 모든 영상(mp4, mkv, mov, avi, webm)을 처리합니다.
- --every : 몇 초마다 한 장씩 뽑을지 (기본 0.5초)
- --diff  : 직전에 저장한 장면과 얼마나 달라야 저장할지 (0~255, 기본 12)
            값이 클수록 비슷한 장면을 더 많이 걸러냅니다.
- --crop  : 브라우저 화면 전체를 녹화했다면 auto 로 영상 부분만 잘라냅니다.
- 결과는 frames/영상이름_00001.jpg 형식으로 저장됩니다.

필요한 라이브러리:
    pip install opencv-python
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


def frame_signature(frame):
    """장면 비교용으로 작게 줄인 흑백 이미지"""
    small = cv2.resize(frame, (64, 36))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def detect_player_box(cap, samples=40):
    """화면 녹화에서 시간에 따라 바뀌는 영역(=영상 플레이어)을 찾아 (x, y, w, h) 반환"""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    std = np.std(np.stack(frames), axis=0)
    mask = std > 20
    cols = np.where(mask.mean(axis=0) > 0.3)[0]
    rows = np.where(mask.mean(axis=1) > 0.3)[0]
    if len(cols) == 0 or len(rows) == 0:
        return None
    x0, x1, y0, y1 = cols[0], cols[-1], rows[0], rows[-1]
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def extract(video_path: Path, out_dir: Path, every: float, diff_thresh: float, crop=None) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [건너뜀] 열 수 없음: {video_path.name}")
        return 0

    box = None
    if crop == "auto":
        box = detect_player_box(cap)
        print(f"  {video_path.name[:40]}: 자르는 영역 x,y,w,h = {box}")
    elif crop:
        box = tuple(int(v) for v in crop.split(","))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps * every)))
    saved, idx, last_sig = 0, 0, None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if box:
            x, y, w, h = box
            frame = frame[y:y + h, x:x + w]
        if idx % step == 0:
            sig = frame_signature(frame)
            # 직전 저장 장면과 너무 비슷하면(정지 화면 등) 저장하지 않음
            if last_sig is None or cv2.absdiff(sig, last_sig).mean() >= diff_thresh:
                saved += 1
                name = f"{video_path.stem}_{saved:05d}.jpg"
                cv2.imencode(".jpg", frame)[1].tofile(str(out_dir / name))  # 한글 경로 대응
                last_sig = sig
        idx += 1

    cap.release()
    return saved


def main():
    p = argparse.ArgumentParser(description="영상에서 학습용 이미지 추출")
    p.add_argument("video_dir", type=Path, nargs="+", help="녹화 영상 폴더 또는 영상 파일(여러 개 가능)")
    p.add_argument("--out", type=Path, default=Path("frames"), help="이미지 저장 폴더")
    p.add_argument("--every", type=float, default=0.5, help="몇 초마다 한 장 (기본 0.5)")
    p.add_argument("--diff", type=float, default=12, help="장면 변화 기준 (기본 12)")
    p.add_argument("--crop", default=None,
                   help="잘라낼 영역: auto(영상 플레이어 자동 감지) 또는 x,y,w,h")
    args = p.parse_args()

    videos = []
    for path in args.video_dir:
        if path.is_dir():
            videos += sorted(f for f in path.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        elif path.suffix.lower() in VIDEO_EXTS:
            videos.append(path)
    if not videos:
        print("영상 파일을 찾지 못했습니다:", args.video_dir)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for v in videos:
        n = extract(v, args.out, args.every, args.diff, args.crop)
        print(f"  {v.name}: {n}장")
        total += n
    print(f"\n완료! 총 {total}장 → {args.out.resolve()}")
    print("짱구/흰둥이/철수가 안 나오는 장면은 직접 지운 뒤 Roboflow에 올리세요.")


if __name__ == "__main__":
    main()
