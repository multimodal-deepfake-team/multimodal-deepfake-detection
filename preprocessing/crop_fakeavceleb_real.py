import cv2
from pathlib import Path
from ultralytics import YOLO

SRC = Path(r"dataset/fakeavceleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/RealVideo-RealAudio")
DST = Path("processed/faces_v2/real_fakeavceleb")
DST.mkdir(parents=True, exist_ok=True)
FRAME_SKIP = 10
KEEP_EVERY = 3          # same as old pipeline: every 3rd sampled frame
MAX_PER_VIDEO = 12

det = YOLO("weights/yolov8n-face.pt")
total, no_face = 0, 0

for vp in sorted(SRC.rglob("*.mp4")):
    ident = vp.parent.name            # id00076
    vid = f"{ident}_{vp.stem}"        # id00076_00109
    cap = cv2.VideoCapture(str(vp))
    i, k, saved = 0, 0, 0
    while saved < MAX_PER_VIDEO:
        ok, frame = cap.read()
        if not ok:
            break
        if i % FRAME_SKIP == 0:
            if k % KEEP_EVERY == 0:
                r = det(frame, verbose=False)[0]
                if len(r.boxes) > 0:
                    x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0].tolist())
                    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    if crop.size:
                        cv2.imwrite(str(DST / f"{vid}_{saved:04d}.jpg"), crop)
                        saved += 1; total += 1
                else:
                    no_face += 1
            k += 1
        i += 1
    cap.release()

print("crops saved:", total, "| sampled frames with no face:", no_face)