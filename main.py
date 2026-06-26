"""
main.py
-------
Entry point. Wires together:
    capture.py     -> webcam frames
    mesh.py         -> face landmark detection
    compositor.py    -> drawing every visual layer
    modules/filters.py (used indirectly via compositor)

Usage:
    python main.py
    python main.py --camera 1 --width 1920 --height 1080
    python main.py --max-faces 2 --no-smoothing
    python main.py --record output.mp4

Keyboard controls (while the window is focused):
    Q  -> quit
    S  -> save a screenshot to ./captures/
    R  -> start/stop recording to a video file
    M  -> toggle the dot-cloud mesh overlay on/off
    W  -> toggle the green wireframe mesh lines on/off (off by default)
    C  -> toggle landmark smoothing on/off
    V  -> toggle the Iron Man sunglasses overlay on/off (on by default)
    F  -> toggle the sparkle/confetti particle effect on/off (off by default)
    H  -> toggle the scanning HUD: face mask, scan line, reticles, panel, badge (on by default)
    1/2/3/4 -> switch reticle style (arrows / rings / dot / brackets)
"""

import argparse
import datetime
import sys
import time
from collections import deque
from pathlib import Path

import cv2

import config
from capture import Capture
from compositor import Compositor
from logger import get_logger
from mesh import Mesh, ModelNotFoundError
from modules.pose import LandmarkSmoother

log = get_logger(__name__)


class FPSCounter:
    """Rolling-window FPS counter (smoother than naive 1/dt)."""

    def __init__(self, window: int = 30):
        self._timestamps = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face Mesh AR Demo (Professional Edition)")
    parser.add_argument("--camera", type=int, default=config.DEFAULT_CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=config.DEFAULT_FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=config.DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--max-faces", type=int, default=config.MAX_NUM_FACES)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--record", type=str, default=None,
                         help="Path to immediately start recording, e.g. out.mp4")
    return parser.parse_args()


def make_video_writer(path: Path, w: int, h: int, fps: float = 30.0) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (w, h))


def main():
    args = parse_args()

    try:
        mesh = Mesh(max_num_faces=args.max_faces)
    except ModelNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    try:
        cam = Capture(args.camera, args.width, args.height)
    except RuntimeError as e:
        log.error(str(e))
        mesh.close()
        sys.exit(1)

    compositor = Compositor()
    fps_counter = FPSCounter()
    smoothers: list[LandmarkSmoother] = []
    smoothing_enabled = not args.no_smoothing
    frame_count = 0

    captures_dir = Path("captures")
    captures_dir.mkdir(exist_ok=True)

    video_writer = None
    recording = False
    if args.record:
        video_writer = make_video_writer(Path(args.record), args.width, args.height)
        recording = True

    log.info("Face Mesh AR running. Press Q in the window to quit.")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                log.warning("Failed to read frame from camera; stopping.")
                break

            result = mesh.process(frame)

            if len(smoothers) != result.num_faces:
                smoothers = [LandmarkSmoother(config.SMOOTHING_ALPHA) for _ in range(result.num_faces)]

            if smoothing_enabled:
                result.faces = [s.smooth(pts) for pts, s in zip(result.faces, smoothers)]

            fps = fps_counter.tick()
            compositor._smoothing_on = smoothing_enabled
            compositor.render(frame, result, frame_count, fps, recording)
            frame_count += 1

            if recording and video_writer is not None:
                video_writer.write(frame)

            cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('m'):
                compositor.show_dots = not compositor.show_dots
            elif key == ord('w'):
                compositor.show_wireframe = not compositor.show_wireframe
            elif key == ord('v'):
                compositor.show_visor = not compositor.show_visor
            elif key == ord('f'):
                compositor.show_sparkle = not compositor.show_sparkle
            elif key == ord('h'):
                compositor.show_scan_hud = not compositor.show_scan_hud
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                compositor.reticle.set_style(int(chr(key)) - 1)
                log.info("Reticle style -> %d", compositor.reticle.index)
            elif key == ord('c'):
                smoothing_enabled = not smoothing_enabled
                for s in smoothers:
                    s.reset()
            elif key == ord('s'):
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = captures_dir / f"capture_{stamp}.png"
                cv2.imwrite(str(out_path), frame)
                log.info("Screenshot saved -> %s", out_path)
            elif key == ord('r'):
                if not recording:
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = captures_dir / f"recording_{stamp}.mp4"
                    video_writer = make_video_writer(out_path, args.width, args.height)
                    recording = True
                    log.info("Recording started -> %s", out_path)
                else:
                    recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    log.info("Recording stopped.")

    finally:
        cam.release()
        if video_writer is not None:
            video_writer.release()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()