import argparse
import time
from collections import deque
from typing import Dict, Optional

import cv2

from .config import DETECTION_CONF
from .hud import draw_hud
from .pipeline import count_by_class, draw_detections, update_trails
from .streams import load_cctv_streams
from .tracking import VehicleDetector

DEFAULT_CAM_ID = 167


def track_stream(stream_url: str, camera_name: str = "CCTV Camera",
                 model_path: str = "yolov8m.mlpackage", conf: float = DETECTION_CONF,
                 display: bool = True, max_frames: Optional[int] = None,
                 save_video: Optional[str] = None, snapshot_out: Optional[str] = None):
    """Single-camera tracker with an OpenCV window; handy for debugging the CV pipeline.
    For multi-camera rolling tracking use the dashboard's indexing daemon."""
    print("=" * 65)
    print("  BSW Kota Bogor CCTV Tracker  [Apple Neural Engine]")
    print(f"  Camera  : {camera_name}")
    print(f"  Stream  : {stream_url}")
    print(f"  Model   : {model_path}")
    print("=" * 65)

    detector = VehicleDetector(model_path, conf=conf)
    trails: Dict[int, deque] = {}

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream: {stream_url}")
        return

    writer = None
    prev_time = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Stream disconnected. Reconnecting in 1 second...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                detector.reset(0)
                continue

            frame_idx += 1
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            detections = detector.track(0, frame)
            counts = count_by_class(detections)
            update_trails(trails, detections)

            if display or save_video or snapshot_out:
                annotated = frame.copy()
                draw_detections(annotated, detections, trails)
                draw_hud(annotated, camera_name, counts, fps, detector.last_inference_ms)

            if save_video:
                if writer is None:
                    h_f, w_f = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        save_video, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w_f, h_f))
                writer.write(annotated)

            if snapshot_out and frame_idx == (max_frames or 20):
                cv2.imwrite(snapshot_out, annotated)
                print(f"[Snapshot] Saved annotated frame to: {snapshot_out}")

            if frame_idx % 10 == 0:
                print(f"[{frame_idx:05d}] {fps:5.1f} FPS | ANE: {detector.last_inference_ms:5.1f}ms | "
                      f"Cars={counts.get('car', 0):2d}, Motos={counts.get('motorcycle', 0):2d}, "
                      f"Trucks={counts.get('truck', 0):2d}, Buses={counts.get('bus', 0):2d}")

            if display:
                cv2.imshow("BSW Kota Bogor - Vehicle Tracker [ANE]", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    fn = f"snapshot_{frame_idx:05d}.jpg"
                    cv2.imwrite(fn, annotated)
                    print(f"[Saved] Snapshot: {fn}")

            if max_frames and frame_idx >= max_frames:
                break

    except KeyboardInterrupt:
        print("\nStopping tracker...")
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="BSW Kota Bogor Vehicle Tracker [ANE]")
    parser.add_argument("--cam-id",     type=int,   default=DEFAULT_CAM_ID, help="Camera ID from cctv_streams.json")
    parser.add_argument("--url",        type=str,   default=None, help="Direct HLS stream URL (overrides --cam-id)")
    parser.add_argument("--model",      type=str,   default="yolov8m.mlpackage", help="Path to CoreML .mlpackage")
    parser.add_argument("--conf",       type=float, default=DETECTION_CONF, help="YOLO confidence threshold")
    parser.add_argument("--headless",   action="store_true", help="Run without an OpenCV window")
    parser.add_argument("--frames",     type=int,   default=None, help="Process N frames and exit")
    parser.add_argument("--save-video", type=str,   default=None, help="Save MP4 output video to path")
    parser.add_argument("--snapshot",   type=str,   default=None, help="Save annotated snapshot frame")
    parser.add_argument("--list-cams",  action="store_true", help="List available cameras and exit")
    args = parser.parse_args()

    streams = load_cctv_streams()

    if args.list_cams:
        print(f"\n{'ID':>5}  {'Camera Name':<42}  Stream URL")
        print("-" * 110)
        for cid, info in sorted(streams.items()):
            print(f"{cid:>5}  {info['name']:<42}  {info['stream_url']}")
        print(f"\nTotal: {len(streams)} cameras available.\n")
        return

    if args.url:
        stream_url, cam_name = args.url, "Custom Stream"
    elif args.cam_id in streams:
        stream_url = streams[args.cam_id]["stream_url"]
        cam_name = streams[args.cam_id]["name"]
    else:
        raise SystemExit(f"Camera ID {args.cam_id} not found in cctv_streams.json")

    track_stream(
        stream_url=stream_url,
        camera_name=cam_name,
        model_path=args.model,
        conf=args.conf,
        display=not args.headless,
        max_frames=args.frames,
        save_video=args.save_video,
        snapshot_out=args.snapshot,
    )
