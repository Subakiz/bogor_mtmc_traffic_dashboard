#!/usr/bin/env python3
"""
Real-Time Car & Vehicle Tracker on Apple Neural Engine (ANE) using CoreML + ByteTrack.
Connects to any BSW Kota Bogor CCTV stream.
Features:
- Persistent Vehicle ID assignment (e.g. Car #14, Truck #3)
- Cumulative unique ID tracking (counts total cars that passed through)
- Trajectory path drawing for each tracked vehicle
- Supports any camera ID from cctv_streams.json (94 cameras available)
- Optimized for Apple Silicon Neural Engine (CPU_AND_NE)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO

# Vehicle classes in COCO: 2: car, 3: motorcycle, 5: bus, 7: truck
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_COLORS = {
    "car": (0, 255, 0),        # Vibrant Green
    "motorcycle": (255, 170, 0),# Sky Blue / Cyan
    "truck": (0, 165, 255),    # Orange
    "bus": (255, 0, 230),      # Magenta
}

def load_cctv_streams(json_path="cctv_streams.json"):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}

def draw_hud(frame, camera_name, active_counts, cumulative_counts, fps, ane_ms, total_tracked):
    """Draw an aesthetic HUD banner with camera metadata, ANE latency, and ID stats."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay box (Top Left)
    overlay = frame.copy()
    box_w = 460
    box_h = 210
    cv2.rectangle(overlay, (15, 15), (15 + box_w, 15 + box_h), (15, 15, 18), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + box_w, 15 + box_h), (0, 215, 255), 2)

    # Title & Camera Name
    cv2.putText(frame, "CCTV CAR TRACKER (Apple Neural Engine)", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 240, 255), 2)
    
    clean_cam = (camera_name[:34] + "..") if len(camera_name) > 36 else camera_name
    cv2.putText(frame, f"Cam: {clean_cam}", (25, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # Hardware & Performance info
    cv2.putText(frame, f"ANE: {ane_ms:.1f}ms | FPS: {fps:.1f} | Active Tracked: {total_tracked}", (25, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 210, 220), 1)

    # Divider line
    cv2.line(frame, (25, 104), (15 + box_w - 15, 104), (80, 80, 90), 1)

    # Section Headers
    cv2.putText(frame, "OBJECT", (25, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(frame, "ACTIVE IN FRAME", (135, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(frame, "CUMULATIVE (UNIQUE)", (295, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    y = 148
    for cls_name in ["car", "motorcycle", "truck", "bus"]:
        color = CLASS_COLORS[cls_name]
        active = active_counts.get(cls_name, 0)
        cumul = len(cumulative_counts.get(cls_name, set()))

        cv2.putText(frame, f"{cls_name.capitalize()}", (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        cv2.putText(frame, f"{active:3d}", (175, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(frame, f"{cumul:4d} unique", (325, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        y += 20


def track_stream(stream_url, camera_name="CCTV Camera", model_path="yolov8m.mlpackage",
                 conf=0.25, display=True, max_frames=None, save_video=None, snapshot_out=None):
    print("=" * 60)
    print(" Starting Car Tracking on Apple Neural Engine (ANE)")
    print(f" Camera: {camera_name}")
    print(f" Stream: {stream_url}")
    print(f" Model : {model_path}")
    print("=" * 60)

    # Load CoreML YOLO model
    model = YOLO(model_path)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Unable to open stream at {stream_url}")
        return

    # Track histories (ID -> deque of center coordinates for trajectory tails)
    track_history = defaultdict(lambda: deque(maxlen=24))
    
    # Cumulative unique ID registries
    cumulative_ids = {
        "car": set(),
        "motorcycle": set(),
        "truck": set(),
        "bus": set()
    }

    writer = None
    prev_time = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Stream dropped. Reconnecting...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue

            frame_idx += 1
            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now

            # Run tracking with ByteTrack
            t0 = time.perf_counter()
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
                conf=conf,
                verbose=False
            )
            ane_ms = (time.perf_counter() - t0) * 1000.0

            res = results[0]
            annotated = frame.copy()
            active_counts = defaultdict(int)

            if res.boxes and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = res.boxes.id.cpu().numpy().astype(int)
                classes = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()

                for box, track_id, cls_id, score in zip(boxes, track_ids, classes, confs):
                    cls_name = CLASS_NAMES.get(cls_id, "vehicle")
                    active_counts[cls_name] += 1
                    cumulative_ids[cls_name].add(track_id)

                    x1, y1, x2, y2 = box
                    center = (int((x1 + x2) / 2), int(y2 - 5))

                    # Update trajectory history
                    track_history[track_id].append(center)

                    color = CLASS_COLORS.get(cls_name, (0, 255, 0))

                    # Draw Bounding Box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Draw ID Label Badge
                    label = f"{cls_name.capitalize()} #{track_id} ({score:.2f})"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    badge_y1 = max(y1 - 18, 0)
                    cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 6, badge_y1 + 18), color, -1)
                    cv2.putText(annotated, label, (x1 + 3, badge_y1 + 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                    # Draw Trajectory Tail (motion vector trail)
                    points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                    if len(points) > 1:
                        cv2.polylines(annotated, [points], isClosed=False, color=color, thickness=2)
                        cv2.circle(annotated, center, 4, color, -1)

            total_active = sum(active_counts.values())
            draw_hud(annotated, camera_name, active_counts, cumulative_ids, fps, ane_ms, total_active)

            # Optional Video Recording
            if save_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(save_video, fourcc, 25.0, (w, h))
                writer.write(annotated)

            # Snapshot save if requested (save after trails have accumulated or at max_frames)
            target_snap_frame = max_frames if max_frames else 15
            if snapshot_out and frame_idx == target_snap_frame:
                cv2.imwrite(snapshot_out, annotated)
                print(f"Tracked snapshot with trajectories saved to: {snapshot_out}")

            # Console output
            if frame_idx % 10 == 0 or not display:
                print(f"[Frame {frame_idx:04d} | {fps:4.1f} FPS | ANE: {ane_ms:4.1f}ms] "
                      f"Active: Cars={active_counts['car']:2d}, Motos={active_counts['motorcycle']:2d}, "
                      f"Trucks={active_counts['truck']:2d} | "
                      f"Total Unique Cars Seen: {len(cumulative_ids['car'])}")

            if display:
                cv2.imshow("BSW Kota Bogor - Car Tracking (Apple Neural Engine)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    snap_name = f"snapshot_frame_{frame_idx}.jpg"
                    cv2.imwrite(snap_name, annotated)
                    print(f"Captured snapshot: {snap_name}")

            if max_frames and frame_idx >= max_frames:
                print(f"\nReached target frame limit ({max_frames}).")
                break

    except KeyboardInterrupt:
        print("\nStopping tracker...")
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Video saved to: {save_video}")
        if display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Car Tracking on Apple Neural Engine (ANE)")
    parser.add_argument("--cam-id", type=int, default=167, help="BSW CCTV Camera ID (e.g. 167, 79, 81)")
    parser.add_argument("--url", type=str, default=None, help="Direct HLS stream URL (overrides --cam-id)")
    parser.add_argument("--model", type=str, default="yolov8m.mlpackage", help="Path to CoreML .mlpackage")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run in terminal without GUI window")
    parser.add_argument("--frames", type=int, default=None, help="Process N frames and exit")
    parser.add_argument("--save-video", type=str, default=None, help="Save tracking output as MP4 video")
    parser.add_argument("--snapshot", type=str, default=None, help="Save annotated snapshot of frame 1 to file")
    parser.add_argument("--list-cams", action="store_true", help="List all available CCTV cameras and exit")
    args = parser.parse_args()

    streams = load_cctv_streams()

    if args.list_cams:
        print(f"\n{'ID':4s} | {'Camera Name':40s} | {'Stream URL'}")
        print("-" * 100)
        for cid, info in sorted(streams.items()):
            print(f"{cid:4d} | {info['name']:40s} | {info['stream_url']}")
        print(f"\nTotal: {len(streams)} cameras available.\n")
        return

    # Determine stream URL and camera name
    if args.url:
        stream_url = args.url
        cam_name = "Custom Stream"
    elif args.cam_id in streams:
        cam_info = streams[args.cam_id]
        stream_url = cam_info["stream_url"]
        cam_name = cam_info["name"]
    else:
        print(f"Camera ID {args.cam_id} not found in cctv_streams.json. Using default ID 167.")
        stream_url = "https://restreamer6.kotabogor.go.id/memfs/fd56d0ac-e4c7-4915-87d3-0d8e59edef1b.m3u8"
        cam_name = "CCTV SIMPANG PAHLAWAN 2 -F"

    track_stream(
        stream_url=stream_url,
        camera_name=cam_name,
        model_path=args.model,
        conf=args.conf,
        display=not args.headless,
        max_frames=args.frames,
        save_video=args.save_video,
        snapshot_out=args.snapshot
    )

if __name__ == "__main__":
    main()
