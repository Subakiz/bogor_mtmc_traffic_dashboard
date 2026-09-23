#!/usr/bin/env python3
"""
CCTV Vehicle Counter on Apple Neural Engine (ANE) using CoreML YOLOv8
Camera: CCTV SIMPANG PAHLAWAN 2 -F (Kota Bogor)
"""

import argparse
import sys
import time
from collections import Counter
import cv2
from ultralytics import YOLO

STREAM_URL = "https://restreamer6.kotabogor.go.id/memfs/fd56d0ac-e4c7-4915-87d3-0d8e59edef1b.m3u8"
MODEL_PATH = "yolov8n.mlpackage"

# COCO vehicle & traffic classes
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
ALL_RELEVANT_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle", "person"}

def draw_overlay(frame, counts, inference_ms, fps=None):
    """Draw a clean HUD showing counts and ANE stats."""
    h, w, _ = frame.shape
    
    # Header overlay box
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (450, 190), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (20, 20), (450, 190), (0, 200, 255), 2)
    
    # Title
    cv2.putText(frame, "Apple Neural Engine (ANE) - YOLOv8", (35, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    
    # Performance metrics
    perf_text = f"Inference: {inference_ms:.1f} ms"
    if fps:
        perf_text += f" | FPS: {fps:.1f}"
    cv2.putText(frame, perf_text, (35, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    # Counts
    y = 105
    colors = {
        "car": (0, 255, 0),
        "motorcycle": (255, 150, 0),
        "truck": (0, 165, 255),
        "bus": (255, 0, 255),
        "bicycle": (0, 255, 255),
        "person": (200, 200, 200)
    }
    
    total_vehicles = sum(counts.get(cls, 0) for cls in VEHICLE_CLASSES)
    cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (35, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    
    y += 25
    col1 = [f"Car: {counts.get('car', 0)}", f"Motorcycle: {counts.get('motorcycle', 0)}"]
    col2 = [f"Truck: {counts.get('truck', 0)}", f"Bus: {counts.get('bus', 0)}"]
    
    cv2.putText(frame, f"Cars: {counts.get('car', 0)}   Motorcycles: {counts.get('motorcycle', 0)}", (35, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    y += 22
    cv2.putText(frame, f"Trucks: {counts.get('truck', 0)}   Buses: {counts.get('bus', 0)}   Persons: {counts.get('person', 0)}", (35, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


def process_snapshot(model, source_url, output_image_path="cctv_snapshot_detected.jpg", conf=0.25):
    print(f"Connecting to stream: {source_url}")
    cap = cv2.VideoCapture(source_url)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Could not fetch frame from stream.")
        return

    # Run inference on ANE
    t0 = time.perf_counter()
    results = model(frame, conf=conf, verbose=False)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    res = results[0]
    counts = Counter()

    # Annotate frame
    annotated = frame.copy()
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        name = model.names[cls_id]
        score = float(box.conf[0].item())
        counts[name] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Draw box
        color = (0, 255, 0) if name == "car" else ((255, 150, 0) if name == "motorcycle" else (0, 200, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {score:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    draw_overlay(annotated, counts, inference_ms)
    cv2.imwrite(output_image_path, annotated)
    print(f"\nAnnotated snapshot saved to: {output_image_path}")
    print("\n--- Object Counts ---")
    for obj, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {obj:12s}: {count}")
    total_vehicles = sum(counts.get(cls, 0) for cls in VEHICLE_CLASSES)
    print(f"  {'TOTAL VEHICLES':12s}: {total_vehicles}")
    print(f"Inference latency: {inference_ms:.2f} ms (Apple Neural Engine)\n")


def process_live(model, source_url, conf=0.25, display=True, max_frames=None, save_video=None):
    print(f"Opening live stream: {source_url}")
    if display:
        print("Press 'q' in the video window to quit.\n")
    cap = cv2.VideoCapture(source_url)

    prev_time = time.time()
    frame_count = 0
    writer = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Reconnecting to stream...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(source_url)
                continue

            frame_count += 1
            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now

            t0 = time.perf_counter()
            results = model(frame, conf=conf, verbose=False)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            res = results[0]
            counts = Counter()
            annotated = frame.copy()

            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                name = model.names[cls_id]
                score = float(box.conf[0].item())
                counts[name] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = (0, 255, 0) if name == "car" else ((255, 150, 0) if name == "motorcycle" else (0, 200, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{name} {score:.2f}", (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            draw_overlay(annotated, counts, inference_ms, fps)

            if save_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(save_video, fourcc, 25.0, (w, h))
                writer.write(annotated)

            if frame_count % 10 == 0 or not display:
                total_v = sum(counts.get(cls, 0) for cls in VEHICLE_CLASSES)
                print(f"[Frame {frame_count:04d} | {fps:4.1f} FPS | ANE: {inference_ms:4.1f}ms] "
                      f"Cars: {counts.get('car', 0):2d} | Motorcycles: {counts.get('motorcycle', 0):2d} | "
                      f"Trucks: {counts.get('truck', 0):2d} | Buses: {counts.get('bus', 0):2d} | Total: {total_v:2d}")

            if display:
                cv2.imshow("CCTV Vehicle Counter - Apple Neural Engine", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if max_frames and frame_count >= max_frames:
                print(f"\nReached maximum frames ({max_frames}).")
                break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Video saved to: {save_video}")
        if display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 CCTV Vehicle Counter on Apple Neural Engine (ANE)")
    parser.add_argument("--url", type=str, default=STREAM_URL, help="HLS stream URL or video source")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to CoreML .mlpackage model")
    parser.add_argument("--snapshot", action="store_true", help="Capture single frame and save annotated image")
    parser.add_argument("--output", type=str, default="cctv_snapshot_detected.jpg", help="Output file for snapshot")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window (print to terminal)")
    parser.add_argument("--frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--save-video", type=str, default=None, help="Save recorded detection video (.mp4)")
    args = parser.parse_args()

    print("Loading CoreML model into Apple Neural Engine...")
    model = YOLO(args.model)

    if args.snapshot:
        process_snapshot(model, args.url, output_image_path=args.output, conf=args.conf)
    else:
        process_live(model, args.url, conf=args.conf, display=not args.headless,
                     max_frames=args.frames, save_video=args.save_video)


if __name__ == "__main__":
    main()
