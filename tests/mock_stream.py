"""
tests/mock_stream.py
Synthetic HLS Stream & Trajectory Generator for BSW Kota Bogor MTMC Test Harness.
Enables offline, deterministic test execution without relying on active internet/CCTV feeds.
"""

import io
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


class SyntheticFrameGenerator:
    """
    Generates deterministic synthetic video frames and vehicle crops
    simulating BSW Kota Bogor CCTV camera feeds.
    """

    def __init__(self, width: int = 640, height: int = 480, seed: int = 42):
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_background(self, junction_type: str = "intersection") -> Image.Image:
        """
        Creates a synthetic road / junction background.
        junction_type: 'intersection', 'corridor', 'roundabout'
        """
        img = Image.new("RGB", (self.width, self.height), color=(40, 44, 52))
        draw = ImageDraw.Draw(img)

        # Draw asphalt roads
        asphalt_color = (65, 70, 75)
        marking_color = (230, 230, 230)
        yellow_line = (235, 190, 40)

        cx, cy = self.width // 2, self.height // 2
        road_width = int(min(self.width, self.height) * 0.45)

        if junction_type in ("intersection", "roundabout"):
            # Horizontal road
            draw.rectangle(
                [0, cy - road_width // 2, self.width, cy + road_width // 2],
                fill=asphalt_color,
            )
            # Vertical road
            draw.rectangle(
                [cx - road_width // 2, 0, cx + road_width // 2, self.height],
                fill=asphalt_color,
            )

            # Center markings / roundabout circle
            if junction_type == "roundabout":
                r = road_width // 3
                draw.ellipse(
                    [cx - r, cy - r, cx + r, cy + r],
                    fill=(30, 70, 32),
                    outline=yellow_line,
                    width=2,
                )
            else:
                # Intersection center crosswalk markings
                for offset in range(-road_width // 2, road_width // 2, 20):
                    draw.rectangle(
                        [
                            cx + offset,
                            cy - road_width // 2 - 15,
                            cx + offset + 10,
                            cy - road_width // 2 - 5,
                        ],
                        fill=marking_color,
                    )
                    draw.rectangle(
                        [
                            cx + offset,
                            cy + road_width // 2 + 5,
                            cx + offset + 10,
                            cy + road_width // 2 + 15,
                        ],
                        fill=marking_color,
                    )
        else:
            # Straight corridor road
            draw.rectangle(
                [0, cy - road_width // 2, self.width, cy + road_width // 2],
                fill=asphalt_color,
            )
            # Center dividing line (dashed yellow)
            for x in range(0, self.width, 30):
                draw.line([(x, cy), (min(x + 15, self.width), cy)], fill=yellow_line, width=2)

        return img

    def draw_vehicle(
        self,
        img: Image.Image,
        bbox: Tuple[int, int, int, int],
        vehicle_class: str = "car",
        color: Tuple[int, int, int] = (200, 30, 30),
        heading_deg: float = 0.0,
        license_text: Optional[str] = None,
    ) -> Image.Image:
        """
        Renders a vehicle onto the image at the given bounding box (x1, y1, x2, y2).
        """
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = bbox
        w, h = max(1, x2 - x1), max(1, y2 - y1)

        # Vehicle chassis body
        chassis_color = color
        roof_color = (
            min(255, int(color[0] * 0.8)),
            min(255, int(color[1] * 0.8)),
            min(255, int(color[2] * 0.8)),
        )

        draw.rounded_rectangle([x1, y1, x2, y2], radius=max(2, min(w, h) // 8), fill=chassis_color)

        # Windshield / cabin (only if large enough)
        if w >= 8 and h >= 8:
            pad_x = max(1, int(w * 0.15))
            pad_y = max(1, int(h * 0.20))
            cabin_box = [x1 + pad_x, y1 + pad_y, max(x1 + pad_x, x2 - pad_x), max(y1 + pad_y, y2 - pad_y)]
            draw.rectangle(cabin_box, fill=roof_color)

        # Headlights / taillights based on heading (only if large enough)
        if w >= 10 and h >= 10:
            hl_color = (255, 255, 200)
            tl_color = (220, 20, 20)

            # Heading determines front/rear
            if 45 <= heading_deg < 135:  # Heading East
                draw.rectangle([x2 - 4, y1 + 2, x2, y1 + 8], fill=hl_color)
                draw.rectangle([x2 - 4, y2 - 8, x2, y2 - 2], fill=hl_color)
                draw.rectangle([x1, y1 + 2, x1 + 4, y1 + 8], fill=tl_color)
                draw.rectangle([x1, y2 - 8, x1 + 4, y2 - 2], fill=tl_color)
            elif 225 <= heading_deg < 315:  # Heading West
                draw.rectangle([x1, y1 + 2, x1 + 4, y1 + 8], fill=hl_color)
                draw.rectangle([x1, y2 - 8, x1 + 4, y2 - 2], fill=hl_color)
                draw.rectangle([x2 - 4, y1 + 2, x2, y1 + 8], fill=tl_color)
                draw.rectangle([x2 - 4, y2 - 8, x2, y2 - 2], fill=tl_color)
            elif 135 <= heading_deg < 225:  # Heading South
                draw.rectangle([x1 + 2, y2 - 4, x1 + 8, y2], fill=hl_color)
                draw.rectangle([x2 - 8, y2 - 4, x2 - 2, y2], fill=hl_color)
                draw.rectangle([x1 + 2, y1, x1 + 8, y1 + 4], fill=tl_color)
                draw.rectangle([x2 - 8, y1, x2 - 2, y1 + 4], fill=tl_color)
            else:  # Heading North
                draw.rectangle([x1 + 2, y1, x1 + 8, y1 + 4], fill=hl_color)
                draw.rectangle([x2 - 8, y1, x2 - 2, y1 + 4], fill=hl_color)
                draw.rectangle([x1 + 2, y2 - 4, x1 + 8, y2], fill=tl_color)
                draw.rectangle([x2 - 8, y2 - 4, x2 - 2, y2], fill=tl_color)

        return img

    def generate_frame(
        self,
        camera_id: int,
        vehicles: List[Dict[str, Any]],
        junction_type: str = "intersection",
        timestamp_str: Optional[str] = None,
    ) -> Tuple[np.ndarray, bytes]:
        """
        Renders a full camera frame with multiple vehicles and timestamp HUD.
        Returns (rgb_numpy_array, jpeg_bytes).
        """
        img = self.generate_background(junction_type)
        draw = ImageDraw.Draw(img)

        # Draw HUD text
        ts = timestamp_str or time.strftime("%Y-%m-%d %H:%M:%S")
        hud_text = f"BSW CCTV #{camera_id} - {ts}"
        draw.text((10, 10), hud_text, fill=(255, 255, 255))

        for v in vehicles:
            bbox = v.get("bbox", (100, 100, 180, 140))
            v_class = v.get("class", "car")
            color = v.get("color", (200, 30, 30))
            heading = v.get("heading", 90.0)
            self.draw_vehicle(img, bbox, v_class, color, heading)

        # Convert to numpy and jpeg bytes
        arr = np.array(img, dtype=np.uint8)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        jpeg_bytes = buf.getvalue()

        return arr, jpeg_bytes

    def generate_vehicle_crop(
        self,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (200, 30, 30),
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[np.ndarray, bytes]:
        """
        Generates an isolated vehicle crop resized to target_size for Re-ID extraction.
        """
        x1, y1, x2, y2 = bbox
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        crop_img = Image.new("RGB", (w, h), color=(50, 50, 50))
        self.draw_vehicle(crop_img, (0, 0, w, h), color=color)

        resized = crop_img.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.uint8)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=90)
        return arr, buf.getvalue()

    def generate_corrupted_frame(self, corruption_type: str = "single_pixel") -> np.ndarray:
        """
        Generates edge-case / corrupted frames for boundary testing:
        'single_pixel', 'extreme_aspect', 'zero_pixels', 'noise'
        """
        if corruption_type == "single_pixel":
            return np.zeros((1, 1, 3), dtype=np.uint8)
        elif corruption_type == "extreme_aspect":
            return np.zeros((1, 1000, 3), dtype=np.uint8)
        elif corruption_type == "zero_pixels":
            return np.zeros((0, 0, 3), dtype=np.uint8)
        elif corruption_type == "noise":
            return self.np_rng.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
        else:
            return np.zeros((10, 10, 3), dtype=np.uint8)


class SyntheticTrajectorySimulator:
    """
    Simulates vehicle journeys across Kota Bogor camera junctions,
    producing realistic spatiotemporal coordinates, bounding box transitions,
    and ground-truth sighting events.
    """

    def __init__(self, topology_data: Optional[Dict[str, Any]] = None, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.topology = topology_data or self._load_default_topology()
        self.nodes = {n["id"]: n for n in self.topology.get("nodes", [])}
        self.edges = self.topology.get("edges", [])

    def _load_default_topology(self) -> Dict[str, Any]:
        """Loads data/bogor_camera_topology.json if exists, else minimal reference network."""
        topology_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "bogor_camera_topology.json"
        )
        if os.path.exists(topology_path):
            with open(topology_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Minimal fallback
        return {
            "metadata": {"total_cameras": 5},
            "nodes": [
                {"id": 79, "name": "CCTV - TUGU KUJANG", "lat": -6.6014, "lng": 106.8051, "corridor": "SSA_RING"},
                {"id": 83, "name": "CCTV - PASAR BOGOR", "lat": -6.6009, "lng": 106.7995, "corridor": "SSA_RING"},
                {"id": 81, "name": "CCTV - BTM", "lat": -6.6036, "lng": 106.7958, "corridor": "SSA_RING"},
                {"id": 88, "name": "CCTV - BALAI KOTA", "lat": -6.5960, "lng": 106.7946, "corridor": "SSA_RING"},
                {"id": 92, "name": "CCTV - JALAK HARUPAT", "lat": -6.5900, "lng": 106.8010, "corridor": "SSA_RING"},
            ],
            "edges": [
                {"source_id": 79, "target_id": 83, "distance_m": 620.0, "typical_speed_kmh": 28.0, "t_min_s": 55.0, "t_max_s": 140.0},
                {"source_id": 83, "target_id": 81, "distance_m": 580.0, "typical_speed_kmh": 25.0, "t_min_s": 50.0, "t_max_s": 130.0},
                {"source_id": 81, "target_id": 88, "distance_m": 850.0, "typical_speed_kmh": 30.0, "t_min_s": 75.0, "t_max_s": 170.0},
                {"source_id": 88, "target_id": 92, "distance_m": 900.0, "typical_speed_kmh": 32.0, "t_min_s": 80.0, "t_max_s": 180.0},
                {"source_id": 92, "target_id": 79, "distance_m": 950.0, "typical_speed_kmh": 35.0, "t_min_s": 85.0, "t_max_s": 190.0},
            ],
        }

    def simulate_vehicle_corridor_journey(
        self,
        global_vehicle_id: int,
        route_camera_ids: List[int],
        start_time_s: float = 0.0,
        vehicle_class: str = "car",
        primary_color: Tuple[int, int, int] = (210, 40, 40),
        speed_factor: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Simulates a continuous vehicle journey traversing a sequence of cameras.
        Returns a list of timestamped camera sighting events.
        """
        sightings: List[Dict[str, Any]] = []
        current_time = start_time_s

        # Generate deterministic synthetic embedding (256-dim unit vector)
        emb_rng = np.random.RandomState(global_vehicle_id * 1000 + 7)
        base_emb = emb_rng.randn(256).astype(np.float32)
        base_emb /= np.linalg.norm(base_emb)

        # Generate deterministic 128-bin color histogram matching primary_color
        color_hist = np.zeros(128, dtype=np.float32)
        r, g, b = primary_color
        # Map primary RGB dominant color to hue index
        max_c = max(r, g, b)
        if max_c == r:
            hue_bin = 10  # Red
        elif max_c == g:
            hue_bin = 45  # Green
        elif max_c == b:
            hue_bin = 80  # Blue
        else:
            hue_bin = 20
        color_hist[hue_bin : hue_bin + 10] = 1.0 / 10.0

        for i, cam_id in enumerate(route_camera_ids):
            cam = self.nodes.get(cam_id, {"id": cam_id, "name": f"CCTV-{cam_id}"})

            # Calculate travel time from previous camera
            if i > 0:
                prev_cam_id = route_camera_ids[i - 1]
                # Look for connecting edge
                matched_edges = [
                    e
                    for e in self.edges
                    if e.get("source_id") == prev_cam_id and e.get("target_id") == cam_id
                ]
                if matched_edges:
                    edge = matched_edges[0]
                    t_min = edge.get("t_min_s", 30.0)
                    t_max = edge.get("t_max_s", 90.0)
                    typical_t = (t_min + t_max) / 2.0
                    travel_time = (typical_t / max(0.2, speed_factor)) + self.rng.uniform(-3.0, 3.0)
                else:
                    travel_time = 45.0 / max(0.2, speed_factor)
                current_time += max(5.0, travel_time)

            # Generate intra-camera track: 5 frames over 3 seconds
            dwell_duration = 3.0
            num_frames = 5
            for f_idx in range(num_frames):
                frame_time = current_time + (f_idx * (dwell_duration / num_frames))
                progress = f_idx / max(1, num_frames - 1)

                # Simulated bounding box progression across frame (e.g. entering left x=50, exiting right x=550)
                x1 = int(50 + progress * 500)
                y1 = int(200 + progress * 20)
                x2 = x1 + 80
                y2 = y1 + 50
                bbox = (x1, y1, x2, y2)

                # Heading
                heading = 90.0 if i < len(route_camera_ids) - 1 else 180.0

                # Slight visual noise per sighting (<0.02)
                noise = emb_rng.randn(256).astype(np.float32) * 0.015
                emb = base_emb + noise
                emb /= np.linalg.norm(emb)

                sighting = {
                    "global_vehicle_id": global_vehicle_id,
                    "local_track_id": 100 + i,
                    "camera_id": cam_id,
                    "camera_name": cam.get("name", f"CCTV-{cam_id}"),
                    "timestamp": frame_time,
                    "timestamp_iso": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(1700000000 + frame_time)
                    ),
                    "vehicle_class": vehicle_class,
                    "confidence": 0.88 + self.rng.uniform(-0.05, 0.05),
                    "bbox": bbox,
                    "heading_deg": heading,
                    "speed_px_s": 120.0 * speed_factor,
                    "embedding": emb.tobytes(),
                    "color_hist": color_hist.tobytes(),
                    "embedding_vector": emb,
                    "color_hist_vector": color_hist,
                    "is_exit_frame": (f_idx == num_frames - 1),
                }
                sightings.append(sighting)

        return sightings


class MockHLSStreamServer:
    """
    Simulates municipal HLS stream endpoint responding with deterministic
    .m3u8 playlists and media chunks.
    """

    def __init__(self, camera_id: int, segment_duration: float = 3.32):
        self.camera_id = camera_id
        self.segment_duration = segment_duration
        self.sequence_number = 1000

    def get_playlist_m3u8(self, current_time_s: Optional[float] = None) -> str:
        """Generates standard HLS m3u8 playlist string matching Kota Bogor restreamer format."""
        t = current_time_s or time.time()
        prog_time = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(t))
        self.sequence_number += 1

        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{int(math.ceil(self.segment_duration))}",
            f"#EXT-X-MEDIA-SEQUENCE:{self.sequence_number}",
            f"#EXT-X-PROGRAM-DATE-TIME:{prog_time}",
            f"#EXTINF:{self.segment_duration:.6f},",
            f"segment_{self.sequence_number - 2}.ts",
            f"#EXTINF:{self.segment_duration:.6f},",
            f"segment_{self.sequence_number - 1}.ts",
            f"#EXTINF:{self.segment_duration:.6f},",
            f"segment_{self.sequence_number}.ts",
        ]
        return "\n".join(lines) + "\n"

    def get_simulated_packet_stream(self, num_packets: int = 10) -> List[Dict[str, Any]]:
        """Simulates raw stream packet arrivals with timestamp and byte size."""
        packets = []
        base_t = time.time()
        for i in range(num_packets):
            packets.append(
                {
                    "camera_id": self.camera_id,
                    "packet_idx": i,
                    "pts_time": base_t + (i * 0.04),  # 25 FPS = 40ms interval
                    "size_bytes": 45000 + random.randint(-5000, 5000),
                    "is_keyframe": (i % 25 == 0),
                }
            )
        return packets


class MockCameraNetwork:
    """
    In-memory simulation of Kota Bogor 94-camera network with active vehicles.
    """

    def __init__(self, topology_path: Optional[str] = None):
        self.frame_gen = SyntheticFrameGenerator()
        self.sim = SyntheticTrajectorySimulator()
        self.active_vehicles: Dict[int, Dict[str, Any]] = {}
        self.camera_states: Dict[int, Dict[str, Any]] = {}

        for n in self.sim.nodes.values():
            self.camera_states[n["id"]] = {
                "id": n["id"],
                "name": n.get("name"),
                "status": "online",
                "stream_url": n.get("stream_url", f"https://mock.bogor.go.id/{n['id']}.m3u8"),
                "vehicle_count": 0,
                "tier": 3,
            }

    def set_camera_tier(self, camera_id: int, tier: int):
        if camera_id in self.camera_states:
            self.camera_states[camera_id]["tier"] = tier

    def get_camera_frame(self, camera_id: int) -> Tuple[np.ndarray, bytes]:
        vehicles = [v for v in self.active_vehicles.values() if v.get("current_camera_id") == camera_id]
        return self.frame_gen.generate_frame(camera_id, vehicles)
