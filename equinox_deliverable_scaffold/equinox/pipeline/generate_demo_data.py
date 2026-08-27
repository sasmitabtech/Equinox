"""
Generates synthetic aerial drone video clips, road corridor masks, track detections CSV,
and ground-truth labels for executing the Equinox pipeline end-to-end.
"""
import json
import os
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


def generate_video(filepath: str, width: int = 640, height: int = 480, fps: int = 15, duration_s: int = 4, incident_type: str = "critical"):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    total_frames = fps * duration_s

    for frame_idx in range(total_frames):
        # Dark asphalt road background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40
        
        # Draw road lane markings (curved corridor)
        cv2.fillPoly(frame, [np.array([[80, 0], [120, height], [520, height], [560, 0]], np.int32)], (60, 60, 60))
        cv2.line(frame, (320, 0), (320, height), (200, 200, 200), 2, cv2.LINE_AA) # center dash line

        # Draw moving/stopped vehicles based on incident_type
        t = frame_idx / total_frames

        if incident_type == "critical":
            # Stalled truck stationary in middle of corridor
            cv2.rectangle(frame, (270, 200), (370, 280), (30, 40, 180), -1) # Stalled truck (reddish)
            cv2.putText(frame, "STALLED TRUCK", (270, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # Debris next to it
            cv2.circle(frame, (390, 260), 18, (0, 140, 255), -1) # Debris (orange)
            # Queuing cars behind
            cv2.rectangle(frame, (280, 340), (340, 390), (120, 120, 120), -1)
            cv2.rectangle(frame, (280, 410), (340, 460), (150, 100, 80), -1)

        elif incident_type == "severe":
            # Illegally parked van on narrow service road
            cv2.rectangle(frame, (130, 150), (210, 270), (40, 160, 200), -1)
            cv2.putText(frame, "ILLEGAL PARKED VAN", (130, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            # Moving car trying to squeeze by
            y_pos = int(50 + t * 300)
            cv2.rectangle(frame, (230, y_pos), (280, y_pos + 60), (100, 180, 100), -1)

        elif incident_type == "moderate":
            # Slow moving maintenance vehicle / partial lane obstruction
            y_pos = int(100 + t * 100)
            cv2.rectangle(frame, (300, y_pos), (360, y_pos + 70), (50, 180, 240), -1)
            cv2.putText(frame, "SLOW VEHICLE", (300, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        else: # normal / clear
            # Clean flowing traffic
            y1 = int((t * 400) % 400)
            y2 = int(((t + 0.5) * 400) % 400)
            cv2.rectangle(frame, (200, y1), (250, y1 + 50), (180, 180, 180), -1)
            cv2.rectangle(frame, (380, 400 - y2), (430, 450 - y2), (160, 200, 160), -1)

        # Drone HUD Overlay
        cv2.putText(frame, f"DRONE FEED · {os.path.basename(filepath).upper()} · FRAME {frame_idx:03d}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
        out.write(frame)

    out.release()
    print(f"Generated video clip: {filepath}")


def generate_mask(filepath: str, width: int = 640, height: int = 480):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array([[80, 0], [120, height], [520, height], [560, 0]], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    cv2.imwrite(filepath, mask)
    print(f"Generated mask: {filepath}")


def build_all_demo_data(base_dir: Path):
    raw_dir = base_dir / "data" / "raw"
    masks_dir = base_dir / "data" / "masks"
    det_dir = base_dir / "data" / "detections"
    labels_dir = base_dir / "data" / "labels"

    for d in [raw_dir, masks_dir, det_dir, labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    clips_info = [
        {
            "id": "clip_01_hosur_j4",
            "location": "Hosur_Rd_J4",
            "lat": 12.845,
            "lon": 77.663,
            "timestamp": "2026-08-26T10:41:52",
            "source": "drone_alpha",
            "type": "critical",
            "severity": "Critical",
            "acc_score": 28
        },
        {
            "id": "clip_02_service_rd",
            "location": "Service_Rd_Bl9",
            "lat": 12.839,
            "lon": 77.678,
            "timestamp": "2026-08-26T10:37:10",
            "source": "drone_beta",
            "type": "severe",
            "severity": "Severe",
            "acc_score": 45
        },
        {
            "id": "clip_03_neeladri_rd",
            "location": "Neeladri_Rd_R2",
            "lat": 12.851,
            "lon": 77.671,
            "timestamp": "2026-08-26T10:22:45",
            "source": "drone_alpha",
            "type": "moderate",
            "severity": "Moderate",
            "acc_score": 68
        },
        {
            "id": "clip_04_wipro_jct",
            "location": "Wipro_Junction",
            "lat": 12.842,
            "lon": 77.665,
            "timestamp": "2026-08-26T09:58:03",
            "source": "drone_gamma",
            "type": "normal",
            "severity": "Normal",
            "acc_score": 95
        }
    ]

    detections_rows = []
    labels_rows = []

    for item in clips_info:
        clip_id = item["id"]
        loc = item["location"]
        video_path = raw_dir / f"{clip_id}.mp4"
        json_path = raw_dir / f"{clip_id}.json"
        mask_path = masks_dir / f"{loc}.png"

        # 1. Generate MP4 video
        generate_video(str(video_path), incident_type=item["type"])

        # 2. Generate sidecar JSON metadata
        with open(json_path, "w") as f:
            json.dump({
                "location": loc,
                "lat": item["lat"],
                "lon": item["lon"],
                "timestamp": item["timestamp"],
                "source": item["source"]
            }, f, indent=2)

        # 3. Generate road mask PNG
        generate_mask(str(mask_path))

        # 4. Generate frame detections for 12 sampled frames (0 to 11)
        for frame_idx in range(12):
            if item["type"] == "critical":
                detections_rows.append({
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    "track_id": 101,
                    "cls": "stalled vehicle",
                    "bbox": "[270, 200, 370, 280]",
                    "conf": 0.94
                })
                detections_rows.append({
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    "track_id": 102,
                    "cls": "debris",
                    "bbox": "[372, 242, 408, 278]",
                    "conf": 0.88
                })
            elif item["type"] == "severe":
                detections_rows.append({
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    "track_id": 201,
                    "cls": "illegal parking",
                    "bbox": "[130, 150, 210, 270]",
                    "conf": 0.91
                })
            elif item["type"] == "moderate":
                detections_rows.append({
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    "track_id": 301,
                    "cls": "congestion",
                    "bbox": "[300, 120, 360, 190]",
                    "conf": 0.82
                })
            else:
                # Normal frame with passing vehicle
                detections_rows.append({
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    "track_id": 401,
                    "cls": "vehicle",
                    "bbox": "[200, 50, 250, 100]",
                    "conf": 0.96
                })

        # 5. Add label row
        labels_rows.append({
            "clip_id": clip_id,
            "start_frame": 0,
            "end_frame": 11,
            "severity": item["severity"],
            "accessibility_score": item["acc_score"]
        })

    # Save detections CSV
    det_df = pd.DataFrame(detections_rows)
    det_df.to_csv(det_dir / "detections.csv", index=False)
    print(f"Saved detections CSV: {det_dir / 'detections.csv'}")

    # Save labels CSV
    lab_df = pd.DataFrame(labels_rows)
    lab_df.to_csv(labels_dir / "labels.csv", index=False)
    print(f"Saved labels CSV: {labels_dir / 'labels.csv'}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    build_all_demo_data(base)
