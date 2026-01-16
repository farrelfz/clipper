from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp

from clipgen.utils import write_json


@dataclass
class FaceDetection:
    time: float
    bbox: Tuple[int, int, int, int]
    score: float
    track_id: int


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.35) -> None:
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: Dict[int, Tuple[int, int, int, int]] = {}

    def assign(self, detections: List[Tuple[int, int, int, int]]) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        assignments: List[Tuple[int, Tuple[int, int, int, int]]] = []
        remaining_tracks = dict(self.tracks)
        for det in detections:
            best_iou = 0.0
            best_track = None
            for track_id, bbox in remaining_tracks.items():
                iou = _iou(det, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track_id
            if best_track is not None and best_iou >= self.iou_threshold:
                assignments.append((best_track, det))
                remaining_tracks.pop(best_track, None)
            else:
                track_id = self.next_id
                self.next_id += 1
                assignments.append((track_id, det))
        self.tracks = {track_id: bbox for track_id, bbox in assignments}
        return assignments


def detect_faces(video_path: Path, output_path: Path, analysis_fps: float) -> List[FaceDetection]:
    detections: List[FaceDetection] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open video for face detection.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps / analysis_fps)))
    tracker = IoUTracker()
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval != 0:
            frame_index += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        bboxes: List[Tuple[int, int, int, int]] = []
        if results.detections:
            height, width, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * width)
                y_min = int(bbox.ymin * height)
                box_width = int(bbox.width * width)
                box_height = int(bbox.height * height)
                bboxes.append((x_min, y_min, x_min + box_width, y_min + box_height))
        assignments = tracker.assign(bboxes)
        time_sec = frame_index / fps
        for track_id, bbox in assignments:
            detections.append(
                FaceDetection(time=time_sec, bbox=bbox, score=1.0, track_id=track_id)
            )
        frame_index += 1
    cap.release()
    write_json(
        output_path,
        {
            "detections": [
                {
                    "time": det.time,
                    "bbox": det.bbox,
                    "score": det.score,
                    "track_id": det.track_id,
                }
                for det in detections
            ]
        },
    )
    return detections


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter_area / float(area_a + area_b - inter_area)
