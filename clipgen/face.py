from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .logging import JsonlLogger


@dataclass
class FaceDetection:
    time_s: float
    bbox: Tuple[float, float, float, float]
    score: float


@dataclass
class Track:
    track_id: int
    detections: Dict[float, Tuple[float, float, float, float]] = field(default_factory=dict)
    scores: Dict[float, float] = field(default_factory=dict)


@dataclass
class TrackResult:
    tracks: List[Track]
    fps: float
    width: int
    height: int


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.active: Dict[int, Tuple[float, float, float, float]] = {}

    @staticmethod
    def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union else 0.0

    def update(self, detections: List[Tuple[float, float, float, float]]) -> Dict[int, Tuple[float, float, float, float]]:
        assignments: Dict[int, Tuple[float, float, float, float]] = {}
        used = set()
        for track_id, prev_box in list(self.active.items()):
            best_iou = 0.0
            best_idx = None
            for idx, box in enumerate(detections):
                if idx in used:
                    continue
                score = self.iou(prev_box, box)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx is not None and best_iou >= self.iou_threshold:
                assignments[track_id] = detections[best_idx]
                used.add(best_idx)
            else:
                self.active.pop(track_id, None)
        for idx, box in enumerate(detections):
            if idx in used:
                continue
            track_id = self.next_id
            self.next_id += 1
            assignments[track_id] = box
        self.active = assignments
        return assignments


def detect_faces(video_path: Path, analysis_fps: float, logger: JsonlLogger) -> TrackResult:
    logger.log("face", "start", fps=analysis_fps)
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe not installed.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or analysis_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(source_fps / analysis_fps)))

    tracker = SimpleTracker()
    tracks: Dict[int, Track] = {}

    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            time_s = frame_idx / source_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            detections: List[FaceDetection] = []
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = max(0.0, bbox.xmin)
                    y1 = max(0.0, bbox.ymin)
                    x2 = min(1.0, bbox.xmin + bbox.width)
                    y2 = min(1.0, bbox.ymin + bbox.height)
                    detections.append(
                        FaceDetection(
                            time_s=time_s,
                            bbox=(x1, y1, x2, y2),
                            score=float(det.score[0]),
                        )
                    )
            assigned = tracker.update([det.bbox for det in detections])
            for track_id, bbox in assigned.items():
                track = tracks.setdefault(track_id, Track(track_id=track_id))
                track.detections[time_s] = bbox
                match_score = 0.0
                for det in detections:
                    if det.bbox == bbox:
                        match_score = det.score
                        break
                track.scores[time_s] = match_score
            frame_idx += 1

    cap.release()
    result = TrackResult(tracks=list(tracks.values()), fps=analysis_fps, width=width, height=height)
    logger.log("face", "complete", tracks=len(result.tracks))
    return result
