"""
pipeline/homography.py
Calibration terrain : 32 keypoints YOLOv8x-pose → homographie image→terrain.
Dimensions officielles FIFA : 105m x 68m.
"""
from __future__ import annotations
import cv2
import numpy as np
import torch
from ultralytics import YOLO

PITCH_W = 105.0
PITCH_H = 68.0

# 32 points du terrain en mètres (schema roboflow/sports)
PITCH_KP_REAL = np.array([
    [ 0.0,  0.0], [ 0.0, 34.0], [ 0.0, 68.0],
    [ 5.5, 13.84],[ 5.5, 54.16],[ 5.5, 24.84],[ 5.5, 43.16],
    [16.5, 13.84],[16.5, 54.16],[11.0, 34.0],
    [52.5,  0.0], [52.5, 68.0], [52.5, 34.0],
    [105.0,  0.0],[105.0,34.0], [105.0,68.0],
    [99.5, 13.84],[99.5, 54.16],[99.5, 24.84],[99.5, 43.16],
    [88.5, 13.84],[88.5, 54.16],[94.0, 34.0],
    [ 0.0, 24.84],[ 0.0, 43.16],
    [105.0,24.84],[105.0,43.16],
    [16.5,  0.0], [16.5, 68.0],
    [88.5,  0.0], [88.5, 68.0],
    [52.5, 34.0],
], dtype=np.float32)


class PitchHomography:
    """
    Calcule et met à jour l'homographie image→terrain.
    Recalcul toutes les `update_every` frames pour performance.
    """

    def __init__(self, model_path: str = "models/pitch_keypoints.pt",
                 device: str | None = None, use_fp16: bool = True,
                 update_every: int = 30, min_keypoints: int = 6):
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.update_every = update_every
        self.min_kp   = min_keypoints
        self.model    = YOLO(model_path)
        self.model.to(self.device)
        if self.use_fp16:
            self.model.model.half()
        self.H: np.ndarray | None = None
        self.frame_count = 0
        print("[Homography] Modèle keypoints chargé.")

    def _detect_kp(self, frame: np.ndarray):
        preds = self.model.predict(source=frame, device=self.device,
                                   half=self.use_fp16, verbose=False)
        if not preds or preds[0].keypoints is None:
            return None
        kps = preds[0].keypoints
        if kps.xy is None or len(kps.xy) == 0:
            return None
        kp_xy   = kps.xy[0].cpu().numpy()
        kp_conf = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(len(kp_xy))
        img_pts, real_pts = [], []
        n = min(len(kp_xy), len(PITCH_KP_REAL))
        for i in range(n):
            conf = float(kp_conf[i]) if i < len(kp_conf) else 1.0
            x, y = float(kp_xy[i][0]), float(kp_xy[i][1])
            if conf < 0.3 or (x == 0 and y == 0):
                continue
            img_pts.append([x, y]); real_pts.append(PITCH_KP_REAL[i])
        if len(img_pts) < self.min_kp:
            return None
        return np.array(img_pts, dtype=np.float32), np.array(real_pts, dtype=np.float32)

    def update(self, frame: np.ndarray) -> bool:
        """Recalcule l'homographie si nécessaire. Retourne True si disponible."""
        self.frame_count += 1
        if self.frame_count % self.update_every == 1 or self.H is None:
            result = self._detect_kp(frame)
            if result:
                H, _ = cv2.findHomography(result[0], result[1], cv2.RANSAC, 5.0)
                if H is not None:
                    self.H = H
        return self.H is not None

    def image_to_pitch(self, pt_px: tuple) -> tuple[float, float] | None:
        """Projette un point image (px) en coordonnées terrain (mètres)."""
        if self.H is None:
            return None
        res = cv2.perspectiveTransform(np.array([[pt_px]], dtype=np.float32), self.H)
        x_m = float(np.clip(res[0][0][0], 0, PITCH_W))
        y_m = float(np.clip(res[0][0][1], 0, PITCH_H))
        return (x_m, y_m)

    def pitch_to_minimap(self, xy_m: tuple, map_w: int, map_h: int, margin: int = 5) -> tuple:
        """Coordonnées terrain (m) → pixels sur la minimap."""
        px = int(margin + (xy_m[0] / PITCH_W) * (map_w - 2*margin))
        py = int(margin + (xy_m[1] / PITCH_H) * (map_h - 2*margin))
        return (px, py)

    def project_all(self, tracks: dict) -> dict[int, tuple]:
        """Projette tous les acteurs. Returns {track_id: (x_m, y_m)}, -1=balle."""
        result: dict[int, tuple] = {}
        if self.H is None:
            return result
        for cat in ("players", "goalkeepers", "referees"):
            for obj in tracks.get(cat, []):
                foot = obj.get("foot") or obj.get("center")
                pos  = self.image_to_pitch(foot)
                if pos:
                    result[obj["track_id"]] = pos
        if tracks.get("ball"):
            pos = self.image_to_pitch(tracks["ball"]["center"])
            if pos:
                result[-1] = pos
        return result
