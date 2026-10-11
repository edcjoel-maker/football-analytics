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

# 32 points réels du terrain officiel en mètres (spécification Roboflow SoccerPitchConfiguration)
# Dimensions FIFA : longueur 105.0m (X), largeur 68.0m (Y)
PITCH_KP_REAL = np.array([
    [0.0, 0.0],       # 1 : coin haut gauche
    [0.0, 13.84],     # 2 : ligne de but, intersection surface réparation haut
    [0.0, 24.84],     # 3 : ligne de but, poteau gauche (surface 5.5m haut)
    [0.0, 43.16],     # 4 : ligne de but, poteau droit (surface 5.5m bas)
    [0.0, 54.16],     # 5 : ligne de but, intersection surface réparation bas
    [0.0, 68.0],      # 6 : coin bas gauche
    [5.5, 24.84],     # 7 : surface 5.5m coin haut gauche
    [5.5, 43.16],     # 8 : surface 5.5m coin bas gauche
    [11.0, 34.0],     # 9 : point de penalty gauche
    [16.5, 13.84],    # 10: grande surface coin haut gauche
    [16.5, 24.84],    # 11: grande surface aligné poteau haut
    [16.5, 43.16],    # 12: grande surface aligné poteau bas
    [16.5, 54.16],    # 13: grande surface coin bas gauche
    [52.5, 0.0],      # 14: ligne médiane touche haut
    [52.5, 24.85],    # 15: rond central intersection médiane haut
    [52.5, 43.15],    # 16: rond central intersection médiane bas
    [52.5, 68.0],     # 17: ligne médiane touche bas
    [88.5, 13.84],    # 18: grande surface droite coin haut
    [88.5, 24.84],    # 19: grande surface droite aligné poteau haut
    [88.5, 43.16],    # 20: grande surface droite aligné poteau bas
    [88.5, 54.16],    # 21: grande surface droite coin bas
    [94.0, 34.0],     # 22: point de penalty droite
    [99.5, 24.84],    # 23: surface 5.5m droite coin haut
    [99.5, 43.16],    # 24: surface 5.5m droite coin bas
    [105.0, 0.0],     # 25: coin haut droit
    [105.0, 13.84],   # 26: ligne de but droite, surface réparation haut
    [105.0, 24.84],   # 27: ligne de but droite, poteau haut
    [105.0, 43.16],   # 28: ligne de but droite, poteau bas
    [105.0, 54.16],   # 29: ligne de but droite, surface réparation bas
    [105.0, 68.0],    # 30: coin bas droit
    [43.35, 34.0],    # 31: sommet gauche du rond central
    [61.65, 34.0],    # 32: sommet droit du rond central
], dtype=np.float32)



class PitchHomography:
    """
    Calcule et met à jour l'homographie image→terrain.
    Recalcul toutes les `update_every` frames pour performance.
    """

    def __init__(self, model_path: str = "models/pitch_keypoints.pt",
                 device: str | None = None, use_fp16: bool = False,
                 update_every: int = 30, min_keypoints: int = 6):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.use_fp16 = False  # Désactivé pour éviter struct c10::Half != float dans la tête keypoints
        self.update_every = update_every
        self.min_kp   = min_keypoints
        self.model    = YOLO(model_path)
        self.model.to(self.device)
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
