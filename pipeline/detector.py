"""
pipeline/detector.py
Module de détection YOLO11 avec inférence FP16 sur GPU NVIDIA.

Classes détectées par player_detection.pt :
  0: player  |  1: goalkeeper  |  2: referee  |  3: ball

Modèle ball_detection.pt :
  0: ball (détection dédiée haute précision)
"""

from __future__ import annotations
import torch
import numpy as np
from ultralytics import YOLO

# IDs des classes dans player_detection.pt
CLASS_PLAYER     = 0
CLASS_GOALKEEPER = 1
CLASS_REFEREE    = 2
CLASS_BALL_PLAYER = 3

# Seuils de confiance
CONF_PLAYERS = 0.4
CONF_BALL    = 0.3


class FootballDetector:
    """
    Détecteur combiné joueurs + balle avec YOLO11 en FP16 (half precision).
    Deux modèles spécialisés pour maximiser les performances.
    """

    def __init__(
        self,
        player_model_path: str = "models/player_detection.pt",
        ball_model_path:   str = "models/ball_detection.pt",
        device: str | None = None,
        use_fp16: bool = True,
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.use_fp16 = use_fp16 and self.device == "cuda"

        print(f"[Detector] Device : {self.device.upper()}")
        if self.device == "cuda":
            print(f"[Detector] GPU    : {torch.cuda.get_device_name(0)}")
            print(f"[Detector] FP16   : {self.use_fp16}")
        elif self.device == "mps":
            print("[Detector] GPU    : Apple Silicon (MPS)")

        self.player_model = YOLO(player_model_path)
        self.ball_model   = YOLO(ball_model_path)
        self.player_model.to(self.device)
        self.ball_model.to(self.device)

        if self.use_fp16:
            self.player_model.model.half()
            self.ball_model.model.half()

        print("[Detector] Modèles chargés.")

    def _box_to_det(self, box) -> dict:
        """Convertit une box YOLO en dict de détection."""
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cx   = int((xyxy[0] + xyxy[2]) / 2)
        cy   = int((xyxy[1] + xyxy[3]) / 2)
        return {
            "bbox":     xyxy.tolist(),
            "center":   (cx, cy),
            "foot":     (cx, int(xyxy[3])),  # Point bas de la bbox (pied)
            "conf":     conf,
            "class_id": cls,
        }

    def detect_frame(
        self,
        frame: np.ndarray,
        conf_players: float = CONF_PLAYERS,
        conf_ball: float = CONF_BALL,
    ) -> dict:
        """
        Détecte joueurs, gardiens, arbitres et ballon dans une frame.

        Returns:
            dict avec clés : players, goalkeepers, referees, ball
        """
        results = {
            "players": [], "goalkeepers": [],
            "referees": [], "ball": None,
        }

        # Détection principale : joueurs / gardiens / arbitres / balle
        preds = self.player_model.predict(
            source=frame, conf=conf_players,
            device=self.device, half=self.use_fp16, verbose=False,
        )[0]

        for box in preds.boxes:
            det = self._box_to_det(box)
            cls = det["class_id"]
            if cls == CLASS_PLAYER:
                results["players"].append(det)
            elif cls == CLASS_GOALKEEPER:
                results["goalkeepers"].append(det)
            elif cls == CLASS_REFEREE:
                results["referees"].append(det)
            elif cls == CLASS_BALL_PLAYER:
                if results["ball"] is None or det["conf"] > results["ball"]["conf"]:
                    results["ball"] = det

        # Modèle balle dédié (fallback ou amélioration)
        if results["ball"] is None or results["ball"]["conf"] < 0.5:
            ball_preds = self.ball_model.predict(
                source=frame, conf=conf_ball,
                device=self.device, half=self.use_fp16, verbose=False,
            )[0]
            for box in ball_preds.boxes:
                det = self._box_to_det(box)
                if results["ball"] is None or det["conf"] > results["ball"]["conf"]:
                    results["ball"] = det

        return results

    def detect_batch(
        self,
        frames: list[np.ndarray],
        conf_players: float = CONF_PLAYERS,
    ) -> list[dict]:
        """
        Détection en batch pour maximiser l'utilisation du GPU.
        """
        all_results = []
        preds_iter = self.player_model.predict(
            source=frames, conf=conf_players,
            device=self.device, half=self.use_fp16,
            verbose=False, stream=True,
        )
        for preds in preds_iter:
            r = {"players": [], "goalkeepers": [], "referees": [], "ball": None}
            for box in preds.boxes:
                det = self._box_to_det(box)
                cls = det["class_id"]
                if cls == CLASS_PLAYER:
                    r["players"].append(det)
                elif cls == CLASS_GOALKEEPER:
                    r["goalkeepers"].append(det)
                elif cls == CLASS_REFEREE:
                    r["referees"].append(det)
                elif cls == CLASS_BALL_PLAYER:
                    if r["ball"] is None or det["conf"] > r["ball"]["conf"]:
                        r["ball"] = det
            all_results.append(r)
        return all_results
