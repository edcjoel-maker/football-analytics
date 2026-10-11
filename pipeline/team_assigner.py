"""
pipeline/team_assigner.py
Attribution des équipes par clustering couleur de maillot (K-Means HSV).
Sépare automatiquement Équipe A, Équipe B. Les arbitres ont team_id=2.
"""
from __future__ import annotations
import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """K-Means sur couleurs HSV des maillots pour séparer les deux équipes."""

    def __init__(self, n_clusters: int = 2):
        self.n_clusters  = n_clusters
        self.kmeans      = None
        self.team_colors: dict[int, dict] = {}
        self.player_teams: dict[int, dict] = {}
        self.is_fitted   = False

    def _extract_jersey_color(self, frame: np.ndarray, bbox: list) -> np.ndarray | None:
        """Extrait la couleur dominante du maillot (partie haute de la bbox)."""
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        if h < 20 or w < 10:
            return None
        # Région maillot : 20%→55% hauteur, 15%→85% largeur
        t = max(0, y1 + int(h*0.20)); b = min(frame.shape[0], y1 + int(h*0.55))
        l = max(0, x1 + int(w*0.15)); r = min(frame.shape[1], x2 - int(w*0.15))
        roi = frame[t:b, l:r]
        if roi.size == 0:
            return None
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        km = KMeans(n_clusters=2, n_init=3, random_state=42)
        km.fit(roi_hsv)
        u, c = np.unique(km.labels_, return_counts=True)
        return km.cluster_centers_[u[np.argmax(c)]]

    def fit(self, frame: np.ndarray, player_tracks: list[dict]) -> None:
        """Entraîne le K-Means sur un échantillon de joueurs."""
        colors = [c for p in player_tracks
                  if (c := self._extract_jersey_color(frame, p["bbox"])) is not None]
        if len(colors) < self.n_clusters * 2:
            return
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self.kmeans.fit(np.array(colors))
        for cid, center in enumerate(self.kmeans.cluster_centers_):
            bgr = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_HSV2BGR)[0][0]
            self.team_colors[cid] = {"hsv": center.tolist(), "bgr": bgr.tolist()}
        self.is_fitted = True

    def get_player_team(self, frame: np.ndarray, track: dict) -> int:
        """Retourne team_id (0 ou 1) avec vote majoritaire pour stabilité."""
        if not self.is_fitted:
            return -1
        tid   = track["track_id"]
        color = self._extract_jersey_color(frame, track["bbox"])
        if color is None:
            return self.player_teams[tid]["team_id"] if tid in self.player_teams else -1
        pred = int(self.kmeans.predict([color])[0])
        if tid not in self.player_teams:
            self.player_teams[tid] = {"votes": [0, 0], "team_id": pred}
        self.player_teams[tid]["votes"][pred] += 1
        v = self.player_teams[tid]["votes"]
        self.player_teams[tid]["team_id"] = 0 if v[0] >= v[1] else 1
        return self.player_teams[tid]["team_id"]

    def assign_teams_to_frame(self, frame: np.ndarray, tracks: dict) -> dict:
        """Assigne les équipes à tous les acteurs de la frame."""
        all_p = tracks.get("players", []) + tracks.get("goalkeepers", [])
        if not self.is_fitted and len(all_p) >= 4:
            self.fit(frame, all_p)
        for p in all_p:
            p["team_id"] = self.get_player_team(frame, p)
        for ref in tracks.get("referees", []):
            ref["team_id"] = 2
        return tracks

    def get_team_color_bgr(self, team_id: int) -> tuple:
        """Couleur BGR de l'équipe pour le rendu visuel."""
        if team_id == 2:
            return (0, 255, 255)
        if team_id not in self.team_colors:
            return (128, 128, 128)
        b = self.team_colors[team_id]["bgr"]
        return (int(b[0]), int(b[1]), int(b[2]))
