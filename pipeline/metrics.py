"""
pipeline/metrics.py
Métriques Veo : possession, heatmaps, passes, distances, vitesses de pointe.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

PITCH_W = 105.0
PITCH_H = 68.0


@dataclass
class MatchMetrics:
    possession_frames: dict = field(default_factory=lambda: {0:0, 1:0, -1:0})
    heatmap_a: np.ndarray = field(default_factory=lambda: np.zeros((14,21), dtype=np.float32))
    heatmap_b: np.ndarray = field(default_factory=lambda: np.zeros((14,21), dtype=np.float32))
    pass_matrix: dict = field(default_factory=lambda: defaultdict(int))
    distances_m:   dict = field(default_factory=lambda: defaultdict(float))
    max_speeds_mps: dict = field(default_factory=lambda: defaultdict(float))
    player_teams:  dict = field(default_factory=dict)
    _last_possessor: int = -1
    total_frames: int = 0


class MetricsCalculator:
    POSSESSION_RADIUS = 3.0   # mètres
    HEATMAP_COLS = 21         # 105m / 5m
    HEATMAP_ROWS = 14         # 68m  / 5m

    def __init__(self):
        self.metrics = MatchMetrics()

    def update(self, tracks: dict, pitch_pos: dict, tracker) -> None:
        """Appeler à chaque frame traitée."""
        self.metrics.total_frames += 1

        # Enregistrer team_id
        for cat in ("players", "goalkeepers"):
            for p in tracks.get(cat, []):
                if p.get("team_id", -1) != -1:
                    self.metrics.player_teams[p["track_id"]] = p["team_id"]

        # Heatmaps
        for cat in ("players", "goalkeepers"):
            for p in tracks.get(cat, []):
                tid  = p["track_id"]
                team = self.metrics.player_teams.get(tid, -1)
                pos  = pitch_pos.get(tid)
                if pos is None or team not in (0, 1):
                    continue
                col = int(np.clip(pos[0]/PITCH_W*self.HEATMAP_COLS, 0, self.HEATMAP_COLS-1))
                row = int(np.clip(pos[1]/PITCH_H*self.HEATMAP_ROWS, 0, self.HEATMAP_ROWS-1))
                (self.metrics.heatmap_a if team == 0 else self.metrics.heatmap_b)[row, col] += 1

        # Possession & détection de passe
        ball_pos = pitch_pos.get(-1)
        if ball_pos is not None:
            poss = self._closest_player(ball_pos, tracks, pitch_pos)
            team_poss = self.metrics.player_teams.get(poss, -1)
            self.metrics.possession_frames[team_poss if team_poss in (0,1) else -1] += 1
            prev = self.metrics._last_possessor
            if prev != -1 and poss != -1 and poss != prev:
                self.metrics.pass_matrix[(prev, poss)] += 1
            self.metrics._last_possessor = poss

        # Distances & vitesses
        for tid in pitch_pos:
            if tid == -1:
                continue
            self.metrics.distances_m[tid]     = tracker.get_distance(tid)
            spd = tracker.get_max_speed(tid)
            self.metrics.max_speeds_mps[tid]  = max(
                self.metrics.max_speeds_mps.get(tid, 0.0), spd)

    def _closest_player(self, ball: tuple, tracks: dict, pitch_pos: dict) -> int:
        best_d, best_id = self.POSSESSION_RADIUS, -1
        bx, by = ball
        for cat in ("players", "goalkeepers"):
            for p in tracks.get(cat, []):
                pos = pitch_pos.get(p["track_id"])
                if pos is None:
                    continue
                d = ((pos[0]-bx)**2 + (pos[1]-by)**2)**0.5
                if d < best_d:
                    best_d, best_id = d, p["track_id"]
        return best_id

    def get_summary(self) -> dict:
        total_poss = max(sum(self.metrics.possession_frames.values()), 1)
        poss_a = self.metrics.possession_frames.get(0,0) / total_poss * 100
        poss_b = self.metrics.possession_frames.get(1,0) / total_poss * 100

        players_stats = {
            str(tid): {
                "team_id":       team,
                "distance_km":   round(self.metrics.distances_m.get(tid,0)/1000, 3),
                "max_speed_kmh": round(self.metrics.max_speeds_mps.get(tid,0)*3.6, 1),
            }
            for tid, team in self.metrics.player_teams.items()
        }
        passes_list = [
            {"from": k[0], "to": k[1], "count": v}
            for k, v in sorted(self.metrics.pass_matrix.items(), key=lambda x:-x[1])[:20]
        ]

        def norm(h):
            mx = h.max()
            return (h/mx).tolist() if mx > 0 else h.tolist()

        return {
            "total_frames": self.metrics.total_frames,
            "possession":   {"team_a_pct": round(poss_a,1), "team_b_pct": round(poss_b,1)},
            "heatmaps":     {"team_a": norm(self.metrics.heatmap_a),
                             "team_b": norm(self.metrics.heatmap_b)},
            "passes":       passes_list,
            "players":      players_stats,
        }
