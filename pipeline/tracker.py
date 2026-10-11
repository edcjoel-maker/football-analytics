"""
pipeline/tracker.py
Tracking ByteTrack (supervision) pour le football.
IDs stables pour joueurs, gardiens, arbitres et ballon.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from supervision import ByteTrack, Detections


@dataclass
class TrackedObject:
    """Objet suivi avec historique de positions terrain."""
    track_id:    int
    class_id:    int
    team_id:     int = -1      # -1=non assigné, 0=TeamA, 1=TeamB, 2=Arbitre
    positions:   list = field(default_factory=list)
    speeds_mps:  list = field(default_factory=list)
    last_bbox:   list = field(default_factory=list)
    last_center: tuple = (0, 0)
    frames_seen: int = 0


class FootballTracker:
    """ByteTrack pour le football avec persistance forte des IDs."""

    def __init__(self, frame_rate: float = 25.0, track_thresh: float = 0.50,
                 track_buffer: int = 90, match_thresh: float = 0.70):
        self.fps = frame_rate
        # track_buffer à 90 frames (~3-4s) pour qu'un joueur masqué ou en bordure garde son ID
        kw = dict(track_activation_threshold=track_thresh,
                  lost_track_buffer=track_buffer,
                  minimum_matching_threshold=match_thresh,
                  frame_rate=int(frame_rate))
        self.player_tracker  = ByteTrack(**kw)
        self.referee_tracker = ByteTrack(**kw)
        self.ball_tracker    = ByteTrack(track_activation_threshold=0.35,
                                         lost_track_buffer=30,
                                         minimum_matching_threshold=0.65,
                                         frame_rate=int(frame_rate))
        self.tracked_players:  dict[int, TrackedObject] = {}
        self.tracked_referees: dict[int, TrackedObject] = {}
        self.ball_track: TrackedObject | None = None
        self.frame_count = 0


    @staticmethod
    def _to_sv(dets: list[dict], cls_override: int | None = None) -> Detections:
        if not dets:
            return Detections.empty()
        xyxy = np.array([d["bbox"] for d in dets], dtype=np.float32)
        conf = np.array([d["conf"] for d in dets], dtype=np.float32)
        cids = np.array([cls_override if cls_override is not None
                         else d["class_id"] for d in dets], dtype=int)
        return Detections(xyxy=xyxy, confidence=conf, class_id=cids)

    @staticmethod
    def _row(sv: Detections, i: int, cls: int, team: int) -> dict:
        x = sv.xyxy[i].astype(int)
        cx, cy = int((x[0]+x[2])//2), int((x[1]+x[3])//2)
        return {"track_id": int(sv.tracker_id[i]), "bbox": x.tolist(),
                "center": (cx, cy), "foot": (cx, int(x[3])),
                "class_id": cls, "team_id": team}

    def update(self, fd: dict) -> dict:
        """Met à jour ByteTrack avec les détections d'une frame."""
        self.frame_count += 1

        # Joueurs + Gardiens
        all_fp = fd["players"] + fd["goalkeepers"]
        t_fp   = self.player_tracker.update_with_detections(self._to_sv(all_fp))
        players, gks = [], []
        for i in range(len(t_fp)):
            tid, cls = int(t_fp.tracker_id[i]), int(t_fp.class_id[i])
            if tid not in self.tracked_players:
                self.tracked_players[tid] = TrackedObject(track_id=tid, class_id=cls)
            obj = self.tracked_players[tid]
            obj.frames_seen += 1
            xyxy = t_fp.xyxy[i].astype(int)
            obj.last_center = (int((xyxy[0]+xyxy[2])//2), int((xyxy[1]+xyxy[3])//2))
            obj.last_bbox   = xyxy.tolist()
            row = self._row(t_fp, i, cls, obj.team_id)
            (gks if cls == 1 else players).append(row)

        # Arbitres
        t_ref = self.referee_tracker.update_with_detections(
            self._to_sv(fd["referees"], cls_override=2))
        refs = []
        for i in range(len(t_ref)):
            tid = int(t_ref.tracker_id[i])
            if tid not in self.tracked_referees:
                self.tracked_referees[tid] = TrackedObject(track_id=tid, class_id=2, team_id=2)
            self.tracked_referees[tid].frames_seen += 1
            xyxy = t_ref.xyxy[i].astype(int)
            self.tracked_referees[tid].last_center = (
                int((xyxy[0]+xyxy[2])//2), int((xyxy[1]+xyxy[3])//2))
            refs.append(self._row(t_ref, i, 2, 2))

        # Ballon
        ball = None
        bd   = fd.get("ball")
        if bd:
            t_b = self.ball_tracker.update_with_detections(
                self._to_sv([bd], cls_override=3))
            if len(t_b) > 0:
                tid  = int(t_b.tracker_id[0])
                xyxy = t_b.xyxy[0].astype(int)
                cx, cy = int((xyxy[0]+xyxy[2])//2), int((xyxy[1]+xyxy[3])//2)
                if self.ball_track is None:
                    self.ball_track = TrackedObject(track_id=tid, class_id=3)
                self.ball_track.last_center = (cx, cy)
                self.ball_track.frames_seen += 1
                ball = {"track_id": tid, "bbox": xyxy.tolist(),
                        "center": (cx, cy), "conf": float(t_b.confidence[0])}

        return {"players": players, "goalkeepers": gks, "referees": refs, "ball": ball}

    def assign_team(self, track_id: int, team_id: int) -> None:
        if track_id in self.tracked_players:
            self.tracked_players[track_id].team_id = team_id

    def update_pitch_pos(self, track_id: int, xy: tuple, is_ref: bool = False) -> None:
        store = self.tracked_referees if is_ref else self.tracked_players
        if track_id not in store:
            return
        obj = store[track_id]
        
        # Filtre anti-téléportation / homographie aberrante :
        # Un joueur ne peut pas se déplacer de plus de 2.0 mètres entre deux frames consécutives (soit > 50-60 km/h)
        if len(obj.positions) >= 1:
            prev_x, prev_y = obj.positions[-1]
            dx = xy[0] - prev_x
            dy = xy[1] - prev_y
            step_m = (dx**2 + dy**2)**0.5
            max_step_per_frame = (10.0 / self.fps)  # max ~36 km/h = 10 m/s
            if step_m > max_step_per_frame:
                # Saut anormal (artefact homographie) : ignorer la position aberrante
                return
            spd_mps = step_m * self.fps
            obj.speeds_mps.append(spd_mps)

        obj.positions.append(xy)

    def get_distance(self, track_id: int) -> float:
        pos = self.tracked_players.get(track_id, TrackedObject(0, 0)).positions
        if len(pos) < 2:
            return 0.0
        tot = 0.0
        for i in range(1, len(pos)):
            d = ((pos[i][0]-pos[i-1][0])**2 + (pos[i][1]-pos[i-1][1])**2)**0.5
            tot += d
        return tot

    def get_max_speed(self, track_id: int) -> float:
        sp = self.tracked_players.get(track_id, TrackedObject(0, 0)).speeds_mps
        if not sp:
            return 0.0
        # Utiliser un quantile 95% pour éliminer les micro-sauts isolés
        sp_filtered = [s for s in sp if s <= 10.0]  # max 36 km/h (record Mbappe ~36 km/h)
        if not sp_filtered:
            return 0.0
        return float(np.percentile(sp_filtered, 95))

