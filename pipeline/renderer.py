"""
pipeline/renderer.py — Rendu visuel style Veo Analytics.
Ellipses équipes, tracé balle, minimap radar 2D.
"""
from __future__ import annotations
import cv2
import numpy as np
from collections import deque

PITCH_W = 105.0
PITCH_H = 68.0

COLOR_TEAM_A  = (235, 115,  50)
COLOR_TEAM_B  = ( 50, 200, 235)
COLOR_REFEREE = ( 30, 230,  30)
COLOR_GK_RING = (255, 215,   0)
COLOR_UNDEF   = (160, 160, 160)
TEAM_COLORS   = {0: COLOR_TEAM_A, 1: COLOR_TEAM_B, 2: COLOR_REFEREE}


class FootballRenderer:
    def __init__(self, frame_w: int, frame_h: int,
                 ball_trail_len: int = 30, minimap_scale: float = 0.22):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.mm_w    = int(frame_w * minimap_scale)
        self.mm_h    = int(self.mm_w * (PITCH_H / PITCH_W))
        self.mm_mg   = 8
        self.ball_trail: deque = deque(maxlen=ball_trail_len)
        self._pitch_bg = self._make_pitch_bg()

    def _make_pitch_bg(self) -> np.ndarray:
        w, h, mg = self.mm_w, self.mm_h, self.mm_mg
        bg = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8)

        def p(xm, ym):
            return (int(mg + xm/PITCH_W*(w-2*mg)), int(mg + ym/PITCH_H*(h-2*mg)))

        lc = (255, 255, 255)
        cv2.rectangle(bg, p(0,0), p(PITCH_W, PITCH_H), lc, 1)
        cv2.line(bg, p(52.5,0), p(52.5, PITCH_H), lc, 1)
        r = int(9.15/PITCH_W*(w-2*mg))
        cv2.circle(bg, p(52.5,34), r, lc, 1)
        cv2.circle(bg, p(52.5,34), 2, lc, -1)
        cv2.rectangle(bg, p(0,13.84), p(16.5,54.16), lc, 1)
        cv2.rectangle(bg, p(0,24.84), p(5.5,43.16),  lc, 1)
        cv2.rectangle(bg, p(88.5,13.84), p(PITCH_W,54.16), lc, 1)
        cv2.rectangle(bg, p(99.5,24.84), p(PITCH_W,43.16), lc, 1)
        cv2.circle(bg, p(11,34), 2, lc, -1)
        cv2.circle(bg, p(94,34), 2, lc, -1)
        return bg

    def draw_player(self, frame: np.ndarray, track: dict) -> None:
        x1, y1, x2, y2 = track["bbox"]
        tid   = track["track_id"]
        team  = track.get("team_id", -1)
        cls   = track.get("class_id", 0)
        color = TEAM_COLORS.get(team, COLOR_UNDEF)
        cx, rw = (x1+x2)//2, max(8, (x2-x1)//2)
        cv2.ellipse(frame, (cx, y2), (rw, max(4,rw//3)), 0, -45, 235, color, 2)
        if cls == 1:
            cv2.circle(frame, (cx, (y1+y2)//2), rw//2+4, COLOR_GK_RING, 2)
        cv2.putText(frame, str(tid), (cx-8, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

    def draw_referee(self, frame: np.ndarray, track: dict) -> None:
        x1, y1, x2, y2 = track["bbox"]
        cx, rw = (x1+x2)//2, max(8,(x2-x1)//2)
        cv2.ellipse(frame, (cx,y2), (rw,max(4,rw//3)), 0,-45,235, COLOR_REFEREE, 2)
        cv2.putText(frame, f"R{track['track_id']}", (cx-12, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_REFEREE, 1, cv2.LINE_AA)

    def draw_ball(self, frame: np.ndarray, ball: dict | None) -> None:
        if ball is None:
            return
        cx, cy = ball["center"]
        cv2.circle(frame, (cx,cy), 12, (255,255,255), 2)
        cv2.circle(frame, (cx,cy),  4, (0,220,255), -1)
        self.ball_trail.append((cx,cy))

    def draw_ball_trail(self, frame: np.ndarray) -> None:
        pts = list(self.ball_trail); n = len(pts)
        for i in range(1, n):
            a = i / n
            cv2.line(frame, pts[i-1], pts[i],
                     (int(a*220), int(a*220), 255), max(1,int(a*4)), cv2.LINE_AA)

    def draw_minimap(self, pitch_pos: dict, tracks: dict, homography) -> np.ndarray:
        mm      = self._pitch_bg.copy()
        team_map= {o["track_id"]: o.get("team_id",-1)
                   for cat in ("players","goalkeepers","referees")
                   for o in tracks.get(cat,[])}
        for tid, pos_m in pitch_pos.items():
            px, py = homography.pitch_to_minimap(pos_m, self.mm_w, self.mm_h, self.mm_mg)
            px = int(np.clip(px, 0, self.mm_w-1))
            py = int(np.clip(py, 0, self.mm_h-1))
            if tid == -1:
                cv2.circle(mm,(px,py),4,(255,255,255),-1)
                cv2.circle(mm,(px,py),5,(0,0,0),1)
            else:
                color = TEAM_COLORS.get(team_map.get(tid,-1), COLOR_UNDEF)
                cv2.circle(mm,(px,py),5,color,-1)
                cv2.circle(mm,(px,py),6,(0,0,0),1)
        cv2.rectangle(mm,(0,0),(self.mm_w-1,self.mm_h-1),(200,200,200),1)
        return mm

    def overlay_minimap(self, frame: np.ndarray, mm: np.ndarray) -> None:
        pad = 10
        x1  = self.frame_w - self.mm_w - pad
        y1  = self.frame_h - self.mm_h - pad
        roi = frame[y1:y1+self.mm_h, x1:x1+self.mm_w]
        cv2.addWeighted(mm, 0.85, roi, 0.15, 0, roi)
        frame[y1:y1+self.mm_h, x1:x1+self.mm_w] = roi

    def draw_hud(self, frame: np.ndarray, summary: dict) -> None:
        """Barre de possession en bas de l'image."""
        poss  = summary.get("possession", {})
        pa, pb = poss.get("team_a_pct",0), poss.get("team_b_pct",0)
        bar_w = 220; bar_h = 18; x0 = 10; y0 = self.frame_h - 45
        fill_a = int(bar_w * pa / 100)
        overlay = frame.copy()
        cv2.rectangle(overlay,(x0-4,y0-22),(x0+bar_w+4,y0+bar_h+6),(0,0,0),-1)
        cv2.addWeighted(overlay,0.5,frame,0.5,0,frame)
        cv2.rectangle(frame,(x0,y0),(x0+fill_a,y0+bar_h),COLOR_TEAM_A,-1)
        cv2.rectangle(frame,(x0+fill_a,y0),(x0+bar_w,y0+bar_h),COLOR_TEAM_B,-1)
        cv2.rectangle(frame,(x0,y0),(x0+bar_w,y0+bar_h),(255,255,255),1)
        cv2.putText(frame,f"A {pa:.0f}%",(x0,y0-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,COLOR_TEAM_A,1,cv2.LINE_AA)
        cv2.putText(frame,f"{pb:.0f}% B",(x0+bar_w-55,y0-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,COLOR_TEAM_B,1,cv2.LINE_AA)

    def render_frame(self, frame: np.ndarray, tracks: dict,
                     pitch_pos: dict, homography, summary: dict,
                     frame_idx: int) -> np.ndarray:
        """Rendu complet d'une frame annotée."""
        out = frame.copy()
        self.draw_ball_trail(out)
        for p  in tracks.get("players",    []): self.draw_player(out, p)
        for gk in tracks.get("goalkeepers",[]): self.draw_player(out, gk)
        for rf in tracks.get("referees",   []): self.draw_referee(out, rf)
        self.draw_ball(out, tracks.get("ball"))
        if homography.H is not None and pitch_pos:
            mm = self.draw_minimap(pitch_pos, tracks, homography)
            self.overlay_minimap(out, mm)
        if summary:
            self.draw_hud(out, summary)
        return out

