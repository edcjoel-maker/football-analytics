"""
server.py — Serveur FastAPI Football Analytics
Démarrage : uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints :
  POST /analyze          — Upload vidéo → task_id (traitement GPU background)
  GET  /status/{id}      — Progression % + étape courante
  GET  /results/{id}     — JSON statistiques du match
  GET  /video/{id}       — Téléchargement vidéo annotée
  GET  /health           — Vérification GPU
"""
from __future__ import annotations
import os, sys, uuid, time, json, logging, traceback
from pathlib import Path
from enum import Enum

import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

# ─── Chemins ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
MODELS_DIR  = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
for d in (UPLOADS_DIR, OUTPUTS_DIR):
    d.mkdir(exist_ok=True)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("football_analytics")


class TaskStatus(str, Enum):
    QUEUED    = "queued"
    DETECTING = "detecting"
    TRACKING  = "tracking"
    RENDERING = "rendering"
    METRICS   = "metrics"
    DONE      = "done"
    ERROR     = "error"


TASKS: dict[str, dict] = {}


def make_task(tid: str, vpath: Path) -> dict:
    return {"task_id": tid, "status": TaskStatus.QUEUED, "progress": 0,
            "step": "En attente", "video_path": str(vpath),
            "output_path": str(OUTPUTS_DIR/f"{tid}_annotated.mp4"),
            "json_path":   str(OUTPUTS_DIR/f"{tid}_results.json"),
            "error": None, "created_at": time.time(), "metrics": None}


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Football Analytics API", version="1.0.0",
              description="Analyse vidéo football style Veo Analytics 2")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ─── Interface Web ────────────────────────────────────────────────────────────
STATIC_DIR = BASE_DIR / "static"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Sert la page web de l'interface utilisateur."""
    html_path = STATIC_DIR / "index.html"
    async with aiofiles.open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=await f.read())


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    gpu = cuda or mps
    gpu_name = torch.cuda.get_device_name(0) if cuda else ("Apple Silicon (MPS)" if mps else None)
    return {"status": "ok", "gpu_available": gpu,
            "gpu_name": gpu_name,
            "device_type": "cuda" if cuda else ("mps" if mps else "cpu"),
            "cuda_version": getattr(torch.version, "cuda", None),
            "active_tasks": len([t for t in TASKS.values()
                                  if t["status"] not in (TaskStatus.DONE, TaskStatus.ERROR)])}


def extract_veo_stream_url(url: str) -> str:
    """
    Extrait le flux vidéo direct depuis une URL de match Veo
    (ex: https://app.veo.co/matches/20260929-untitled-recording-2026-09-29_22-38-52-v61cb888/)
    """
    import urllib.request, re, json
    # Si c'est déjà un lien direct .mp4 ou autre
    if url.lower().endswith(".mp4") or "video.mp4" in url:
        return url

    # Match Veo slug
    m = re.search(r'matches/([a-zA-Z0-9_\-]+)', url)
    if not m:
        return url

    slug = m.group(1).rstrip("/")
    api_url = f"https://app.veo.co/api/app/matches/{slug}/"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    rec_id = data.get("identifier")
    thumb = data.get("thumbnail") or ""
    # Format thumbnail : https://c.veocdn.com/{rec_id}/standard/machine/{hash}/thumbnail.jpg
    hm = re.search(r'/standard/machine/([^/]+)/thumbnail\.jpg', thumb)
    if hm and rec_id:
        vhash = hm.group(1)
        return f"https://c.veocdn.com/{rec_id}/standard/machine/{vhash}/video.mp4"

    # Fallback reel_url si dispo
    if data.get("reel_url"):
        return data["reel_url"]

    return url


@app.post("/analyze-url")
async def analyze_url(background_tasks: BackgroundTasks,
                      payload: dict):
    """Reçoit une URL Veo ou vidéo directe et lance l'analyse sans téléchargement manuel."""
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "URL requise")

    try:
        video_stream_url = extract_veo_stream_url(url)
    except Exception as e:
        raise HTTPException(400, f"Impossible d'extraire la vidéo depuis ce lien Veo : {e}")

    tid   = str(uuid.uuid4())
    log.info(f"[{tid}] Analyse lancée depuis URL : {url} -> {video_stream_url}")
    TASKS[tid] = make_task(tid, video_stream_url)
    background_tasks.add_task(run_pipeline, tid)
    return {"task_id": tid, "source_url": url,
            "message": "Traitement lancé — GET /status/{task_id}"}


@app.post("/analyze")
async def analyze(background_tasks: BackgroundTasks,
                  file: UploadFile = File(...)):
    """Upload vidéo et démarre le traitement GPU en tâche de fond."""
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    sfx     = Path(file.filename or "x.mp4").suffix.lower()
    if sfx not in allowed:
        raise HTTPException(400, f"Format non supporté. Acceptés : {allowed}")

    tid   = str(uuid.uuid4())
    vpath = UPLOADS_DIR / f"{tid}{sfx}"

    async with aiofiles.open(vpath, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)

    log.info(f"[{tid}] Vidéo reçue : {file.filename} "
             f"({vpath.stat().st_size/1e6:.1f} MB)")
    TASKS[tid] = make_task(tid, vpath)
    background_tasks.add_task(run_pipeline, tid)
    return {"task_id": tid, "filename": file.filename,
            "message": "Traitement lancé — GET /status/{task_id}"}



@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(404, "Tâche introuvable")
    t = TASKS[task_id]
    return {"task_id": task_id, "status": t["status"],
            "progress": t["progress"], "step": t["step"], "error": t["error"]}


@app.get("/results/{task_id}")
async def get_results(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(404, "Tâche introuvable")
    t = TASKS[task_id]
    if t["status"] == TaskStatus.ERROR:
        raise HTTPException(500, f"Erreur : {t['error']}")
    if t["status"] != TaskStatus.DONE:
        return JSONResponse({"status": t["status"], "progress": t["progress"]}, 202)
    jpath = Path(t["json_path"])
    if not jpath.exists():
        raise HTTPException(404, "Résultats introuvables")
    async with aiofiles.open(jpath, "r") as f:
        return JSONResponse(json.loads(await f.read()))


@app.get("/video/{task_id}")
async def get_video(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(404, "Tâche introuvable")
    t = TASKS[task_id]
    if t["status"] != TaskStatus.DONE:
        return JSONResponse({"status": t["status"], "progress": t["progress"]}, 202)
    vp = Path(t["output_path"])
    if not vp.exists():
        raise HTTPException(404, "Vidéo introuvable")
    return FileResponse(str(vp), media_type="video/mp4",
                        filename=f"football_{task_id[:8]}.mp4")


@app.get("/tasks")
async def list_tasks():
    return [{"task_id": k, "status": v["status"], "progress": v["progress"]}
            for k, v in TASKS.items()]


# ─── Pipeline de traitement GPU ──────────────────────────────────────────────

def run_pipeline(task_id: str) -> None:
    """
    Pipeline complet de traitement vidéo football.
    Exécuté en thread de fond par BackgroundTasks de FastAPI.

    Étapes :
      1. Initialisation des modules (Detector, Tracker, TeamAssigner, Homography)
      2. Boucle frame par frame : Détection → Tracking → Équipes → Homographie
      3. Calcul métriques
      4. Rendu visuel + écriture vidéo de sortie
      5. Export JSON
    """
    from pipeline.detector      import FootballDetector
    from pipeline.tracker       import FootballTracker
    from pipeline.team_assigner import TeamAssigner
    from pipeline.homography    import PitchHomography
    from pipeline.metrics       import MetricsCalculator
    from pipeline.renderer      import FootballRenderer

    task = TASKS[task_id]

    def set_progress(pct: int, step: str, status: TaskStatus = TaskStatus.DETECTING):
        task["progress"] = pct
        task["step"]     = step
        task["status"]   = status
        log.info(f"[{task_id}] {pct:3d}% — {step}")

    try:
        set_progress(0, "Initialisation des modèles...", TaskStatus.DETECTING)

        # ── Initialisation ────────────────────────────────────────────────
        detector  = FootballDetector(
            player_model_path=str(MODELS_DIR / "player_detection.pt"),
            ball_model_path=str(MODELS_DIR   / "ball_detection.pt"),
        )
        homography = PitchHomography(
            model_path=str(MODELS_DIR / "pitch_keypoints.pt"),
            update_every=25,
        )
        tracker  = FootballTracker(frame_rate=25.0)
        assigner = TeamAssigner()
        metrics  = MetricsCalculator()

        # ── Ouverture vidéo ───────────────────────────────────────────────
        cap = cv2.VideoCapture(task["video_path"])
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo : {task['video_path']}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tracker.fps  = fps

        log.info(f"[{task_id}] Vidéo : {frame_w}x{frame_h} @ {fps:.1f} fps, "
                 f"{total_frames} frames")

        # ── Writer de sortie (H.264) ──────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            task["output_path"], fourcc, fps, (frame_w, frame_h)
        )

        renderer = FootballRenderer(frame_w, frame_h)
        summary  = {}
        frame_idx = 0

        set_progress(5, "Analyse en cours...", TaskStatus.TRACKING)

        # ── Boucle principale ─────────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            pct = min(95, 5 + int(frame_idx / max(total_frames, 1) * 88))
            if frame_idx % 30 == 0:
                set_progress(pct, f"Frame {frame_idx}/{total_frames}", TaskStatus.RENDERING)

            # 1. Détection
            dets = detector.detect_frame(frame)

            # 2. Tracking ByteTrack
            tracks = tracker.update(dets)

            # 3. Attribution équipes (K-Means)
            tracks = assigner.assign_teams_to_frame(frame, tracks)

            # Mise à jour team_id dans le tracker pour métriques
            for p in tracks.get("players", []) + tracks.get("goalkeepers", []):
                if p.get("team_id", -1) != -1:
                    tracker.assign_team(p["track_id"], p["team_id"])

            # 4. Homographie terrain
            homography.update(frame)
            pitch_pos = homography.project_all(tracks)

            # Mise à jour positions terrain dans le tracker (pour distances/vitesses)
            for tid, pos_m in pitch_pos.items():
                if tid != -1:
                    is_ref = any(r["track_id"] == tid for r in tracks.get("referees", []))
                    tracker.update_pitch_pos(tid, pos_m, is_ref)

            # 5. Métriques
            metrics.update(tracks, pitch_pos, tracker)
            if frame_idx % 30 == 0:
                summary = metrics.get_summary()

            # 6. Rendu visuel
            annotated = renderer.render_frame(
                frame, tracks, pitch_pos, homography, summary, frame_idx
            )
            writer.write(annotated)

        # ── Finalisation ──────────────────────────────────────────────────
        cap.release()
        writer.release()

        set_progress(97, "Calcul des métriques finales...", TaskStatus.METRICS)
        final_summary = metrics.get_summary()
        final_summary["video_info"] = {
            "total_frames": frame_idx,
            "fps": fps,
            "duration_s": round(frame_idx / fps, 1),
            "resolution": f"{frame_w}x{frame_h}",
        }

        # Export JSON
        import json as _json
        with open(task["json_path"], "w", encoding="utf-8") as jf:
            _json.dump(final_summary, jf, ensure_ascii=False, indent=2)

        task["metrics"] = final_summary
        set_progress(100, "Terminé ✓", TaskStatus.DONE)
        log.info(f"[{task_id}] ✅ Pipeline terminé — {frame_idx} frames traitées")

    except Exception as exc:
        task["status"] = TaskStatus.ERROR
        task["error"]  = str(exc)
        task["step"]   = f"Erreur : {exc}"
        log.error(f"[{task_id}] ❌ Erreur pipeline : {traceback.format_exc()}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


