"""
download_models.py
Télécharge les modèles pré-entraînés depuis HuggingFace.
Exécuter une seule fois avant de lancer le serveur.
"""
import os
import urllib.request
from pathlib import Path

# Répertoire de destination
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "player_detection.pt": (
        "https://huggingface.co/martinjolif/yolo-football-player-detection"
        "/resolve/main/yolo-football-player-detection.pt"
    ),
    "ball_detection.pt": (
        "https://huggingface.co/martinjolif/yolo-football-ball-detection"
        "/resolve/main/yolo-football-ball-detection.pt"
    ),
    "pitch_keypoints.pt": (
        "https://huggingface.co/martinjolif/yolo-football-pitch-detection"
        "/resolve/main/yolo-football-pitch-detection.pt"
    ),
}


def download_with_progress(url: str, dest: Path) -> None:
    """Télécharge un fichier avec barre de progression simple."""
    if dest.exists():
        print(f"  ✅ {dest.name} déjà présent — téléchargement ignoré.")
        return

    print(f"  ⬇️  Téléchargement de {dest.name}...")
    print(f"      Source : {url}")

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb_done = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r      {percent:3d}% | {mb_done:.1f} / {mb_total:.1f} MB", end="")

    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print(f"\n  ✅ {dest.name} téléchargé ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    print("=" * 60)
    print("  Football Analytics — Téléchargement des modèles")
    print("=" * 60)

    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        try:
            download_with_progress(url, dest)
        except Exception as e:
            print(f"\n  ❌ Erreur pour {filename}: {e}")
            print(f"     Téléchargez manuellement depuis :\n     {url}")
            print(f"     Et placez le fichier dans : {MODELS_DIR.resolve()}")

    print("\n" + "=" * 60)
    print("  Tous les modèles sont prêts.")
    print("  Lancez maintenant : run_server.bat")
    print("=" * 60)


if __name__ == "__main__":
    main()
