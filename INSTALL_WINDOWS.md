# 🏟️ Football Analytics System — Installation Windows + NVIDIA GPU

## Prérequis système
- Windows 10/11 64-bit
- NVIDIA GPU (RTX 2060+ recommandé, RTX 3080/4090 idéal)
- CUDA 12.8 (driver NVIDIA ≥ 527.41)
- Python 3.11 (recommandé)
- Git for Windows

## Étape 1 : Vérifier/Installer CUDA

Télécharger depuis : https://developer.nvidia.com/cuda-12-8-0-download-archive
Puis vérifier :
```powershell
nvidia-smi
# Doit afficher "CUDA Version: 12.x"
```

## Étape 2 : Installer Python 3.11

Télécharger depuis : https://www.python.org/downloads/release/python-3119/
⚠️ Cocher "Add Python to PATH" lors de l'installation

## Étape 3 : Créer l'environnement virtuel

```powershell
cd C:\football_analytics
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

## Étape 4 : Installer PyTorch avec CUDA 12.8

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Étape 5 : Installer toutes les dépendances

```powershell
pip install ultralytics==8.3.0
pip install supervision==0.25.0
pip install opencv-python==4.10.0.84
pip install fastapi==0.115.0
pip install uvicorn[standard]==0.32.0
pip install python-multipart==0.0.12
pip install numpy==1.26.4
pip install scipy==1.14.1
pip install scikit-learn==1.5.2
pip install matplotlib==3.9.2
pip install pillow==10.4.0
pip install huggingface-hub==0.26.0
pip install aiofiles==24.1.0
pip install tqdm==4.67.0
```

## Étape 6 : Vérifier l'installation GPU

```python
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Résultat attendu :
```
CUDA: True
GPU: NVIDIA GeForce RTX XXXX
```

## Étape 7 : Télécharger les modèles pré-entraînés

```powershell
python download_models.py
```

## Structure des fichiers attendus

```
C:\football_analytics\
├── venv\
├── models\
│   ├── player_detection.pt       # YOLO11m - joueurs/gardiens/arbitres/balle
│   ├── ball_detection.pt         # YOLO11n - ballon dédié
│   └── pitch_keypoints.pt        # YOLOv8x-pose - 32 keypoints terrain
├── outputs\                      # Vidéos et JSON générés
├── uploads\                      # Vidéos uploadées
├── server.py                     # Serveur FastAPI
├── pipeline\                     # Modules du pipeline
│   ├── __init__.py
│   ├── detector.py
│   ├── tracker.py
│   ├── team_assigner.py
│   ├── homography.py
│   ├── metrics.py
│   └── renderer.py
├── download_models.py
└── run_server.bat
```
