# 🍎 Football Analytics System — Guide d'installation macOS (Apple Silicon / Intel)

## Prérequis système
- macOS 12+ (Monterey, Ventura, Sonoma, Sequoia)
- Apple Silicon (M1/M2/M3/M4) avec accélération GPU Metal Performance Shaders (MPS) ou Mac Intel
- Python 3.10, 3.11 ou 3.12 (recommandé via Homebrew)
- Git

## Étape 1 : Cloner le projet (si pas déjà fait)
```bash
git clone <url-du-repo>
cd football_analytics
```

## Étape 2 : Créer et activer l'environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

## Étape 3 : Installer PyTorch pour macOS
Sur macOS, PyTorch inclut directement le support GPU Apple Silicon (MPS) :
```bash
pip install torch torchvision torchaudio
```

Vérifier la détection MPS :
```bash
python3 -c "import torch; print('MPS disponible:', torch.backends.mps.is_available())"
```

## Étape 4 : Installer les dépendances du projet
```bash
pip install -r requirements.txt
```

## Étape 5 : Télécharger les modèles pré-entraînés
```bash
python3 download_models.py
```

## Étape 6 : Lancer le serveur
```bash
./run_server.sh
# ou directement
python3 server.py
```

Accédez à l'interface web dans votre navigateur :
- **Interface Web** : http://localhost:8000
- **Documentation API** : http://localhost:8000/docs
