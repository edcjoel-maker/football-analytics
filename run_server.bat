@echo off
:: ============================================================
:: run_server.bat — Lancement du serveur Football Analytics
:: Double-cliquer pour démarrer. Accessible sur http://localhost:8000
:: ============================================================

title Football Analytics Server
color 0A

echo.
echo  ============================================================
echo   Football Analytics Server — Veo Analytics Clone
echo  ============================================================
echo.

:: Répertoire du script
cd /d "%~dp0"

:: Vérifier que l'environnement virtuel existe
if not exist "venv\Scripts\activate.bat" (
    echo  [ERREUR] Environnement virtuel introuvable.
    echo  Executez d'abord :
    echo    python -m venv venv
    echo    venv\Scripts\activate
    echo    pip install -r requirements.txt
    pause
    exit /b 1
)

:: Activer l'environnement virtuel
call venv\Scripts\activate.bat

:: Vérifier que les modèles sont présents
if not exist "models\player_detection.pt" (
    echo  [INFO] Modeles absents. Telechargement en cours...
    python download_models.py
    if errorlevel 1 (
        echo  [ERREUR] Echec du telechargement des modeles.
        pause
        exit /b 1
    )
)

:: Vérifier CUDA
echo  [INFO] Verification GPU NVIDIA...
python -c "import torch; gpu=torch.cuda.is_available(); print(f'  GPU: {torch.cuda.get_device_name(0) if gpu else \"CPU uniquement\"} | CUDA: {gpu}')"

echo.
echo  [INFO] Demarrage du serveur sur http://0.0.0.0:8000
echo  [INFO] Interface API : http://localhost:8000/docs
echo  [INFO] Appuyez sur Ctrl+C pour arreter
echo.

:: Lancer uvicorn avec les paramètres optimaux
:: --workers 1 obligatoire (les modeles GPU ne supportent pas le multi-process)
:: --timeout-keep-alive 300 pour les uploads longs
uvicorn server:app ^
    --host 0.0.0.0 ^
    --port 8000 ^
    --workers 1 ^
    --loop asyncio ^
    --timeout-keep-alive 300 ^
    --log-level info

echo.
echo  [INFO] Serveur arrete.
pause
