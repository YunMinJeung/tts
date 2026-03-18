@echo off
chcp 65001 >nul
title Qwen3-TTS
echo ============================================
echo  Qwen3-TTS Web UI
echo  http://localhost:7860
echo ============================================
echo.
wsl -e bash -c "cd /mnt/d/APPforBaby/TTS && ./run.sh"
echo.
echo [Done] Check errors above.
pause
