@echo off
title Hotel Review Intelligence
echo.
echo  Hotel Review Intelligence — Local Startup
echo  ==========================================
echo.

REM ── Check .env ──────────────────────────────────────────────────────────────
if not exist ".env" (
    echo  ERROR: .env file not found.
    echo  Copy .env.example to .env and add your OPENAI_API_KEY.
    echo.
    pause
    exit /b 1
)

REM ── Stage 5: ingest if ChromaDB not yet built ────────────────────────────────
if not exist "chromadb\" (
    echo  ChromaDB not found. Running Stage 5 ingestion...
    echo  This embeds ~735k review sentences via OpenAI text-embedding-3-small.
    echo  Estimated time: 20-40 minutes. Estimated cost: ~$0.30.
    echo.
    pause

    .venv\Scripts\python.exe scripts\run_pipeline.py --only 5
    if errorlevel 1 (
        echo.
        echo  Stage 5 failed. Check the error above and re-run.
        pause
        exit /b 1
    )
    echo.
    echo  Ingestion complete.
    echo.
)

REM ── Launch UI ────────────────────────────────────────────────────────────────
echo  Starting Gradio UI at http://localhost:7860
echo  Press Ctrl+C to stop.
echo.
.venv\Scripts\python.exe src\ui\app.py
pause
