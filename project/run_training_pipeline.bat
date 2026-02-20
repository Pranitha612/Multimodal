@echo off
echo Starting Training Pipeline...

echo 1. Training Emotion Recognition Model (EfficientNet-B0)
c:\Users\Ritwik\relation-project\emotion-relationship-recognition\venv\Scripts\python.exe train_emotion.py
if %ERRORLEVEL% NEQ 0 (
    echo Emotion training failed!
    exit /b %ERRORLEVEL%
)

echo.
echo 2. Training Relationship Detection Model (Using Emotion Embeddings)
c:\Users\Ritwik\relation-project\emotion-relationship-recognition\venv\Scripts\python.exe train_relationship.py
if %ERRORLEVEL% NEQ 0 (
    echo Relationship training failed!
    exit /b %ERRORLEVEL%
)

echo.
echo Training Pipeline Completed Successfully!
echo You can now run the app with: python -m streamlit run app.py
pause
