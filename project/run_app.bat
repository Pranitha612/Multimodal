@echo off
echo Starting Emotion & Relationship Recognition App...
call venv\Scripts\activate.bat
python -m streamlit run app.py
pause
