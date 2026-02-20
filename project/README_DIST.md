# How to Run Emotion & Relationship Recognition App

## Prerequisites
-   **Python 3.8+** must be installed on your computer.

## Installation (First Time Only)
1.  Double-click **`setup.bat`**.
2.  Wait for the script to create a virtual environment and install all necessary libraries.
3.  Once it says "Setup Complete!", close the window.

## Running the App
1.  Double-click **`run_app.bat`**.
2.  The application will open in your default web browser (usually at `http://localhost:8501`).
3.  To stop the app, close the terminal window.

## Usage
-   **Upload Image**: Click "Browse files" to upload a group photo (JPG/PNG).
-   **View Results**: The app will automatically detect faces, recognize emotions, and identifying relationships.

## Troubleshooting
-   **"Python not found"**: Ensure Python is installed and added to your system PATH.
-   **Model Errors**: Ensure the `checkpoints` folder contains `emotion_model_best.pth` and `relationship_model_best.pth`.
