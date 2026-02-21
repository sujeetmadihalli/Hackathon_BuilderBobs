# BuilderBobs Tracking Pipeline

This repository contains the source code for a computer vision pipeline developed during a hackathon to analyze construction worker activity and productivity from first-person (POV/body-cam) video feeds.

## Overview

The system processes raw video footage to accomplish the following:
1.  **Track Hand Movement:** Utilizes the MediaPipe Tasks Vision API to identify and track the pixel coordinates of the left and right wrists.
2.  **Detect Construction Objects:** Uses a custom-trained YOLOv8 model (`yolov8n-construction.pt`) to detect context-specific objects in the frame (e.g., hard hats, tools, materials).
3.  **Quantify Exertion and Productivity:** Calculates the Euclidean pixel distance traveled by the hands. If the hands are moving past a specific velocity threshold *and* interacting with a detected object, the worker is mathematically classified as "active".
4.  **Visualize Data:** Provides an interactive `streamlit` dashboard for supervisors to view site-wide aggregates, leaderboards, and individual worker performance plots.
5.  **Generate AI Site Reports:** Uses the Gemini 2.5 Flash LLM to automatically parse the generated pipeline metrics and produce a qualitative executive summary focused on productivity standouts and safety/ergonomic warnings.

## Core Components

*   **`first_person_pipeline.py`**: The core batch-processing engine. It reads all `.mp4` files from the `IronsiteHackathonData/` directory. To optimize processing time without losing the full timeline context, it mathematically downsamples the video feeds (e.g., analyzing at 5 FPS instead of 30 FPS). Outputs include:
    *   Annotated MP4 videos with YOLO bounding boxes and MediaPipe landmarks.
    *   Frame-by-frame exertion CSVs.
    *   Matplotlib exertion line plots highlighting periods of active work.
    *   A single `master_dashboard.csv` containing aggregated statistics for the entire site.
*   **`dashboard.py`**: A Streamlit web application that reads the `master_dashboard.csv` to provide a clean, high-level UI for supervisors. Run via `streamlit run dashboard.py`.
*   **`analyze_results.py`**: A standalone script that pipes the `master_dashboard.csv` aggregates into a structured prompt for the Gemini LLM. It outputs `outputs/Final_AI_Site_Report.txt`.
*   **`excertion.py` & `test.py`**: Legacy scratchpad files demonstrating early iterations using YOLOv8 pose estimation before the pivot to the POV MediaPipe architecture.

## Requirements

*   Python 3.10+
*   `opencv-python`
*   `mediapipe`
*   `ultralytics`
*   `pandas`
*   `matplotlib`
*   `streamlit`
*   `loguru`
*   `google-generativeai` (for the Gemini reporting script)

## Setup & Execution

1.  Place raw `.mp4` videos in an `IronsiteHackathonData/` directory.
2.  Ensure you have the required models downloaded (`hand_landmarker.task` and `yolov8n-construction.pt`) in the root directory.
3.  Run the main pipeline:
    ```bash
    python first_person_pipeline.py
    ```
4.  Launch the supervisor dashboard:
    ```bash
    streamlit run dashboard.py
    ```
5.  (Optional) Generate an AI Site Report (requires a valid Google AI Studio API key in the script):
    ```bash
    python analyze_results.py
    ```
