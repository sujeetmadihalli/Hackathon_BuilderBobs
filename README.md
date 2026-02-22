# 🏗️ IronSite BuilderBobs — AI Construction Productivity Pipeline

> **Hackathon Project — February 2026**  
> Analyze construction worker body-cam footage with computer vision + AI vision models to power a real-time supervisor dashboard.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## Overview

BuilderBobs is a **two-stage AI pipeline** that processes first-person (POV) body-camera footage from construction workers:

| Stage | Technology | Output |
|-------|-----------|--------|
| **1 — Quantitative** | OpenCV · MediaPipe · YOLOv8 | Productivity %, Peak Exertion, Annotated Video |
| **2 — Qualitative** | Ollama LLaVA 7B (remote GPU) | Trade ID, Task Description, Universal Efficiency Score |

Both stages feed a unified **Streamlit supervisor dashboard**.

---

## Architecture

```
📹 Raw Body-Cam MP4s
        │
        ▼
┌─────────────────────────────────────┐
│  Stage 1 · first_person_pipeline.py │
│  • MediaPipe wrist tracking (5 FPS) │
│  • YOLOv8 construction objects      │
│  • Global motion compensation       │
│  • Activity classification          │
└───────────────┬─────────────────────┘
                │  master_dashboard.csv
                │  outputs/*_plot.png
                │  outputs/*_annotated.mp4
                ▼
┌─────────────────────────────────────┐
│  Stage 2 · agent_video_analyzer.py  │
│  • SSH tunnel → Ollama LLaVA (GPU)  │
│  • 16 frames sampled by ffmpeg      │
│  • Structured JSON response         │
│  • AI_Trade, AI_UES, AI_Summary     │
└───────────────┬─────────────────────┘
                │  outputs/Agent_Analysis_*.json
                │  master_dashboard.csv (enriched)
                ▼
        🖥️  dashboard.py (Streamlit)
```

---

## Results (14 Videos Processed)

| Metric | Value |
|--------|-------|
| Average Productivity | **91.9%** |
| Average AI Efficiency (UES) | **85.4 / 100** |
| Site Peak Exertion | **83.84 px** |
| Trades Identified | Construction Workers (11), Plumbers (3) |
| Stage 2 Processing Speed | **~8 seconds/video** on GPU |

---

## Project Structure

```
├── first_person_pipeline.py     # Stage 1: OpenCV batch processor
├── agent_video_analyzer.py      # Stage 2: Ollama LLaVA vision agent
├── batch_agent_analysis.py      # Runs Stage 2 across all videos
├── dashboard.py                 # Streamlit supervisor dashboard
├── analyze_results.py           # Gemini text-based site report (legacy)
├── apply_global_motion.py       # Camera-shake compensation utility
├── recalculate_metrics.py       # Recalculate metrics from existing CSVs
├── master_dashboard.csv         # Aggregated metrics (both stages)
├── requirements.txt             # Python dependencies
├── outputs/
│   ├── Agent_Analysis_*.json    # Per-video AI analysis
│   ├── *_plot.png               # Exertion time-series plots
│   ├── *_data.csv               # Per-frame exertion data
│   └── Final_AI_Site_Report.txt # Text-based executive summary
├── hand_landmarker.task         # MediaPipe model
├── yolov8n-construction.pt      # Custom YOLOv8 construction model
└── IronsiteHackathonData/       # Raw MP4s (gitignored)
```

---

## Setup & Execution

### Requirements

```bash
pip install -r requirements.txt
# Also requires: ffmpeg, opencv-python, mediapipe, ultralytics
```

### Stage 1 — OpenCV Pipeline

```bash
# Place raw .mp4 files in IronsiteHackathonData/
python3 first_person_pipeline.py
```

Outputs to `outputs/` and writes `master_dashboard.csv`.

### Stage 2 — AI Vision Agent (Ollama LLaVA)

Requires a running Ollama instance with `llava:latest`. Using Vast.ai remote GPU:

```bash
# 1. Start SSH tunnel (maps remote Ollama to localhost:11434)
ssh -p 56834 root@YOUR_VAST_IP -L 8080:localhost:11434

# 2. Set env var (or edit .env)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_VISION_MODEL=llava:latest

# 3. Run batch analysis
python3 batch_agent_analysis.py
```

To process a single video:
```bash
python3 agent_video_analyzer.py IronsiteHackathonData/14_production_mp.mp4
```

### Launch Dashboard

```bash
streamlit run dashboard.py
# → http://localhost:8501
```

### Deploy to Streamlit Community Cloud

1. Push repo to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo `sujeetmadihalli/Hackathon_BuilderBobs` · branch `main` · file `dashboard.py`
4. Click **Deploy**

---

## AI Analysis Output Schema

Each video produces an `outputs/Agent_Analysis_{video}.json`:

```json
{
  "primary_trade": "Plumber",
  "specific_tasks": "Fixing pipes, cutting materials",
  "quantified_output": "10 joints welded, 2 pipes cut",
  "universal_efficiency_score": 96,
  "performance_summary": "The plumber demonstrates high physical exertion and efficiency in completing the tasks at hand."
}
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Computer Vision | OpenCV, MediaPipe HandLandmarker, YOLOv8 |
| AI Vision Model | Ollama LLaVA 7B (self-hosted on Vast.ai) |
| Frame Extraction | ffmpeg |
| Dashboard | Streamlit + Altair |
| Data | pandas, CSV |
| Remote GPU | Vast.ai (SSH tunnel) |
| Deployment | Streamlit Community Cloud |

---

*BuilderBobs · IronSite Hackathon 2026*
