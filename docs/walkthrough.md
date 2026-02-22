# IronSite BuilderBobs — Walkthrough

## What Was Built

A two-stage AI pipeline to analyze construction worker productivity from first-person body-camera footage:

1. **Stage 1 — Quantitative OpenCV Pipeline** (`first_person_pipeline.py`)  
   Deterministic computer vision: MediaPipe wrist tracking + YOLOv8 construction object detection → productivity score, exertion plot, annotated video.

2. **Stage 2 — AI Qualitative Vision Agent** (`agent_video_analyzer.py` + `batch_agent_analysis.py`)  
   Ollama LLaVA 7B running on a remote Vast.ai GPU, tunnelled over SSH, analyzes 16 sampled frames per video and produces structured JSON with trade identification, task description, quantified output, and Universal Efficiency Score (UES).

3. **Streamlit Dashboard** (`dashboard.py`)  
   Unified supervisor view showing both stages of data with Altair charts, trade breakdown, leaderboard, and per-video drill-down.

---

## Stage 1: OpenCV Pipeline

```bash
python3 first_person_pipeline.py
```

- Processed **14 videos** from `IronsiteHackathonData/`
- Ran MediaPipe HandLandmarker + YOLOv8 at **5 FPS** (6× speedup vs 30 FPS)
- Applied global motion compensation to remove camera-shake noise
- Output per video: `outputs/{name}_annotated.mp4`, `outputs/{name}_data.csv`, `outputs/{name}_plot.png`
- Aggregated metrics saved to `master_dashboard.csv`

---

## Stage 2: Ollama LLaVA via SSH Tunnel

### Tunnel Setup
```bash
ssh -p 56834 root@171.248.243.88 -L 8080:localhost:11434
```
This maps `localhost:11434` on your machine to the remote Ollama instance.

### Models Available on Remote
```
llava:latest     4.7 GB   ← used for vision analysis
llama3:latest    4.7 GB
qwen3:14b        9.3 GB
```

### Running Single Video Test
```bash
python3 agent_video_analyzer.py IronsiteHackathonData/14_production_mp.mp4
```
**Result:** Completed in **~7 seconds**. Extracted 16 frames, sent to LLaVA, received JSON, saved to `outputs/Agent_Analysis_14_production_mp.json`, updated `master_dashboard.csv`.

### Running Batch (All 14 Videos)
```bash
python3 batch_agent_analysis.py
```
**Result:** All 14 videos completed in **~2 minutes total** (~8s/video).

````carousel
```
2026-02-22 03:12:21 | INFO  | Starting batch AI analysis on 14 videos...
2026-02-22 03:12:34 | ✅    | Saved → outputs/Agent_Analysis_01_production_masonry.json
2026-02-22 03:12:51 | ✅    | Saved → outputs/Agent_Analysis_02_production_masonry.json
...
2026-02-22 03:15:53 | ✅    | Saved → outputs/Agent_Analysis_13_transit_prep_mp.json
2026-02-22 03:15:58 | INFO  | Skipping 14_production_mp.mp4 (already completed)
2026-02-22 03:15:58 | ✅    | Batch analysis complete!
```
<!-- slide -->
**Sample AI Output — `05_production_mp` (Plumber)**
```json
{
  "primary_trade": "Plumber",
  "specific_tasks": ["Fixing pipes, Cutting materials"],
  "quantified_output": "10 joints welded, 2 pipes cut",
  "universal_efficiency_score": 96,
  "performance_summary": "The plumber demonstrates high physical exertion and efficiency in completing the tasks at hand."
}
```
<!-- slide -->
**Sample AI Output — `11_prep_masonry` (Anomaly)**
```json
{
  "primary_trade": "Construction Worker",
  "specific_tasks": ["Lifting heavy materials, operating machinery"],
  "quantified_output": "100% of the video",
  "universal_efficiency_score": 7,
  "performance_summary": "The worker appears focused but output was difficult to quantify."
}
```
⚠️ 100% OpenCV productivity, but UES = 7 — movement without proportional output detected.
````

---

## Dashboard Upgrade

Enhanced `dashboard.py` with:
- **5 KPI metrics** bar at top
- **Altair grouped bar chart** — Productivity % vs AI UES per video
- **Altair peak exertion chart** by video
- **Donut chart** — Trade breakdown (11 Construction Workers, 3 Plumbers)
- **Leaderboard** with ProgressColumn render
- **Per-video drill-down** with AI JSON expander + exertion plot + annotated video player

```bash
streamlit run dashboard.py
# Running at: http://localhost:8501
```

---

## GitHub Push & Deployment

```bash
git add . && git commit -m "feat: Ollama LLaVA batch results + enhanced dashboard"
git push origin main   # ✅ 53 files, 7.2 MB
```

**Repo:** [github.com/sujeetmadihalli/Hackathon_BuilderBobs](https://github.com/sujeetmadihalli/Hackathon_BuilderBobs)

### Deploy to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. **New app** → `sujeetmadihalli/Hackathon_BuilderBobs` / `main` / `dashboard.py`
3. Click **Deploy** — `requirements.txt` handles all dependencies automatically

---

## Validation Checklist

- [x] All 14 `Agent_Analysis_*.json` files present in `outputs/`
- [x] `master_dashboard.csv` has AI columns: `AI_Trade`, `AI_Tasks`, `AI_Output`, `AI_UES`, `AI_Summary`
- [x] Streamlit dashboard running locally at `localhost:8501`
- [x] All files pushed to GitHub (`main` branch, commit `b471215`)
- [x] `requirements.txt` present for cloud deployment
