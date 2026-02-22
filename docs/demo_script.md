# 🎬 IronSite BuilderBobs — Technical Video Demo Script

**Target length:** 5–7 minutes  
**Format:** Screen recording (terminal + browser) with voiceover  
**Tone:** Technical, fast-paced, confident

---

## Pre-Demo Setup Checklist

Before hitting record:
- [ ] Terminal open at `/mnt/d/Projects/Hackathon21Feb20226`
- [ ] Streamlit dashboard running: `streamlit run dashboard.py` → `localhost:8501`
- [ ] SSH tunnel active: `ssh -p 56834 root@171.248.243.88 -L 8080:localhost:11434`
- [ ] `agent_video_analyzer.py` open in your editor
- [ ] `first_person_pipeline.py` open in a second editor tab
- [ ] Browser on `localhost:8501`, scrolled to top
- [ ] Font size **increased** in terminal and editor (≥16pt) for readability

---

## SECTION 1 — Cold Open (0:00–0:30)

**🖥️ SHOW:** Terminal. Run this live:
```bash
curl -s http://localhost:11434/api/tags | python3 -c \
  "import sys,json; [print(m['name'], m['details']['parameter_size']) \
   for m in json.load(sys.stdin)['models']]"
```

**Expected output:**
```
llava:latest 7B
llama3:latest 8.0B
qwen3:14b 14.8B
```

**🎙️ SAY:**
> "We're starting on a remote Vast.ai GPU instance, tunnelled over SSH. You can see three models loaded in Ollama — we're using `llava:latest`, a 7-billion-parameter multimodal model that can see and reason about images. This is the AI brain of our pipeline. Zero cloud API calls. Zero rate limits. Eight seconds per video."

---

## SECTION 2 — The Problem & Architecture (0:30–1:15)

**🖥️ SHOW:** Open `first_person_pipeline.py` in your editor. Highlight lines 25–30 (the constants block).

```python
PROCESS_FPS = 5
ACTIVE_MOVEMENT_THRESHOLD = 5.0  # Min pixels moved per frame
ROLLING_WINDOW_FRAMES = 30       # ~1 second of video at 30fps
```

**🎙️ SAY:**
> "Construction workers already wear body cameras for safety. That footage is sitting on a server, unwatched. Our system turns it into productivity intelligence. 
>
> Stage 1 is pure computer vision — we run at 5 frames per second, a 6× speedup over raw 30fps. That's enough temporal resolution to capture hand movement without burning compute."

**🖥️ SHOW:** Scroll to lines 185–198. Highlight the exertion calculation and `is_working` line:

```python
df['lw_dist'] = np.sqrt(df['lw_x'].diff()**2 + df['lw_y'].diff()**2)
df['rw_dist'] = np.sqrt(df['rw_x'].diff()**2 + df['rw_y'].diff()**2)
df['raw_movement'] = df['lw_dist'] + df['rw_dist']
df['smoothed_exertion'] = df['raw_movement'].rolling(window=PROCESS_FPS).mean()
df['is_working'] = (df['smoothed_exertion'] > ACTIVE_MOVEMENT_THRESHOLD) \
                   & (df['objects_detected'] > 0)
```

**🎙️ SAY:**
> "The motion metric is Euclidean pixel distance — we track the left and right wrist separately using MediaPipe HandLandmarker, compute displacement frame-over-frame, then apply a rolling mean to kill micro-jitter. The critical gate is the AND condition on line 198: a worker is only classified as active if they're both *moving* AND *interacting with a detected object*. Walking across the site doesn't count. Hands moving with a tool in frame counts."

---

## SECTION 3 — Live Stage 2 Demo (1:15–2:30)

**🖥️ SHOW:** Open `agent_video_analyzer.py`. Highlight the `extract_frames` function (lines ~35–60).

```python
def extract_frames(video_path: str, num_frames: int = 16) -> list[str]:
    interval = max(duration / num_frames, 1.0)
    ffmpeg_cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps=1/{interval:.2f}",
        "-vframes", str(num_frames), ...
    ]
```

**🎙️ SAY:**
> "Stage 2 is where it gets interesting. We don't upload the whole video to a cloud API. We use `ffmpeg` to extract 16 frames, evenly spaced across the full duration. For a 3-minute video, that's one frame every 11 seconds — enough visual context for a language model to understand what trade this worker is in and what they accomplished."

**🖥️ SHOW:** Highlight the `ollama_generate` function and the prompt construction:

```python
def ollama_generate(prompt: str, images_b64: list[str]) -> str:
    payload = {
        "model":  VISION_MODEL,
        "prompt": prompt,
        "images": images_b64,   # All 16 frames at once
        "stream": False,
        "options": {"temperature": 0.1},
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=300)
```

**🎙️ SAY:**
> "We encode all 16 frames as base64 and send them in a single POST to Ollama's generate endpoint. Temperature is set to 0.1 — we want deterministic, low-hallucination answers. The prompt itself is grounded in the Stage 1 numbers."

**🖥️ SHOW:** Switch to terminal. Run a live analysis:
```bash
python3 agent_video_analyzer.py IronsiteHackathonData/05_production_mp.mp4
```

**🎙️ SAY (while it runs):**
> "Watch the timing. Frame extraction takes about 3 seconds. The LLaVA inference — 16 images through a 7B parameter model — takes another 5 to 7 seconds on the GPU."

**Expected terminal output to see:**
```
INFO  | OpenCV metrics → Productivity: 96.4%, Peak Exertion: 61.71px
INFO  | Extracted 16 frames → /tmp/ollama_frames_xxxxx
INFO  | Sending request to http://localhost:11434/api/generate with 16 image(s)...
SUCCESS | AI Analysis:
{
  "primary_trade": "Plumber",
  "specific_tasks": ["Fixing pipes, Cutting materials"],
  "quantified_output": "10 joints welded, 2 pipes cut",
  "universal_efficiency_score": 96,
  "performance_summary": "The plumber demonstrates high physical exertion..."
}
SUCCESS | Saved → outputs/Agent_Analysis_05_production_mp.json
SUCCESS | Updated master_dashboard.csv for 05_production_mp
```

**🎙️ SAY:**
> "Eight seconds. The model correctly identified this as a **Plumber**, enumerated specific tasks, quantified discrete outputs, and assigned a **Universal Efficiency Score of 96 out of 100**. That score is the most novel part of what we built — it uses the Stage 1 exertion data as a calibration signal, then asks: did the physical effort translate to actual work?"

---

## SECTION 4 — The UES Formula (2:30–3:00)

**🖥️ SHOW:** Stay in terminal. Pull up the raw JSON:
```bash
cat outputs/Agent_Analysis_05_production_mp.json | python3 -m json.tool
```

**🎙️ SAY:**
> "Every video produces a structured JSON. The UES is what we spend most time talking about with supervisors. Think of it as:
>
> If Productivity is *how hard did they work*, then UES is *how smart did they work*.
>
> A UES of 96 with 96% productivity means this plumber was maximally efficient — constant focused output. Compare that to our most interesting finding..."

---

## SECTION 5 — The Anomaly (3:00–3:45)

**🖥️ SHOW:** Terminal. Run:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('master_dashboard.csv')
print(df[['Video','Productivity %','AI_UES']].sort_values('AI_UES').to_string(index=False))
"
```

**🎙️ SAY:**
> "Sort by UES ascending. Look at the bottom."

**Point to the output row:**
```
11_prep_masonry    100.0    7.08
```

**🎙️ SAY:**
> "Video 11. One hundred percent OpenCV productivity — the highest achievable score. The worker was in constant motion the entire clip. But the UES is **7 out of 100**. 
>
> LLaVA looked at the frames and couldn't identify any discrete work output. The quantified output field came back as: '100% of the video' — which is what the model outputs when there's nothing concrete to enumerate. 
>
> This is a worker who is *moving a lot but accomplishing little*. That's an alert a supervisor doing a 5-minute floor check would never catch. Our system catches it automatically, every shift."

---

## SECTION 6 — Dashboard Walkthrough (3:45–5:30)

**🖥️ SHOW:** Switch to browser, `localhost:8501`. Start at the top.

**Point to 5 KPI metrics.**

**🎙️ SAY:**
> "The Streamlit dashboard surfaces both pipeline stages in one view. Five KPIs top-line the site: 14 videos, 91.9% average productivity, 83.84 peak exertion pixels, 85.4 average UES, 2 distinct trades identified."

**🖥️ SHOW:** Scroll to the **grouped bar chart** (Productivity % vs AI UES).

**🎙️ SAY:**
> "This grouped Altair chart is the core visual. Orange is Stage 1 — raw OpenCV motion productivity. Blue is Stage 2 — AI efficiency score. Where they're close together, the worker is converting exertion to output efficiently. Where they diverge — like Video 11 on the far left — that's where you investigate."

**🖥️ SHOW:** Scroll to the **peak exertion chart**.

**🎙️ SAY:**
> "Peak exertion in pixels per frame. Video 03 peaks at 83 pixels — that's a masonry worker we confirmed was operating a crane and laying brick. High exertion, justified by output. Video 12 — downtime prep clip — peaks at 42 pixels. You can see the signal is physically interpretable, not abstract."

**🖥️ SHOW:** Scroll to the **donut chart**.

**🎙️ SAY:**
> "Trade classification across the batch: 11 Construction Workers, 3 Plumbers. LLaVA identified this from raw frames with no labeling, no fine-tuning, zero training data specific to our videos."

**🖥️ SHOW:** Scroll to the **leaderboard table**. Point to the ProgressColumn bars.

**🎙️ SAY:**
> "The leaderboard ranks by productivity. The progress bars on both the productivity and UES columns let a supervisor scan for mismatches in about three seconds."

**🖥️ SHOW:** Use the **drill-down selector**. Select `05_production_mp`.

**🎙️ SAY:**
> "Individual drill-down. We pull the specific worker's metrics, the AI trade, UES, performance summary, and specific task list."

**🖥️ SHOW:** Click **"📄 View Raw AI JSON Output"** expander.

**🎙️ SAY:**
> "The raw JSON is embedded for any downstream system to consume — an ERP, a payroll verification system, a compliance audit trail."

**🖥️ SHOW:** Scroll to the **exertion plot image** (the matplotlib chart).

**🎙️ SAY:**
> "And the time-series exertion chart from Stage 1. Cyan line is hand movement intensity, green shaded regions are frames classified as active work. You can see the rhythm — continuous productive bursts with minimal idle time. That's a UES 96 worker."

---

## SECTION 7 — Architecture Close (5:30–6:00)

**🖥️ SHOW:** Split screen — terminal on left showing the `git log --oneline -5`, browser on right.

```bash
git log --oneline -5
```

**🎙️ SAY:**
> "The full pipeline is open on GitHub — `sujeetmadihalli/Hackathon_BuilderBobs`. It includes `requirements.txt` for one-click Streamlit Community Cloud deployment. The repo has the full 14-video analysis results committed, so the dashboard loads immediately with real data.
>
> Two-stage pipeline. 14 videos. 2 minutes of total AI processing time. One dashboard. Zero cloud API dependencies.
>
> That's BuilderBobs."

---

## 🎯 Key Lines to Hit

| Moment | The line |
|--------|----------|
| Opening | *"Zero cloud API calls. Zero rate limits. Eight seconds per video."* |
| AND condition | *"Walking across the site doesn't count. Hands moving with a tool counts."* |
| UES intro | *"Productivity is how hard they worked. UES is how smart they worked."* |
| Video 11 | *"One hundred percent productivity. Seven UES. A supervisor would never catch this."* |
| Close | *"Two-stage pipeline. 14 videos. 2 minutes. One dashboard. Zero cloud API dependencies."* |
