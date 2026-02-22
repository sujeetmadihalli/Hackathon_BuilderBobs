import os
import sys
import time
import json
import base64
import tempfile
import subprocess
import requests
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Ollama connection via SSH tunnel ─────────────────────────────────────────
# Assumes: ssh -p 56834 root@171.248.243.88 -L 8080:localhost:8080
# The tunnel maps remote Ollama (on port 8080 on the server) → localhost:8080
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:8080")
VISION_MODEL    = os.getenv("OLLAMA_VISION_MODEL", "llava:latest")

MASTER_CSV = "master_dashboard.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int = 16) -> list[str]:
    """Extract up to `num_frames` evenly-spaced frames from *video_path*.
    Returns a list of temporary PNG file paths (caller must delete them)."""
    # Get duration via ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        duration = float(subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL).strip())
    except Exception:
        duration = 60.0  # fallback

    interval = max(duration / num_frames, 1.0)
    tmp_dir  = tempfile.mkdtemp(prefix="ollama_frames_")
    out_pattern = os.path.join(tmp_dir, "frame_%04d.png")

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps=1/{interval:.2f}",
        "-vframes", str(num_frames),
        "-q:v", "3",
        out_pattern,
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    frames = sorted(
        os.path.join(tmp_dir, f)
        for f in os.listdir(tmp_dir)
        if f.endswith(".png")
    )
    logger.info(f"Extracted {len(frames)} frames from {video_path} → {tmp_dir}")
    return frames


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ollama_generate(prompt: str, images_b64: list[str]) -> str:
    """Call the Ollama /api/generate endpoint with vision support."""
    payload = {
        "model":  VISION_MODEL,
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
        },
    }
    url = f"{OLLAMA_BASE_URL}/api/generate"
    logger.info(f"Sending request to {url} with {len(images_b64)} image(s)...")
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "")


def parse_json_from_response(text: str) -> dict:
    """Extract the first JSON object found in *text*."""
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    # Find the outermost {}
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start:end])


# ── Main entry ────────────────────────────────────────────────────────────────

def analyze_video(video_path: str, num_frames: int = 16):
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        return

    base_name = os.path.basename(video_path).replace(".mp4", "")

    # 1. Read existing OpenCV Metrics
    try:
        df = pd.read_csv(MASTER_CSV)
        video_row = df[df["Video"] == base_name]
        if video_row.empty:
            logger.warning(f"No OpenCV metrics found in {MASTER_CSV} for {base_name}.")
            cv_productivity  = "Unknown"
            cv_peak_exertion = "Unknown"
        else:
            cv_productivity  = video_row.iloc[0]["Productivity %"]
            cv_peak_exertion = video_row.iloc[0]["Peak Exertion (px)"]
    except Exception as e:
        logger.error(f"Failed to read {MASTER_CSV}: {e}")
        return

    logger.info(f"OpenCV metrics → Productivity: {cv_productivity}%, Peak Exertion: {cv_peak_exertion}px")

    # 2. Extract frames
    frames = []
    try:
        frames = extract_frames(video_path, num_frames=num_frames)

        # Encode frames to base64
        images_b64 = [encode_image_b64(f) for f in frames]

        # 3. Build prompt
        prompt = f"""You are analyzing frames extracted from first-person (POV) body-camera footage of a construction worker.

A computer vision pipeline already computed these quantitative metrics for the full video:
- Physical Productivity (Active Global Motion): {cv_productivity}% of the video
- Peak Physical Exertion: {cv_peak_exertion} pixels of frame-shake

Based on the provided frames, return ONLY a valid JSON object (no markdown, no extra text) with these exact keys:

{{
  "primary_trade": "String (e.g., Plumber, Mason, Electrician)",
  "specific_tasks": "String (short list of specific tasks observed)",
  "quantified_output": "String (quantify actions, e.g., '5 joints welded, 3 pipes cut')",
  "universal_efficiency_score": <Integer 1-100 judging how well physical exertion translated to actual work output>,
  "performance_summary": "String (2-sentence summary of work ethic and technique)"
}}

Return ONLY valid JSON. No markdown code fences."""

        # 4. Call Ollama
        raw_response = ollama_generate(prompt, images_b64)
        logger.debug(f"Raw Ollama response:\n{raw_response}")

        # 5. Parse JSON
        ai_data = parse_json_from_response(raw_response)
        logger.success(f"AI Analysis:\n{json.dumps(ai_data, indent=2)}")

        # 6. Save raw JSON
        os.makedirs("outputs", exist_ok=True)
        output_file = f"outputs/Agent_Analysis_{base_name}.json"
        with open(output_file, "w") as f:
            json.dump(ai_data, f, indent=4)
        logger.success(f"Saved → {output_file}")

        # 7. Update Master CSV
        if not video_row.empty:
            idx = df.index[df["Video"] == base_name].tolist()[0]
            df.at[idx, "AI_Trade"]   = ai_data.get("primary_trade", "Unknown")
            df.at[idx, "AI_Tasks"]   = ai_data.get("specific_tasks", "Unknown")
            df.at[idx, "AI_Output"]  = ai_data.get("quantified_output", "Unknown")
            df.at[idx, "AI_UES"]     = ai_data.get("universal_efficiency_score", 0)
            df.at[idx, "AI_Summary"] = ai_data.get("performance_summary", "Unknown")
            df.to_csv(MASTER_CSV, index=False)
            logger.success(f"Updated {MASTER_CSV} for {base_name}")

    except Exception as e:
        logger.error(f"Analysis failed for {video_path}: {e}")
        raise
    finally:
        # Cleanup temp frames
        for f in frames:
            try:
                os.remove(f)
                os.rmdir(os.path.dirname(f))
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_to_analyze = sys.argv[1]
    else:
        video_to_analyze = "IronsiteHackathonData/14_production_mp.mp4"

    analyze_video(video_to_analyze)
