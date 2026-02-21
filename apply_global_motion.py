import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger

INPUT_DIR = 'IronsiteHackathonData/'
OUTPUT_DIR = 'outputs/'
MASTER_CSV = 'master_dashboard.csv'
PROCESS_FPS = 5

global_movement_threshold = 2.0  # Mean pixel difference required to be "Action"

def process_motion_for_video(video_path, base_name):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, int(fps / PROCESS_FPS))
    
    prev_frame = None
    frame_count = 0
    analyzed_frames = 0
    motion_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        if frame_count % frame_skip != 0: continue
            
        analyzed_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        score = 0
        if prev_frame is not None:
            score = np.mean(cv2.absdiff(prev_frame, gray))
            
        motion_data.append({"frame": analyzed_frames, "motion_score": score})
        prev_frame = gray

    cap.release()
    
    # --- Map this back to the existing CSV data ---
    csv_path = os.path.join(OUTPUT_DIR, f"{base_name}_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # We need to map our new motion_score back into the DataFrame
        # The lengths should match exactly since we used the same frame_skip logic
        min_len = min(len(df), len(motion_data))
        df = df.iloc[:min_len].copy()
        df['smoothed_exertion'] = [m['motion_score'] for m in motion_data[:min_len]]
        
        # Smooth the global motion to remove micro-jitters
        df['smoothed_exertion'] = df['smoothed_exertion'].rolling(window=PROCESS_FPS, min_periods=1).mean()
        
        # WORKER IS ACTIVE IF THE CAMERA/BODY IS SHAKING
        df['is_working'] = df['smoothed_exertion'] > global_movement_threshold
        
        # Find dominant task from YOLO to keep the "Task Name"
        objects_detected = df[df['objects_detected'] > 0]['objects_list'].dropna()
        all_tools = [t.strip() for tops in objects_detected for t in str(tops).split(',') if t.strip()]
        
        task = "Manual Labor"
        if all_tools:
            from collections import Counter
            task = f"Handling {Counter(all_tools).most_common(1)[0][0]}".title()
            
        df.to_csv(csv_path, index=False)
        
        # Output Plot
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['frame'], df['smoothed_exertion'], color='cyan', label='Body Exertion (Camera Motion)', linewidth=1.5)
        
        in_active = False
        start_f = 0
        for i, row in df.iterrows():
            if row['is_working'] and not in_active:
                start_f = row['frame']
                in_active = True
            elif not row['is_working'] and in_active:
                ax.axvspan(start_f, row['frame'], color='green', alpha=0.3)
                in_active = False
        if in_active:
            ax.axvspan(start_f, df.iloc[-1]['frame'], color='green', alpha=0.3)
            
        ax.set_title(f"Worker Exertion Pipeline: {base_name}", fontsize=16, pad=20)
        ax.set_xlabel("Frame Number (Time)", fontsize=12)
        ax.set_ylabel("Global Motion Intensity", fontsize=12)
        
        patch = mpatches.Patch(color='green', alpha=0.3, label='Active Work Detected')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(patch)
        ax.legend(handles=handles, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_plot.png"), dpi=300)
        plt.close()

        prod = (df['is_working'].sum() / len(df)) * 100
        logger.info(f"{base_name} mapped. New Prod: {prod:.1f}%")
        
        return {
            "Video": base_name,
            "Total Frames": len(df),
            "Working Frames": int(df['is_working'].sum()),
            "Productivity %": round(prod, 1),
            "Peak Exertion (px)": round(df['smoothed_exertion'].max(), 2),
            "Detected Task": task
        }

def run_all():
    videos = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')])
    metrics = []
    
    for v in videos:
        base = v.replace('.mp4', '')
        res = process_motion_for_video(os.path.join(INPUT_DIR, v), base)
        if res: metrics.append(res)
        
    pd.DataFrame(metrics).to_csv(MASTER_CSV, index=False)
    logger.success("All videos updated with Global Motion tracking!")

if __name__ == "__main__":
    run_all()
