import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO
import matplotlib.pyplot as plt
from loguru import logger

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
INPUT_DIR = 'IronsiteHackathonData/'
OUTPUT_DIR = 'outputs/'
MASTER_CSV = 'master_dashboard.csv'

import os 
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Hackathon Demo Settings
# Instead of cutting the video off, we downsample the framerate.
# How many frames per second should the AI actually analyze?
# Set to 5 FPS to make it 6x faster than a 30fps video while still capturing motion.
PROCESS_FPS = 5

# Exertion Thresholds 
ACTIVE_MOVEMENT_THRESHOLD = 5.0  # Min pixels moved per frame to count as "active"
ROLLING_WINDOW_FRAMES = 30       # ~1 second of video at 30fps

# ---------------------------------------------------------
# MODEL INITIALIZATION
# ---------------------------------------------------------
# 1. MediaPipe Hand Tracking 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2,
                                       min_hand_detection_confidence=0.5,
                                       min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

# 2. YOLOv8 Object Detection (Custom Construction Model)
# We swap the generic yolov8n for one trained on construction sites!
try:
    yolo_model = YOLO('yolov8n-construction.pt') 
    logger.info("Loaded CUSTOM YOLOv8 Construction Object model.")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    sys.exit(1)


def process_video(input_video_path):
    video_filename = os.path.basename(input_video_path)
    base_name = os.path.splitext(video_filename)[0]
    
    output_video_path = os.path.join(OUTPUT_DIR, f"{base_name}_annotated.mp4")
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {input_video_path}")
        return None

    # Video Properties for Exporter
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use 'avc1' (h264) so Streamlit/HTML5 can play the video natively!
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    # We will write the output video at the *desired* process FPS 
    # so the annotated playback looks normal (just choppy)
    out = cv2.VideoWriter(output_video_path, fourcc, PROCESS_FPS, (width, height))

    exertion_data = []
    
    # Calculate how many frames to skip 
    frame_skip = max(1, int(fps / PROCESS_FPS))
    
    logger.info(f"Video is {fps} FPS. Running AI at {PROCESS_FPS} FPS (Skipping every {frame_skip} frames).")
    
    frame_count = 0
    analyzed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # DOWN-SAMPLING LOGIC: Only run AI on the target frames
        if frame_count % frame_skip != 0:
            continue
            
        analyzed_frames += 1
        current_frame_data = {
            "frame": analyzed_frames, # We log the index of the analyzed frame (1, 2, 3...)
            "lw_x": np.nan, "lw_y": np.nan,  # Left Wrist Position
            "rw_x": np.nan, "rw_y": np.nan,  # Right Wrist Position
            "objects_detected": 0,
            "objects_list": ""               # What are they holding?
        }

        # --- 1. YOLO INFERENCE (Object Detection) ---
        # Detect what is in the frame (tools, brick, etc)
        # We run verbose=False to keep the console clean
        yolo_results = yolo_model(frame, verbose=False)[0]
        
        # Draw bounding boxes
        annotated_frame = yolo_results.plot() 
        current_frame_data["objects_detected"] = len(yolo_results.boxes)
        
        # Log the specific classes detected (e.g., 'hard-hat', 'tool', etc)
        detected_classes = [yolo_model.names[int(cls)] for cls in yolo_results.boxes.cls]
        current_frame_data["objects_list"] = ", ".join(detected_classes)

        # --- 2. MEDIAPIPE INFERENCE (Hand Tracking) ---
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = detector.detect(mp_image)
        
        # --- 3. DATA EXTRACTION & ANNOTATION ---
        if results.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                # Manually draw landmarks with OpenCV to avoid legacy framework import issues
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                
                # Determine handedness (Left/Right)
                handedness = results.handedness[hand_idx][0].category_name
                
                # Extract Wrist coordinates (Landmark 0)
                wrist = hand_landmarks[0]
                px_x, px_y = int(wrist.x * width), int(wrist.y * height)
                
                if handedness == "Left":
                    current_frame_data["lw_x"] = px_x
                    current_frame_data["lw_y"] = px_y
                else:
                    current_frame_data["rw_x"] = px_x
                    current_frame_data["rw_y"] = px_y
                    
        exertion_data.append(current_frame_data)
        out.write(annotated_frame)
        
        # Progress indicator (Update every 100 analyzed frames)
        if analyzed_frames % 100 == 0:
            logger.info(f"Processed {analyzed_frames} sampled frames...")

    # Clean up
    cap.release()
    out.release()
    
    logger.success(f"Video processing complete. Saved to {output_video_path}")
    
    # --- 4. DATA ANALYSIS & VISUALIZATION ---
    return calculate_and_plot_metrics(exertion_data, base_name)


def calculate_and_plot_metrics(data, base_name):
    csv_output_path = os.path.join(OUTPUT_DIR, f"{base_name}_data.csv")
    plot_output_path = os.path.join(OUTPUT_DIR, f"{base_name}_plot.png")
    
    logger.info("Calculating exertion metrics...")
    
    df = pd.DataFrame(data)
    
    # 1. Calculate Pixel Distance Traveled (Euclidean)
    # Forward fill NaNs so distance is 0 when hands briefly disappear
    df['lw_x'] = df['lw_x'].ffill()
    df['lw_y'] = df['lw_y'].ffill()
    df['rw_x'] = df['rw_x'].ffill()
    df['rw_y'] = df['rw_y'].ffill()
    
    df['lw_dist'] = np.sqrt(df['lw_x'].diff()**2 + df['lw_y'].diff()**2).fillna(0)
    df['rw_dist'] = np.sqrt(df['rw_x'].diff()**2 + df['rw_y'].diff()**2).fillna(0)
    
    # 2. Raw Movement = Total physical hand displacement
    df['raw_movement'] = df['lw_dist'] + df['rw_dist']
    
    # 3. Smooth the data (Rolling Average) to remove micro-jitters
    # Since we are downsampling (e.g., 5 fps), 1 second is only 5 frames!
    df['smoothed_exertion'] = df['raw_movement'].rolling(window=PROCESS_FPS, min_periods=1).mean()
    
    # 4. Activity Classification
    # Worker is 'active' if hands are moving AND they are interacting with an object
    df['is_moving'] = df['smoothed_exertion'] > ACTIVE_MOVEMENT_THRESHOLD
    df['is_working'] = df['is_moving'] & (df['objects_detected'] > 0)
    
    # NEW: Determine the dominant task based on the most common object held during active periods
    active_objects = df[df['is_working']]['objects_list'].dropna()
    # Flatten the comma-separated lists and remove empty strings
    all_active_tools = [tool.strip() for tools in active_objects for tool in str(tools).split(',') if tool.strip()]
    dominant_task = "Unknown Task (No Tools)"
    if all_active_tools:
        # Find the most common tool class
        from collections import Counter
        dominant_task = f"Handling {Counter(all_active_tools).most_common(1)[0][0]}"
    
    # Save raw data
    df.to_csv(csv_output_path, index=False)
    logger.success(f"Metrics saved to {csv_output_path}")
    
    # --- PLOT GENERATION ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot smoothed exertion as a continuous line
    ax.plot(df['frame'], df['smoothed_exertion'], color='cyan', label='Hand Exertion (Pixels/Frame)', linewidth=1.5)
    
    # Highlight areas where the worker is actively 'working'
    # Find contiguous blocks of 'is_working' == True
    active_frames = df[df['is_working']]['frame']
    
    # Instead of plotting individual dots, let's shade the regions
    in_active_block = False
    start_frame = 0
    for idx, row in df.iterrows():
        if row['is_working'] and not in_active_block:
            start_frame = row['frame']
            in_active_block = True
        elif not row['is_working'] and in_active_block:
            ax.axvspan(start_frame, row['frame'], color='green', alpha=0.3)
            in_active_block = False
            
    # Catch a trailing block
    if in_active_block:
        ax.axvspan(start_frame, df.iloc[-1]['frame'], color='green', alpha=0.3)

    # Plot formatting
    ax.set_title(f"Worker Exertion Pipeline: {base_name}", fontsize=16, pad=20)
    ax.set_xlabel("Frame Number (Time)", fontsize=12)
    ax.set_ylabel("Movement Intensity", fontsize=12)
    
    # Custom legend for the shaded region
    import matplotlib.patches as mpatches
    patch = mpatches.Patch(color='green', alpha=0.3, label='Active Work Detected (Moving + Objects)')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(patch)
    ax.legend(handles=handles, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=300)
    logger.success(f"Dashboard plot saved to {plot_output_path}")
    plt.close() # Important to avoid memory leaks across multiple videos
    
    # Print Final Summary
    total_frames = len(df)
    working_frames = df['is_working'].sum()
    productivity_pct = (working_frames / total_frames) * 100 if total_frames > 0 else 0
    peak_intensity = df['smoothed_exertion'].max()
    
    # Return metrics for the master dashboard list
    return {
        "Video": base_name,
        "Total Frames": total_frames,
        "Working Frames": working_frames,
        "Productivity %": round(productivity_pct, 1),
        "Peak Exertion (px)": round(peak_intensity, 2),
        "Detected Task": dominant_task.title()
    }


def main():
    logger.info(f"Starting Multi-Video Batch Processing in {INPUT_DIR}")
    
    # Find all mp4 files
    mp4_files = []
    if os.path.exists(INPUT_DIR):
        mp4_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
    
    if not mp4_files:
        logger.error(f"No .mp4 files found in {INPUT_DIR}")
        sys.exit(1)
        
    logger.info(f"Found {len(mp4_files)} videos to process.")
    
    all_metrics = []
    
    # Process each video
    for idx, filename in enumerate(mp4_files):
        logger.info(f"--- Processing Video {idx+1}/{len(mp4_files)}: {filename} ---")
        filepath = os.path.join(INPUT_DIR, filename)
        
        metrics = process_video(filepath)
        if metrics:
            all_metrics.append(metrics)
            
            # Immediately save the master CSV after every video so we don't lose progress if it crashes
            pd.DataFrame(all_metrics).to_csv(MASTER_CSV, index=False)
            logger.info(f"Updated {MASTER_CSV}")

    detector.close()
    logger.success(f"Batch processing complete! All 14 videos analyzed. Master dashboard ready at {MASTER_CSV}")

if __name__ == "__main__":
    main()
