import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# 1. Load the pre-trained model
model = YOLO('yolov8n-pose.pt')

# 2. Setup Video I/O
input_video_path = '08_prep_production_mp.mp4' # Ensure this is your correct file name
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"❌ ERROR: Cannot find '{input_video_path}'")
    exit()

exertion_data = []

print("Extracting spatial data... (Running headless for maximum speed)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run YOLO inference (Note: skipping enhance_frame_for_yolo here for pure speed, 
    # but add it back inside model() if you need the accuracy bump!)
    results = model(frame, verbose=False) 
    
    # 4. Safely extract keypoints
    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy() 
        
        # Check for wrists (Indices 9 and 10)
        if len(keypoints) >= 11 and keypoints[9][0] != 0.0 and keypoints[10][0] != 0.0: 
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            
            frame_data = {
                "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "lw_x": float(left_wrist[0]),
                "lw_y": float(left_wrist[1]),
                "rw_x": float(right_wrist[0]),
                "rw_y": float(right_wrist[1])
            }
            exertion_data.append(frame_data)

cap.release()

# ---------------------------------------------------------
# PANDAS EXERTION QUANTIFICATION
# ---------------------------------------------------------

if len(exertion_data) == 0:
    print("❌ No spatial data extracted. The POV trap caught us (0 full bodies detected).")
else:
    print(f"✅ Data extracted for {len(exertion_data)} frames. Calculating exertion metrics...")
    
    df = pd.DataFrame(exertion_data)

    # 1. Calculate pixel distance traveled between frames
    df['lw_dist'] = np.sqrt(df['lw_x'].diff()**2 + df['lw_y'].diff()**2)
    df['rw_dist'] = np.sqrt(df['rw_x'].diff()**2 + df['rw_y'].diff()**2)
    df.fillna(0, inplace=True)

    # 2. Combine into a single raw score
    df['raw_movement'] = df['lw_dist'] + df['rw_dist']

    

    # 3. Smooth the data (rolling average over 30 frames / ~1 second)
    window_size = 30 
    df['smoothed_exertion'] = df['raw_movement'].rolling(window=window_size).mean()

    # 4. Active labor threshold (Tune this number if needed)
    active_threshold = 10.0 
    df['is_active'] = df['smoothed_exertion'] > active_threshold

    # 5. Final Metrics
    total_frames = len(df)
    active_frames = df['is_active'].sum()
    productivity_pct = (active_frames / total_frames) * 100 if total_frames > 0 else 0
    peak_intensity = df['smoothed_exertion'].max()

    print("\n--- 📊 SUPERVISOR DASHBOARD METRICS ---")
    print(f"Total Productivity Time:  {productivity_pct:.1f}%")
    print(f"Peak Exertion Intensity:  {peak_intensity:.2f} pixels/frame")