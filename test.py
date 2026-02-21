import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8n-pose model
model = YOLO('yolov8n-pose.pt')

# 2. Setup Video I/O
input_video_path = '08_prep_production_mp.mp4' # Replace with your video file
output_video_path = 'annotated_output.mp4'


cap = cv2.VideoCapture(input_video_path)

# ADD THIS SANITY CHECK:
if not cap.isOpened():
    print(f"❌ ERROR: OpenCV cannot find or open '{input_video_path}'. Check the filename!")
    exit()

# Get video properties for the VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize the VideoWriter (mp4v codec is standard for mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize an array to store our time-series spatial data
exertion_data = []

print("Processing video... grab a coffee, this might take a minute.")

def enhance_frame_for_yolo(frame):
    # 1. Apply CLAHE to fix uneven construction lighting
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Create a CLAHE object (clipLimit controls the contrast threshold)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    # Merge the enhanced L channel back and convert to BGR
    limg = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 2. Apply a Sharpening Kernel to combat motion blur
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened_frame = cv2.filter2D(enhanced_frame, -1, kernel)
    
    return sharpened_frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # End of video

    # 3. Run YOLO pose inference
    results = model(enhance_frame_for_yolo(frame), verbose=False) 
    annotated_frame = results[0].plot()
    
    # 5. Build the Logic: SAFELY check if any people were detected
    # We check if keypoints exist AND if the list of detected people is greater than 0
    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        
        # Now it is safe to grab the first person's keypoints
        keypoints = results[0].keypoints.xy[0].cpu().numpy() 
        
        # Make sure we have enough keypoints (wrists are 9 and 10) 
        # and that they aren't just [0., 0.] (which YOLO returns if obscured)
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
    
    # 6. Write the frame out
    out.write(annotated_frame)
    
    cv2.imshow('YOLOv8 Pose Tracking', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done! Annotated video saved to {output_video_path}")
print(f"Extracted spatial data for {len(exertion_data)} frames.")

# You now have 'exertion_data', a list of dictionaries tracking wrist movement over time.


import pandas as pd
import numpy as np

# Convert your extracted list of dictionaries into a Pandas DataFrame
df = pd.DataFrame(exertion_data)

# 1. Calculate the pixel distance each wrist traveled between frames (Euclidean distance)
df['lw_dist'] = np.sqrt(df['lw_x'].diff()**2 + df['lw_y'].diff()**2)
df['rw_dist'] = np.sqrt(df['rw_x'].diff()**2 + df['rw_y'].diff()**2)

# Fill the first NaN row with 0
df.fillna(0, inplace=True)

# 2. Combine the movement of both hands into a single raw exertion score
df['raw_movement'] = df['lw_dist'] + df['rw_dist']

# 3. Smooth the noise: Calculate a rolling average over ~1 second (assuming 30fps)
# This creates a clean time-series line graph, smoothing out camera micro-jitters
window_size = 30 
df['smoothed_exertion'] = df['raw_movement'].rolling(window=window_size).mean()

# 4. Define an "Active Labor" threshold 
# If they move more than 10 pixels per frame on average, they are working.
# (You may need to tweak this number based on your video's resolution)
active_threshold = 10.0 
df['is_active'] = df['smoothed_exertion'] > active_threshold

# 5. Generate the Final Quantifiable Metrics for the judges
total_frames = len(df)
active_frames = df['is_active'].sum()

# Metric 1: Productivity Percentage (Amount of Time Actually Working)
productivity_pct = (active_frames / total_frames) * 100 if total_frames > 0 else 0

# Metric 2: Peak Physical Intensity (Maximum effort output)
peak_intensity = df['smoothed_exertion'].max()

print(f"Productivity Score: {productivity_pct:.1f}%")
print(f"Peak Exertion Metric: {peak_intensity:.2f}")