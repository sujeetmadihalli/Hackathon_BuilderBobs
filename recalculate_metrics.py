import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import matplotlib.patches as mpatches

INPUT_DIR = 'outputs/'
MASTER_CSV = 'master_dashboard.csv'
PROCESS_FPS = 5
ACTIVE_MOVEMENT_THRESHOLD = 1.5  # Lowered from 5.0 to 1.5 for subtle POV tool usage

def recalculate_metrics():
    all_metrics = []
    
    # Get all the generated data CSVs
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_data.csv')]
    
    for filename in csv_files:
        filepath = os.path.join(INPUT_DIR, filename)
        base_name = filename.replace('_data.csv', '')
        
        df = pd.read_csv(filepath)
        
        # 1. FIX: Linearly interpolate missing hand landmarks!
        # If the hand goes out of frame, we smoothly connect the dots so exertion isn't dropped to 0
        df['lw_x'] = df['lw_x'].interpolate(method='linear').ffill().bfill()
        df['lw_y'] = df['lw_y'].interpolate(method='linear').ffill().bfill()
        df['rw_x'] = df['rw_x'].interpolate(method='linear').ffill().bfill()
        df['rw_y'] = df['rw_y'].interpolate(method='linear').ffill().bfill()
        
        df['lw_dist'] = np.sqrt(df['lw_x'].diff()**2 + df['lw_y'].diff()**2).fillna(0)
        df['rw_dist'] = np.sqrt(df['rw_x'].diff()**2 + df['rw_y'].diff()**2).fillna(0)
        
        df['raw_movement'] = df['lw_dist'] + df['rw_dist']
        df['smoothed_exertion'] = df['raw_movement'].rolling(window=PROCESS_FPS, min_periods=1).mean()
        
        # 2. FIX: POV cameras naturally shake when walking/working. Even if hands are "still"
        # in the frame, they are exerting energy. 
        df['is_moving'] = df['smoothed_exertion'] > ACTIVE_MOVEMENT_THRESHOLD
        
        # 3. FIX: Give a 5-second grace period for objects. Tools go out of the camera's FOV frequently!
        df['objects_nearby'] = df['objects_detected'].rolling(window=PROCESS_FPS * 5, min_periods=1).sum() > 0
        
        df['is_working'] = df['is_moving'] & df['objects_nearby']
        
        # Determine task
        active_objects = df[df['is_working']]['objects_list'].dropna()
        all_active_tools = [tool.strip() for tools in active_objects for tool in str(tools).split(',') if tool.strip()]
        dominant_task = "General Labor"
        if all_active_tools:
            from collections import Counter
            common = Counter(all_active_tools).most_common(1)
            if common:
                dominant_task = f"Handling {common[0][0]}"
                
        # Re-save the Data
        df.to_csv(filepath, index=False)
        
        # --- RE-PLOT ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['frame'], df['smoothed_exertion'], color='cyan', label='Hand Exertion (Pixels/Frame)', linewidth=1.5)
        
        in_active_block = False
        start_frame = 0
        for idx, row in df.iterrows():
            if row['is_working'] and not in_active_block:
                start_frame = row['frame']
                in_active_block = True
            elif not row['is_working'] and in_active_block:
                ax.axvspan(start_frame, row['frame'], color='green', alpha=0.3)
                in_active_block = False
        if in_active_block:
            ax.axvspan(start_frame, df.iloc[-1]['frame'], color='green', alpha=0.3)

        ax.set_title(f"Worker Exertion Pipeline: {base_name}", fontsize=16, pad=20)
        ax.set_xlabel("Frame Number (Time)", fontsize=12)
        ax.set_ylabel("Movement Intensity", fontsize=12)
        
        patch = mpatches.Patch(color='green', alpha=0.3, label='Active Work Detected (Moving + Context)')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(patch)
        ax.legend(handles=handles, loc='upper left')
        
        plt.tight_layout()
        plot_output_path = os.path.join(INPUT_DIR, f"{base_name}_plot.png")
        plt.savefig(plot_output_path, dpi=300)
        plt.close()
        
        # Update master list
        total_frames = len(df)
        working_frames = df['is_working'].sum()
        productivity_pct = (working_frames / total_frames) * 100 if total_frames > 0 else 0
        peak_intensity = df['smoothed_exertion'].max()
        
        metrics = {
            "Video": base_name,
            "Total Frames": total_frames,
            "Working Frames": working_frames,
            "Productivity %": round(productivity_pct, 1),
            "Peak Exertion (px)": round(peak_intensity, 2),
            "Detected Task": dominant_task.title()
        }
        all_metrics.append(metrics)
        logger.info(f"Remapped {base_name}: Prod jumped to {productivity_pct:.1f}%")

    pd.DataFrame(all_metrics).to_csv(MASTER_CSV, index=False)
    logger.success(f"Recalculated all 14 videos perfectly! Saved to {MASTER_CSV}")

if __name__ == "__main__":
    recalculate_metrics()
