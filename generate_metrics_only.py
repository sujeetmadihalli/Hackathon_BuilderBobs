import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import ast

df = pd.read_csv('exertion_data.csv')

# --- PLOT GENERATION ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))

# Plot smoothed exertion as a continuous line
ax.plot(df['frame'], df['smoothed_exertion'], color='cyan', label='Hand Exertion (Pixels/Frame)', linewidth=1.5)

# Highlight areas where the worker is actively 'working'
active_frames = df[df['is_working']]['frame']

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

ax.set_title("First-Person Worker Exertion Pipeline", fontsize=16, pad=20)
ax.set_xlabel("Frame Number (Time)", fontsize=12)
ax.set_ylabel("Movement Intensity", fontsize=12)

import matplotlib.patches as mpatches
patch = mpatches.Patch(color='green', alpha=0.3, label='Active Work Detected (Moving + Objects)')
handles, labels = ax.get_legend_handles_labels()
handles.append(patch)
ax.legend(handles=handles, loc='upper left')

plt.tight_layout()
plt.savefig('exertion_plot.png', dpi=300)
logger.success(f"Dashboard plot saved.")

# Print Final Summary
total_frames = len(df)
working_frames = df['is_working'].sum()
productivity_pct = (working_frames / total_frames) * 100 if total_frames > 0 else 0

print("\n" + "="*50)
print("📊 SUPERVISOR DASHBOARD SUMMARY 📊")
print("="*50)
print(f"Total Video Frames:       {total_frames}")
print(f"Total 'Working' Frames:   {working_frames}")
print(f"Overall Productivity Score: {productivity_pct:.1f}%")
print(f"Peak Physical Intensity:  {df['smoothed_exertion'].max():.2f} px/frame")
print("="*50 + "\n")
