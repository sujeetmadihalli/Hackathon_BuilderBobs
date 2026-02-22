import os
import time
from loguru import logger
import agent_video_analyzer

VIDEO_DIR = "IronsiteHackathonData/"

def main():
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    videos.sort()
    
    logger.info(f"Starting batch AI analysis on {len(videos)} videos...")
    
    for i, video in enumerate(videos):
        # We know 14_production_mp.mp4 was already processed
        if video == "14_production_mp.mp4":
            logger.info("Skipping 14_production_mp.mp4 as it was already completed.")
            continue
            
        video_path = os.path.join(VIDEO_DIR, video)
        logger.info(f"[{i+1}/{len(videos)}] Processing {video_path}...")
        try:
            agent_video_analyzer.analyze_video(video_path)
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            
        # Optional sleep to be careful with rate limits if needed
        # but the video processing takes ~1 min itself
        time.sleep(5)
        
    logger.success("Batch analysis complete!")

if __name__ == "__main__":
    main()
