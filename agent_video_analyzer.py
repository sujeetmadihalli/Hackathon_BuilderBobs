import os
import sys
import time
import google.generativeai as genai
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Get a free key at: https://aistudio.google.com/
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=API_KEY)

def analyze_video(video_path):
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        return

    logger.info(f"Uploading {video_path} to Gemini...")
    video_file = genai.upload_file(path=video_path)
    
    logger.info(f"File uploaded as {video_file.name}. Waiting for processing...")
    while video_file.state.name == "PROCESSING":
        logger.info("Processing...")
        time.sleep(10)
        video_file = genai.get_file(video_file.name)
        
    if video_file.state.name == "FAILED":
        logger.error("Video processing failed.")
        return
        
    logger.info("Video processing complete! Running Agent Analysis...")
    
    prompt = """
    Analyze this first-person (POV) body camera footage of a construction worker. 
    Please watch the entire video and provide a detailed report addressing the following:
    
    1. **What the work was:** Describe the specific tasks and actions being performed in detail.
    2. **How much of the work did he do:** Quantify the actions if possible (e.g., number of repetitions, volume of material moved, sustained effort over time).
    3. **How does he compare with other workers:** Based on the observed pace, continuity of work vs. resting, and general efficiency, extrapolate how his productivity compares to a standard baseline or other workers doing similar work.
    4. **What is his work:** Identify his likely specific role or trade (e.g., mason, laborer, electrician) based on the tools and environment.
    
    Format the output cleanly with bold headings and be as observant as possible about the worker's manual labor and tool usage.
    """
    
    # We use Gemini 2.5 Flash for complex video understanding and reasoning on the free tier
    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
    
    try:
        response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
        print("\n" + "="*50)
        print("AGENT VIDEO ANALYSIS REPORT")
        print("="*50)
        print(response.text)
        print("="*50 + "\n")
        
        # Save output for reference
        output_file = f"outputs/Agent_Analysis_{os.path.basename(video_path)}.txt"
        with open(output_file, "w") as f:
            f.write(response.text)
        logger.success(f"Report saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate analysis: {e}")
        
    finally:
        logger.info(f"Cleaning up uploaded file {video_file.name}...")
        genai.delete_file(video_file.name)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_to_analyze = sys.argv[1]
    else:
        video_to_analyze = "IronsiteHackathonData/14_production_mp.mp4"
        
    analyze_video(video_to_analyze)
