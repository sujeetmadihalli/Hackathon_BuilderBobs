import os
import pandas as pd
import google.generativeai as genai
from loguru import logger

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MASTER_CSV = 'master_dashboard.csv'
REPORT_OUTPUT = 'outputs/Final_AI_Site_Report.txt'

# TODO: Add your Gemini API Key here
# Get a free key at: https://aistudio.google.com/
API_KEY = "[ENCRYPTION_KEY]"
genai.configure(api_key=API_KEY)

def generate_site_report(df):
    logger.info("Preparing data for the LLM...")
    
    # We use Gemini 2.5 Flash because it is highly capable for text/data tasks and very fast
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # 1. Calculate some site-wide aggregates from your exact data
    total_videos = len(df)
    avg_productivity = round(df['Productivity %'].mean(), 1)
    max_exertion_val = df['Peak Exertion (px)'].max()
    max_exertion_video = df.loc[df['Peak Exertion (px)'].idxmax()]['Video']
    common_tasks = df['Detected Task'].value_counts().index.tolist()
    
    # 2. Convert the dataframe to a string format the LLM can easily read
    raw_data_string = df.to_string(index=False)
    
    # 3. Build the Prompt
    prompt = f"""
    You are an AI Site Supervisor for a construction company. 
    I have run a computer vision pipeline on {total_videos} video feeds across the site today. 
    
    Here is the exact numerical data extracted:
    {raw_data_string}
    
    Site-Wide Aggregates:
    - Average Site Productivity: {avg_productivity}%
    - Highest Peak Exertion: {max_exertion_val} pixels/frame (Observed in video: {max_exertion_video})
    - Most common tasks detected: {', '.join(common_tasks)}
    
    Based ONLY on the data provided above, write a comprehensive but concise 4-part report for the site manager:
    
    1. Executive Summary: A brief 2-sentence overview of site activity and overall productivity.
    2. Productivity Standouts: Identify which video/worker had the highest productivity and what task they were doing. 
    3. Ergonomic & Safety Warnings: Identify the worker with the highest peak exertion. Explain why this specific task might cause high exertion and recommend a safety check.
    4. Task Distribution: Briefly summarize the main types of tools/tasks being performed across the site.
    
    Format the output cleanly with bold headings. Maintain a professional, analytical tone.
    """
    
    logger.info("Sending data to Gemini for analysis...")
    
    try:
        # 4. Generate the response
        response = model.generate_content(prompt)
        report_text = response.text
        
        # 5. Save the report
        os.makedirs('outputs', exist_ok=True)
        with open(REPORT_OUTPUT, "w") as f:
            f.write(report_text)
            
        logger.success(f"Successfully generated AI Report! Saved to {REPORT_OUTPUT}")
        print("\n" + "="*50)
        print("FINAL AI SITE REPORT")
        print("="*50)
        print(report_text)
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to generate report. Error: {e}")

def main():
    if not os.path.exists(MASTER_CSV):
        logger.error(f"Could not find {MASTER_CSV}. Please ensure your previous script generated it.")
        return
        
    # Read the existing data
    df = pd.read_csv(MASTER_CSV)
    
    if df.empty:
        logger.warning(f"The file {MASTER_CSV} is empty.")
        return
        
    generate_site_report(df)

if __name__ == "__main__":
    main()
