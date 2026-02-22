import streamlit as st
import pandas as pd
import os
from PIL import Image

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ironsite Supervisor Dashboard",
    page_icon="🏗️",
    layout="wide"
)

MASTER_CSV = "master_dashboard.csv"
OUTPUT_DIR = "outputs/"

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
def load_data():
    if not os.path.exists(MASTER_CSV):
        return pd.DataFrame() # Return empty if not run yet
    return pd.read_csv(MASTER_CSV)

df = load_data()

# ---------------------------------------------------------
# DASHBOARD UI
# ---------------------------------------------------------
st.title("🏗️ Ironsite AI: Supervisor Dashboard")
st.markdown("Monitor body-camera productivity and task exertion across your workforce.")

if df.empty:
    st.warning(f"⚠️ No data found. Please run the `first_person_pipeline.py` script first to generate {MASTER_CSV}.")
    st.stop()

# --- TOP LEVEL SUMMARY STATS ---
st.header("Site Overview")

col1, col2, col3, col4 = st.columns(4)
overall_productivity = df['Productivity %'].mean()
peak_effort = df['Peak Exertion (px)'].max()
top_task = df['Detected Task'].mode()[0] if not df.empty else "N/A"

col1.metric("Total Videos Analyzed", f"{len(df)}")
col2.metric("Average Productivity", f"{overall_productivity:.1f}%")
col3.metric("Site Peak Intensity", f"{peak_effort:.1f} px")
col4.metric("Most Common Task", top_task)

st.divider()

# --- LEADERBOARD ---
st.subheader("📋 Worker Productivity Leaderboard")
st.markdown("Compare productivity scores across different shifts and workers.")

# Sort by productivity descending
sorted_df = df.sort_values(by="Productivity %", ascending=False).reset_index(drop=True)

# We can format the dataframe for a prettier display
st.dataframe(
    sorted_df,
    column_config={
        "Video": st.column_config.TextColumn("Source Footage"),
        "Detected Task": st.column_config.TextColumn("Primary Task"),
        "Productivity %": st.column_config.ProgressColumn(
            "Productivity Score",
            help="Percentage of time spent actively working",
            format="%f%%",
            min_value=0,
            max_value=100,
        ),
        "Peak Exertion (px)": st.column_config.NumberColumn("Peak Intensity"),
        "AI_Trade": st.column_config.TextColumn("AI Identified Trade"),
        "AI_UES": st.column_config.ProgressColumn(
            "Universal Efficiency Score",
            help="AI-generated synthetic score combining physical exertion with work output",
            format="%f",
            min_value=0,
            max_value=100,
        ),
    },
    column_order=[
        "Video", 
        "Detected Task", 
        "Productivity %", 
        "Peak Exertion (px)", 
        "AI_Trade", 
        "AI_UES"
    ],
    use_container_width=True,
    hide_index=True,
)

st.divider()

# --- DRILL DOWN VIEW ---
st.subheader("🔍 Deep Dive: Individual Review")

# Dropdown to select a specific video
video_list = sorted_df['Video'].tolist()
selected_video = st.selectbox("Select a video to view detailed exertion metrics:", video_list)

if selected_video:
    video_data = sorted_df[sorted_df['Video'] == selected_video].iloc[0]
    
    st.markdown(f"### Report: `{selected_video}`")
    
    # Show the specific metrics for this guy
    m1, m2, m3 = st.columns(3)
    m1.metric("Task Identified", video_data['Detected Task'])
    m2.metric("Productivity Score", f"{video_data['Productivity %']}%")
    m3.metric("Peak Exertion", f"{video_data['Peak Exertion (px)']} px")
    
    if 'AI_Trade' in video_data and pd.notna(video_data['AI_Trade']):
        st.markdown("---")
        st.markdown("### 🤖 Agent Qualitative Analysis")
        ai1, ai2 = st.columns(2)
        ai1.metric("Unified Efficiency Score (UES)", f"{video_data.get('AI_UES', 'N/A')}/100")
        ai2.metric("Primary Trade Identified", video_data.get('AI_Trade', 'N/A'))
        
        st.info(f"**Performance Summary:** {video_data.get('AI_Summary', 'N/A')}")
        st.write(f"**Specific Tasks Completed:** {video_data.get('AI_Tasks', 'N/A')}")
        st.write(f"**Quantified Output:** {video_data.get('AI_Output', 'N/A')}")
        st.markdown("---")
    
    # Load and display the generated matplotlib graph
    plot_path = os.path.join(OUTPUT_DIR, f"{selected_video}_plot.png")
    if os.path.exists(plot_path):
        st.image(Image.open(plot_path), caption=f"Time-Series Exertion for {selected_video}", use_container_width=True)
    else:
        st.error(f"Could not find plot image at {plot_path}")
    
    # Optional: Display the annotated video if needed
    video_path = os.path.join(OUTPUT_DIR, f"{selected_video}_annotated.mp4")
    with st.expander("▶️ Watch Annotated Highlight Reel"):
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.warning("Annotated video file not found.")
