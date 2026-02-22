import streamlit as st
import pandas as pd
import os
import json
import altair as alt
from PIL import Image

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ironsite Supervisor Dashboard",
    page_icon="🏗️",
    layout="wide",
)

# Custom CSS for premium look
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #888; }
.block-container { padding-top: 2rem; }
h1 { letter-spacing: -1px; }
</style>
""", unsafe_allow_html=True)

MASTER_CSV = "master_dashboard.csv"
OUTPUT_DIR = "outputs/"

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
@st.cache_data(ttl=30)
def load_data():
    if not os.path.exists(MASTER_CSV):
        return pd.DataFrame()
    df = pd.read_csv(MASTER_CSV)
    # Clean AI_UES column — coerce any non-numeric partial values to NaN
    df["AI_UES"] = pd.to_numeric(df["AI_UES"], errors="coerce")
    return df

df = load_data()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🏗️ Ironsite AI — Supervisor Dashboard")
st.caption("Powered by Ollama LLaVA · OpenCV Global Motion Analysis · Real-time body-cam intelligence")

if df.empty:
    st.warning(f"⚠️ No data found. Run `first_person_pipeline.py` then `batch_agent_analysis.py`.")
    st.stop()

# ---------------------------------------------------------
# SITE OVERVIEW METRICS
# ---------------------------------------------------------
st.header("📊 Site Overview", divider="orange")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Videos Analyzed", len(df))
c2.metric("Avg Productivity",       f"{df['Productivity %'].mean():.1f}%")
c3.metric("Site Peak Intensity",    f"{df['Peak Exertion (px)'].max():.1f} px")
c4.metric("Avg AI Efficiency (UES)", f"{df['AI_UES'].mean():.1f}/100")
c5.metric("Trades Identified",      df["AI_Trade"].nunique())

st.divider()

# ---------------------------------------------------------
# CHARTS ROW
# ---------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("⚡ Productivity vs AI Efficiency")
    chart_df = df[["Video", "Productivity %", "AI_UES"]].dropna().melt(
        id_vars="Video", var_name="Metric", value_name="Score"
    )
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Video:N", sort="-y", axis=alt.Axis(labelAngle=-40, labelLimit=150)),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 110])),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(range=["#F97316", "#3B82F6"]),
            ),
            tooltip=["Video", "Metric", alt.Tooltip("Score:Q", format=".1f")],
            xOffset="Metric:N",
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

with col_right:
    st.subheader("🔨 Peak Exertion Intensity by Video")
    exertion_df = df[["Video", "Peak Exertion (px)"]].sort_values("Peak Exertion (px)", ascending=False)
    bar = (
        alt.Chart(exertion_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#F97316")
        .encode(
            x=alt.X("Video:N", sort="-y", axis=alt.Axis(labelAngle=-40, labelLimit=150)),
            y=alt.Y("Peak Exertion (px):Q"),
            tooltip=["Video", "Peak Exertion (px)"],
        )
        .properties(height=320)
    )
    st.altair_chart(bar, use_container_width=True)

# Trade breakdown pie
st.subheader("👷 Trade Breakdown")
trade_counts = df["AI_Trade"].value_counts().reset_index()
trade_counts.columns = ["Trade", "Count"]
pie = (
    alt.Chart(trade_counts)
    .mark_arc(innerRadius=60)
    .encode(
        theta="Count:Q",
        color=alt.Color("Trade:N", scale=alt.Scale(scheme="tableau10")),
        tooltip=["Trade", "Count"],
    )
    .properties(height=250)
)
st.altair_chart(pie, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# LEADERBOARD TABLE
# ---------------------------------------------------------
st.subheader("📋 Worker Productivity Leaderboard")

sorted_df = df.sort_values(by="Productivity %", ascending=False).reset_index(drop=True)

st.dataframe(
    sorted_df,
    column_config={
        "Video": st.column_config.TextColumn("Source Footage"),
        "Detected Task": st.column_config.TextColumn("Primary Task"),
        "Productivity %": st.column_config.ProgressColumn(
            "Productivity Score",
            help="% of time spent actively working",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
        "Peak Exertion (px)": st.column_config.NumberColumn("Peak Intensity (px)", format="%.1f"),
        "AI_Trade": st.column_config.TextColumn("AI Trade"),
        "AI_UES": st.column_config.ProgressColumn(
            "Universal Efficiency Score",
            help="AI-generated score: how well physical exertion translated to output",
            format="%.1f",
            min_value=0,
            max_value=100,
        ),
        "AI_Tasks": None,
        "AI_Output": None,
        "AI_Summary": None,
        "Total Frames": None,
        "Working Frames": None,
    },
    column_order=["Video", "Detected Task", "Productivity %", "Peak Exertion (px)", "AI_Trade", "AI_UES"],
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ---------------------------------------------------------
# DRILL DOWN VIEW
# ---------------------------------------------------------
st.subheader("🔍 Deep Dive: Individual Worker Review")

video_list = sorted_df["Video"].tolist()
selected_video = st.selectbox("Select a recording:", video_list)

if selected_video:
    vdata = sorted_df[sorted_df["Video"] == selected_video].iloc[0]

    st.markdown(f"### 📁 `{selected_video}`")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Task",            vdata["Detected Task"])
    m2.metric("Productivity",    f"{vdata['Productivity %']}%")
    m3.metric("Peak Exertion",   f"{vdata['Peak Exertion (px)']} px")
    m4.metric("Total Frames",    f"{int(vdata['Total Frames']):,}")

    if "AI_Trade" in vdata and pd.notna(vdata.get("AI_Trade")):
        st.markdown("---")
        st.markdown("### 🤖 AI Agent Analysis (Ollama LLaVA)")
        ai1, ai2 = st.columns(2)
        ues_val = vdata.get("AI_UES", "N/A")
        ai1.metric("Universal Efficiency Score (UES)", f"{ues_val}/100")
        ai2.metric("Primary Trade", vdata.get("AI_Trade", "N/A"))

        st.info(f"**Performance Summary:** {vdata.get('AI_Summary', 'N/A')}")

        det1, det2 = st.columns(2)
        with det1:
            st.markdown("**🔧 Specific Tasks Completed**")
            st.write(vdata.get("AI_Tasks", "N/A"))
        with det2:
            st.markdown("**📦 Quantified Output**")
            st.write(vdata.get("AI_Output", "N/A"))

        # Load raw JSON if available
        json_path = os.path.join(OUTPUT_DIR, f"Agent_Analysis_{selected_video}.json")
        if os.path.exists(json_path):
            with st.expander("📄 View Raw AI JSON Output"):
                with open(json_path) as f:
                    st.json(json.load(f))

    st.markdown("---")

    # Exertion plot image
    plot_path = os.path.join(OUTPUT_DIR, f"{selected_video}_plot.png")
    if os.path.exists(plot_path):
        st.image(Image.open(plot_path), caption=f"Time-Series Exertion — {selected_video}", use_container_width=True)
    else:
        st.info("📈 Time-series exertion plot not generated for this clip.")

    # Annotated video
    with st.expander("▶️ Watch Annotated Highlight Reel"):
        video_path = os.path.join(OUTPUT_DIR, f"{selected_video}_annotated.mp4")
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.warning("Annotated video file not found in outputs/.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption("BuilderBobs · Ironsite Hackathon 2026 · Built with OpenCV + Ollama LLaVA + Streamlit")
