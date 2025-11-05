import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(
    page_title="EduDRISHTI: NEET 2024 Inequality Analysis",
    layout="wide"
)

# --- DATA LOADING (Caches data for fast reloading) ---
@st.cache_data
def load_data():
    """Loads and preprocesses the final ML-ready data."""
    df = pd.read_csv('data/NEET_Master_ML_Data.csv')
    return df

df_master = load_data()


# ----------------------------------------------------------------------
# --- PAGE FUNCTIONS ---
# ----------------------------------------------------------------------

def page_overview():
    """Page 1: Key Performance Indicators and National Gaps."""
    st.title("🎓 Page 1: National Overview & Performance Gaps")
    st.markdown("---")

    # --- 1. Key Performance Indicators (KPIs) ---
    national_avg = df_master['national_average_marks'].iloc[0].round(2)
    total_centers = len(df_master)
    anomaly_count = df_master[df_master['Anomaly_Flag'] == -1].shape[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("National Average Marks", f"{national_avg}")
    col2.metric("Total Centers Analyzed", f"{total_centers:,}")
    col3.metric("Centers Flagged as Anomalies (ML)", f"{anomaly_count} ({round(anomaly_count/total_centers * 100, 2)}%)")

    st.markdown("---")

    # --- 2. Interactive Performance Gap Plot (Plotly) ---
    st.header("Top Performing Centers (Above National Average)")

    # Sort data for the chart
    df_sorted = df_master.sort_values(by='Center_v_National_Gap', ascending=False).head(50)

    fig = px.bar(
        df_sorted,
        x='Center_v_National_Gap',
        y='center_name',
        orientation='h',
        color='Ultra_High_Score_Ratio',
        title='Top 50 Centers by Performance Gap (Interactive)',
        hover_data=['state', 'city', 'total_students', 'Ultra_High_Score_Ratio'],
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def page_regional_drilldown():
    """Page 2: Focus on State and City Level Inequality."""
    st.title("🗺️ Page 2: Regional Inequality Drilldown")
    st.markdown("---")

    # --- 1. State Selector ---
    # Create a sidebar selection box for states
    state_list = ['All States'] + sorted(df_master['state'].unique().tolist())
    selected_state = st.sidebar.selectbox("Select State for Analysis", state_list)

    if selected_state != 'All States':
        df_filtered = df_master[df_master['state'] == selected_state]
        st.header(f"Performance Distribution in: {selected_state}")

        # Display State-level KPI
        state_avg = df_filtered['state_average_marks'].iloc[0].round(2)
        st.metric(f"State Average Marks ({selected_state})", f"{state_avg}")
        st.caption(f"National Average: {df_master['national_average_marks'].iloc[0].round(2)}")

        # --- 2. Box Plot for Intra-State Variance ---
        # Show the distribution of performance gap within the selected state
        fig_box = px.box(
            df_filtered,
            y='Center_v_State_Gap',
            color='city',
            title=f'Center Performance Variance by City in {selected_state}',
            hover_data=['center_name', 'Center_v_State_Gap']
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="State Average")
        st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.info("Select a State from the sidebar to view detailed intra-state inequality.")


def page_anomaly_detection():
    """Page 3: ML Results - Isolation Forest Anomaly Centers."""
    st.title("🚨 Page 3: Anomaly Detection (ML Results)")
    st.markdown("---")

    df_anomalies = df_master[df_master['Anomaly_Flag'] == -1].sort_values(by='Anomaly_Score')
    anomaly_count = df_anomalies.shape[0]

    st.header(f"Total Centers Flagged as Anomalies: {anomaly_count}")
    st.caption("These centers were mathematically flagged by the Isolation Forest model as statistically irregular based on a combination of performance, score distribution shape (Skewness/Kurtosis), and geography.")

    # --- 1. Anomaly Feature Plot ---
    st.subheader("Anomaly Score vs. Performance Gap")

    # Visualize the Anomaly Score against the Performance Gap
    fig_scatter = px.scatter(
        df_master,
        x='Anomaly_Score',
        y='Center_v_National_Gap',
        color='Anomaly_Type', # Use the flag for color
        hover_data=['center_name', 'state', 'Ultra_High_Score_Ratio', 'Center_Skewness'],
        size='total_students',
        title='Anomaly Score Distribution',
        color_discrete_map={'Anomalous Center': 'red', 'Normal Center': 'blue'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- 2. Table of Flagged Anomalies ---
    st.subheader("Detailed List of Anomalous Centers")
    
    # Select key columns for the final table
    table_cols = [
        'state',
        'city',
        'center_name',
        'Center_v_National_Gap',
        'Ultra_High_Score_Ratio',
        'Center_Skewness',
        'Anomaly_Score'
    ]

    st.dataframe(df_anomalies[table_cols].head(50).style.highlight_max(axis=0, subset=['Center_v_National_Gap', 'Ultra_High_Score_Ratio'], color='yellow'),
                 height=500, use_container_width=True)


# ----------------------------------------------------------------------
# --- MAIN APP NAVIGATION ---
# ----------------------------------------------------------------------

# Create the sidebar for navigation
st.sidebar.title("EduDRISHTI:")
st.sidebar.markdown("Data research for inequality in student test insights")
st.sidebar.subheader("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("1. National Overview", "2. Regional Drilldown", "3. ML Anomaly Results")
)
st.sidebar.markdown("---")
st.sidebar.info("Inspired by the 2024 NEET controversy.")


# Route the user to the selected page function
if page == "1. National Overview":
    page_overview()
elif page == "2. Regional Drilldown":
    page_regional_drilldown()
elif page == "3. ML Anomaly Results":

    page_anomaly_detection()
