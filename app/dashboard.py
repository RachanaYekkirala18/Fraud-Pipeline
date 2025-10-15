import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import yaml
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Load YAML configuration
def load_yaml(filepath):
    """Load YAML configuration file"""
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {filepath}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Fraud Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💳 Fraud Risk Analytics (IEEE-CIS)")

# Load paths from config
# Get the project root directory (parent of app folder)
project_root = Path(__file__).parent.parent
config_path = project_root / "configs" / "paths.yaml"

try:
    paths = load_yaml(str(config_path))["local"]
    GOLD = paths["gold"]
except Exception as e:
    st.error(f"Error loading paths configuration: {str(e)}")
    st.info(f"Looking for config at: {config_path}")
    st.stop()

# Find scored data files (try both parquet and CSV)
files = glob.glob(f"{GOLD}/scored.parquet")
if not files:
    files = glob.glob(f"{GOLD}/scored.csv")

if not files:
    st.warning("⚠️ No scored data found. Run batch_score.py first.")
    st.info("Expected location: " + f"{GOLD}/scored.parquet or {GOLD}/scored.csv")
    st.stop()

# Load data
@st.cache_data
def load_data(filepath):
    """Load scored data file with caching"""
    if filepath.endswith('.parquet'):
        return pd.read_parquet(filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

with st.spinner("Loading data..."):
    df = load_data(files[-1])

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Risk threshold filter
risk_threshold = st.sidebar.slider(
    "Fraud Score Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Transactions above this score are considered high risk"
)

# Top N filter
top_n = st.sidebar.number_input(
    "Number of top risks to display",
    min_value=10,
    max_value=100,
    value=20,
    step=10
)

# Calculate metrics
high_risk = df[df["fraud_score"] >= risk_threshold]
high_risk_pct = (len(high_risk) / len(df)) * 100 if len(df) > 0 else 0

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Transactions",
        value=f"{len(df):,}"
    )

with col2:
    st.metric(
        label="High Risk Transactions",
        value=f"{len(high_risk):,}",
        delta=f"{high_risk_pct:.1f}%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Mean Fraud Score",
        value=f"{df['fraud_score'].mean():.3f}"
    )

with col4:
    st.metric(
        label="Max Fraud Score",
        value=f"{df['fraud_score'].max():.3f}"
    )

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "⚠️ High Risk Transactions",
    "📈 Distributions",
    "📉 Trends"
])

# TAB 1: Overview
with tab1:
    st.subheader("Fraud Score Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = px.histogram(
            df,
            x="fraud_score",
            nbins=50,
            title="Fraud Score Distribution",
            labels={"fraud_score": "Fraud Score", "count": "Count"},
            color_discrete_sequence=["#1f77b4"]
        )
        fig_hist.add_vline(
            x=risk_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {risk_threshold}",
            annotation_position="top"
        )
        fig_hist.update_layout(
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot
        fig_box = px.box(
            df,
            y="fraud_score",
            title="Fraud Score Statistics",
            labels={"fraud_score": "Fraud Score"},
            color_discrete_sequence=["#1f77b4"]
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Statistics table
    st.subheader("Summary Statistics")
    stats = df["fraud_score"].describe().to_frame().T
    stats.columns = ["Count", "Mean", "Std Dev", "Min", "25%", "50%", "75%", "Max"]
    st.dataframe(stats, use_container_width=True)

# TAB 2: High Risk Transactions
with tab2:
    st.subheader(f"Top {top_n} Highest Risk Transactions")
    
    top_risks = df.sort_values("fraud_score", ascending=False).head(top_n)
    
    # Display styled dataframe
    styled_df = top_risks.style.background_gradient(
        subset=["fraud_score"],
        cmap="Reds",
        vmin=0,
        vmax=1
    ).format({"fraud_score": "{:.4f}"})
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download button
    col1, col2 = st.columns([1, 4])
    with col1:
        csv = top_risks.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"high_risk_transactions_top{top_n}.csv",
            mime="text/csv"
        )

# TAB 3: Distributions
with tab3:
    st.subheader("Risk Category Analysis")
    
    # Create risk categories
    bins = [0, 0.3, 0.5, 0.7, 1.0]
    labels = ["Low", "Medium", "High", "Critical"]
    df["risk_category"] = pd.cut(
        df["fraud_score"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        risk_counts = df["risk_category"].value_counts().sort_index()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Category Distribution",
            color=risk_counts.index,
            color_discrete_map={
                "Low": "#2ecc71",
                "Medium": "#f39c12",
                "High": "#e67e22",
                "Critical": "#e74c3c"
            }
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Transactions by Risk Category",
            labels={"x": "Risk Category", "y": "Number of Transactions"},
            color=risk_counts.index,
            color_discrete_map={
                "Low": "#2ecc71",
                "Medium": "#f39c12",
                "High": "#e67e22",
                "Critical": "#e74c3c"
            }
        )
        fig_bar.update_layout(
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Category breakdown table
    st.subheader("Category Breakdown")
    category_stats = df.groupby("risk_category")["fraud_score"].agg([
        ("Count", "count"),
        ("Mean Score", "mean"),
        ("Min Score", "min"),
        ("Max Score", "max")
    ]).round(4)
    st.dataframe(category_stats, use_container_width=True)

# TAB 4: Trends
with tab4:
    st.subheader("Fraud Score Trends Over Time")
    
    # Rolling window selector
    window_size = st.slider(
        "Rolling Window Size",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Number of transactions to include in rolling average"
    )
    
    # Calculate rolling statistics
    df_sorted = df.reset_index(drop=True).copy()
    df_sorted["rolling_mean"] = df_sorted["fraud_score"].rolling(
        window=window_size,
        center=False
    ).mean()
    df_sorted["rolling_std"] = df_sorted["fraud_score"].rolling(
        window=window_size,
        center=False
    ).std()
    
    # Create trend plot
    fig_trend = go.Figure()
    
    # Add rolling mean
    fig_trend.add_trace(go.Scatter(
        x=df_sorted.index,
        y=df_sorted["rolling_mean"],
        mode="lines",
        name="Rolling Mean",
        line=dict(color="blue", width=2)
    ))
    
    # Add confidence band (±1 std)
    upper_bound = df_sorted["rolling_mean"] + df_sorted["rolling_std"]
    lower_bound = df_sorted["rolling_mean"] - df_sorted["rolling_std"]
    
    fig_trend.add_trace(go.Scatter(
        x=df_sorted.index,
        y=upper_bound,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=df_sorted.index,
        y=lower_bound,
        mode="lines",
        line=dict(width=0),
        fillcolor="rgba(0, 100, 255, 0.2)",
        fill="tonexty",
        name="±1 Std Dev",
        hoverinfo="skip"
    ))
    
    fig_trend.update_layout(
        title=f"Fraud Score Rolling Average (Window: {window_size})",
        xaxis_title="Transaction Index",
        yaxis_title="Fraud Score",
        hovermode="x unified",
        height=500
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Additional trend insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trend_mean = df_sorted["rolling_mean"].dropna().mean()
        st.metric("Average Rolling Mean", f"{trend_mean:.4f}")
    
    with col2:
        trend_std = df_sorted["rolling_std"].dropna().mean()
        st.metric("Average Volatility (Std)", f"{trend_std:.4f}")
    
    with col3:
        recent_mean = df_sorted["rolling_mean"].dropna().tail(100).mean()
        st.metric("Recent Trend (Last 100)", f"{recent_mean:.4f}")

# Footer
st.divider()
file_info = files[-1].split("/")[-1] if "/" in files[-1] else files[-1].split("\\")[-1]
st.caption(
    f"📊 Data source: {file_info} | "
    f"Total records: {len(df):,} | "
    f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
)