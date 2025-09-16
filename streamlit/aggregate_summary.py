"""
Aggregate Summary Page for Route Optimization Dashboard

Shows system-wide summary statistics from aggregate-summary.parquet file.
"""

# standard library imports
import sys
from pathlib import Path

# 3rd-party imports
import streamlit as st
import polars as pl

# add current directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

# local imports
from utils import load_aggregate_metrics


def show_aggregate_summary() -> None:
    """
    Display the aggregate summary page showing system-wide statistics.

    :return: None
    """
    st.title("📊 Aggregate Summary")
    st.markdown("System-wide performance metrics across all algorithm combinations")

    # load aggregate summary data
    with st.spinner("Loading aggregate summary data..."):
        summary_df = load_aggregate_metrics()

    if summary_df is None:
        st.error("Failed to load aggregate summary data")
        return

    if len(summary_df) == 0:
        st.warning("No aggregate summary data available")
        return

    # convert to pandas for easier display
    if isinstance(summary_df, pl.DataFrame):
        summary_pd = summary_df.to_pandas()
    else:
        summary_pd = summary_df

    # display summary statistics
    st.subheader("📈 Performance Overview")

    # create metric columns
    col1, col2, col3, col4 = st.columns(4)

    # calculate aggregate metrics across all combinations
    avg_weekly_duration = summary_pd['average_weekly_duration'].mean()
    avg_utilization = summary_pd['average_utilization'].mean()
    avg_overutilized = summary_pd['average_overutilized_days'].mean()
    avg_underutilized = summary_pd['average_underutilized_days'].mean()

    with col1:
        st.metric(
            "Avg Weekly Duration",
            f"{avg_weekly_duration:.1f} hrs",
            help="Average weekly route duration across all algorithm combinations"
        )

    with col2:
        st.metric(
            "Avg Utilization",
            f"{avg_utilization:.1%}",
            help="Average capacity utilization across all combinations"
        )

    with col3:
        st.metric(
            "Avg Overutilized Days",
            f"{avg_overutilized:.1f}",
            help="Average number of overutilized days per zone"
        )

    with col4:
        st.metric(
            "Avg Underutilized Days",
            f"{avg_underutilized:.1f}",
            help="Average number of underutilized days per zone"
        )

    # detailed results table
    st.subheader("🔍 Detailed Results by Algorithm Combination")

    # add algorithm combination column for clarity
    summary_display = summary_pd.copy()
    summary_display['Algorithm Combination'] = (
        summary_display['clusterer'] + " + " + summary_display['balancer']
    )

    # select columns to display
    display_columns = [
        'Algorithm Combination',
        'average_weekly_duration',
        'average_utilization',
        'average_overutilized_days',
        'average_underutilized_days',
        'average_daily_pos_time',
        'average_daily_drive_time',
        'average_duration_standard_deviation',
        'created_on'
    ]

    # rename columns for better display
    column_mapping = {
        'average_weekly_duration': 'Weekly Duration (hrs)',
        'average_utilization': 'Utilization (%)',
        'average_overutilized_days': 'Overutilized Days',
        'average_underutilized_days': 'Underutilized Days',
        'average_daily_pos_time': 'Daily POS Time (hrs)',
        'average_daily_drive_time': 'Daily Drive Time (hrs)',
        'average_duration_standard_deviation': 'Duration StdDev',
        'created_on': 'Created On'
    }

    summary_display = summary_display[display_columns].rename(columns=column_mapping)

    # format numeric columns
    numeric_columns = [
        'Weekly Duration (hrs)',
        'Daily POS Time (hrs)',
        'Daily Drive Time (hrs)',
        'Secondary Duration StdDev'
    ]

    for col in numeric_columns:
        if col in summary_display.columns:
            summary_display[col] = summary_display[col].round(2)

    # format utilization as percentage
    if 'Utilization (%)' in summary_display.columns:
        summary_display['Utilization (%)'] = (summary_display['Utilization (%)'] * 100).round(1)

    # sort by algorithm combination
    summary_display = summary_display.sort_values('Algorithm Combination')

    # display table with formatting
    st.dataframe(
        summary_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Weekly Duration (hrs)': st.column_config.NumberColumn(format="%.2f"),
            'Utilization (%)': st.column_config.NumberColumn(format="%.1f%%"),
            'Overutilized Days': st.column_config.NumberColumn(format="%.1f"),
            'Underutilized Days': st.column_config.NumberColumn(format="%.1f"),
            'Daily POS Time (hrs)': st.column_config.NumberColumn(format="%.2f"),
            'Daily Drive Time (hrs)': st.column_config.NumberColumn(format="%.2f"),
            'Secondary Duration StdDev': st.column_config.NumberColumn(format="%.2f"),
            'Created On': st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss")
        }
    )

    # algorithm comparison charts
    st.subheader("📊 Algorithm Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # underutilized days by clusterer
        clusterer_perf = summary_pd.groupby('clusterer')['average_underutilized_days'].mean().sort_values()
        st.bar_chart(
            clusterer_perf,
            use_container_width=True
        )
        st.caption("Average Underutilized Days by Clusterer")

    with col2:
        # underutilized days by balancer
        balancer_perf = summary_pd.groupby('balancer')['average_underutilized_days'].mean().sort_values()
        st.bar_chart(
            balancer_perf,
            use_container_width=True
        )
        st.caption("Average Underutilized Days by Balancer")

    # data info
    st.subheader("ℹ️ Data Information")
    st.info(f"""
    **Total Algorithm Combinations**: {len(summary_pd)}

    **Data Source**: aggregate-summary.parquet

    **Last Updated**: {summary_pd['created_on'].max() if 'created_on' in summary_pd.columns else 'Unknown'}

    This page shows system-wide performance metrics aggregated across all zones for each algorithm combination.
    """)