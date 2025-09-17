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
    st.title("Aggregate Summary")
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

    # Add algorithm selectors
    st.subheader("Algorithm Filters")

    # Add custom CSS for colored multiselect tags
    selector_colors_css = """
    <style>
    /* Style the multiselect selected tags (the items under the dropdown) */
    span[data-baseweb="tag"] {
        background-color: #4A5568 !important;
        color: white !important;
    }
    span[data-baseweb="tag"] > span {
        color: white !important;
    }
    </style>
    """
    st.markdown(selector_colors_css, unsafe_allow_html=True)

    # Get unique algorithms
    available_clusterers = sorted(summary_pd['clusterer'].unique())
    available_balancers = sorted(summary_pd['balancer'].unique())

    selected_clusterers = st.multiselect(
        "Select Clusterers:",
        options=available_clusterers,
        default=available_clusterers,
        key="aggregate_clusterer_select"
    )

    selected_balancers = st.multiselect(
        "Select Balancers:",
        options=available_balancers,
        default=available_balancers,
        key="aggregate_balancer_select"
    )

    # Ensure at least one option is selected for each
    if not selected_clusterers:
        st.error("Please select at least one clusterer.")
        return
    if not selected_balancers:
        st.error("Please select at least one balancer.")
        return

    # Filter data based on selections
    summary_pd = summary_pd[
        (summary_pd['clusterer'].isin(selected_clusterers)) &
        (summary_pd['balancer'].isin(selected_balancers))
    ]

    # KEY METRICS TABLE
    st.subheader("Key Metrics Overview")

    # Calculate and display key metrics averages
    col1, col2 = st.columns(2)
    with col1:
        avg_weekly_duration = summary_pd['average_weekly_duration'].mean()
        st.metric("Average Weekly Duration", f"{avg_weekly_duration:.1f} hrs")
    with col2:
        avg_duration_std = summary_pd['average_duration_standard_deviation'].mean()
        st.metric("Average Duration StdDev", f"{avg_duration_std:.2f}")

    # prepare display dataframe
    summary_display = summary_pd.copy()

    # select primary columns to display
    primary_columns = [
        'clusterer',
        'balancer',
        'average_weekly_duration',
        'average_duration_standard_deviation'
    ]

    # rename columns for better display
    primary_column_mapping = {
        'clusterer': 'Clusterer',
        'balancer': 'Balancer',
        'average_weekly_duration': 'Weekly Duration (hrs)',
        'average_duration_standard_deviation': 'Duration StdDev'
    }

    primary_display = summary_display[primary_columns].rename(columns=primary_column_mapping)

    # format numeric columns
    primary_display['Weekly Duration (hrs)'] = primary_display['Weekly Duration (hrs)'].round(2)
    primary_display['Duration StdDev'] = primary_display['Duration StdDev'].round(2)

    # sort by clusterer and balancer
    primary_display = primary_display.sort_values(['Clusterer', 'Balancer'])

    # display primary table with formatting
    st.dataframe(
        primary_display,
        width='stretch',
        hide_index=True,
        column_config={
            'Weekly Duration (hrs)': st.column_config.NumberColumn(format="%.2f"),
            'Duration StdDev': st.column_config.NumberColumn(format="%.2f")
        }
    )

    # Scatter plot of Weekly Duration vs Duration StdDev
    import plotly.graph_objects as go
    import plotly.express as px

    with st.expander("Weekly Duration vs Duration Standard Deviation", expanded=True):
        # Get unique values for clusterers and balancers
        clusterers = summary_pd['clusterer'].unique()
        balancers = summary_pd['balancer'].unique()

        # Define soft color palette and symbol mapping
        soft_colors = ['#FF7F7F', '#7FDF7F', '#7F9FFF', '#FFD700', '#FF9F7F', '#D77FD7', '#9F9FFF']
        colors = soft_colors[:len(clusterers)]
        symbols = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'cross', 'x'][:len(balancers)]

        # Create color and symbol mappings
        clusterer_colors = {clusterer: colors[i] for i, clusterer in enumerate(clusterers)}
        balancer_symbols = {balancer: symbols[i] for i, balancer in enumerate(balancers)}

        # Create figure
        fig = go.Figure()

        # Add traces for each combination, but group by clusterer for color legend
        for clusterer in clusterers:
            for balancer in balancers:
                data_subset = summary_pd[(summary_pd['clusterer'] == clusterer) & (summary_pd['balancer'] == balancer)]
                if not data_subset.empty:
                    fig.add_trace(go.Scatter(
                        x=data_subset['average_weekly_duration'],
                        y=data_subset['average_duration_standard_deviation'],
                        mode='markers',
                        marker=dict(
                            color='rgba(0,0,0,0)',
                            symbol=balancer_symbols[balancer],
                            size=12,
                            line=dict(
                                color=clusterer_colors[clusterer],
                                width=2
                            )
                        ),
                        name=clusterer,
                        legendgroup='clusterer',
                        legendgrouptitle_text='Clusterer',
                        hovertemplate='<b>Weekly Duration:</b> %{x:.2f} hrs<br>' +
                                      '<b>Duration StdDev:</b> %{y:.2f}<br>' +
                                      '<b>Clusterer:</b> ' + clusterer + '<br>' +
                                      '<b>Balancer:</b> ' + balancer + '<extra></extra>',
                        showlegend=True if balancer == balancers[0] else False  # Only show clusterer legend once
                    ))

        # Add invisible traces for balancer legend
        for balancer in balancers:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    color='rgba(0,0,0,0)',
                    symbol=balancer_symbols[balancer],
                    size=12,
                    line=dict(
                        color='gray',
                        width=2
                    )
                ),
                name=balancer,
                legendgroup='balancer',
                legendgrouptitle_text='Balancer',
                showlegend=True
            ))

        fig.update_layout(
            xaxis_title='Weekly Duration (hrs)',
            yaxis_title='Duration StdDev',
            height=500,
            xaxis=dict(range=[0, None], showgrid=True),
            yaxis=dict(range=[0, None], showgrid=True),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                groupclick="toggleitem"
            ),
            shapes=[
                # Vertical line at x=0
                dict(
                    type="line",
                    x0=0, x1=0,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color="lightgray", width=1)
                ),
                # Horizontal line at y=0
                dict(
                    type="line",
                    x0=0, x1=1,
                    xref="paper",
                    y0=0, y1=0,
                    line=dict(color="lightgray", width=1)
                )
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

    # Violin plots for weekly duration
    with st.expander("Weekly Duration Distribution", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            # Create violin plot for clusterers - weekly duration
            fig_clusterer = go.Figure()
            for clusterer in clusterers:
                clusterer_data = summary_pd[summary_pd['clusterer'] == clusterer]['average_weekly_duration']
                if len(clusterer_data) > 0:
                    fig_clusterer.add_trace(go.Violin(
                        y=clusterer_data,
                        name=clusterer,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=clusterer_colors.get(clusterer, '#cccccc'),
                        opacity=0.6,
                        line_color=clusterer_colors.get(clusterer, '#cccccc')
                    ))

            fig_clusterer.update_layout(
                xaxis_title='Clusterer',
                yaxis_title='Weekly Duration (hrs)',
                height=350,
                showlegend=False
            )
            fig_clusterer.update_yaxes(range=[0, None])

            st.plotly_chart(fig_clusterer, use_container_width=True)

        with col2:
            # Create violin plot for balancers - weekly duration
            fig_balancer = go.Figure()
            for balancer in balancers:
                balancer_data = summary_pd[summary_pd['balancer'] == balancer]['average_weekly_duration']
                if len(balancer_data) > 0:
                    fig_balancer.add_trace(go.Violin(
                        y=balancer_data,
                        name=balancer,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor='#7F9FFF',
                        opacity=0.6,
                        line_color='#7F9FFF'
                    ))

            fig_balancer.update_layout(
                xaxis_title='Balancer',
                yaxis_title='Weekly Duration (hrs)',
                height=350,
                showlegend=False
            )
            fig_balancer.update_yaxes(range=[0, None])

            st.plotly_chart(fig_balancer, use_container_width=True)

    # Bar charts for weekly duration
    with st.expander("Weekly Duration Averages", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for clusterers - weekly duration
            clusterer_duration_avg = summary_pd.groupby('clusterer')['average_weekly_duration'].mean().sort_values()
            fig_clusterer_duration_bar = go.Figure()
            fig_clusterer_duration_bar.add_trace(go.Bar(
                x=clusterer_duration_avg.index,
                y=clusterer_duration_avg.values,
                marker_color=[clusterer_colors.get(c, '#cccccc') for c in clusterer_duration_avg.index],
                text=[f"{v:.1f}h" for v in clusterer_duration_avg.values],
                textposition='auto'
            ))

            fig_clusterer_duration_bar.update_layout(
                xaxis_title='Clusterer',
                yaxis_title='Average Weekly Duration (hrs)',
                height=300,
                showlegend=False
            )
            fig_clusterer_duration_bar.update_yaxes(range=[0, None])

            st.plotly_chart(fig_clusterer_duration_bar, use_container_width=True)

        with col2:
            # Create bar chart for balancers - weekly duration
            balancer_duration_avg = summary_pd.groupby('balancer')['average_weekly_duration'].mean().sort_values()
            fig_balancer_duration_bar = go.Figure()
            fig_balancer_duration_bar.add_trace(go.Bar(
                x=balancer_duration_avg.index,
                y=balancer_duration_avg.values,
                marker_color='#7F9FFF',
                text=[f"{v:.1f}h" for v in balancer_duration_avg.values],
                textposition='auto'
            ))

            fig_balancer_duration_bar.update_layout(
                xaxis_title='Balancer',
                yaxis_title='Average Weekly Duration (hrs)',
                height=300,
                showlegend=False
            )
            fig_balancer_duration_bar.update_yaxes(range=[0, None])

            st.plotly_chart(fig_balancer_duration_bar, use_container_width=True)

    # Violin plots for duration standard deviation
    with st.expander("Duration Standard Deviation Distribution", expanded=True):
        col3, col4 = st.columns(2)

        with col3:
            # Create violin plot for clusterers - duration std dev
            fig_clusterer_std = go.Figure()
            for clusterer in clusterers:
                clusterer_std_data = summary_pd[summary_pd['clusterer'] == clusterer]['average_duration_standard_deviation']
                if len(clusterer_std_data) > 0:
                    fig_clusterer_std.add_trace(go.Violin(
                        y=clusterer_std_data,
                        name=clusterer,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=clusterer_colors.get(clusterer, '#cccccc'),
                        opacity=0.6,
                        line_color=clusterer_colors.get(clusterer, '#cccccc')
                    ))

            fig_clusterer_std.update_layout(
                xaxis_title='Clusterer',
                yaxis_title='Duration StdDev',
                height=350,
                showlegend=False
            )
            fig_clusterer_std.update_yaxes(range=[0, None])

            st.plotly_chart(fig_clusterer_std, use_container_width=True)

        with col4:
            # Create violin plot for balancers - duration std dev
            fig_balancer_std = go.Figure()
            for balancer in balancers:
                balancer_std_data = summary_pd[summary_pd['balancer'] == balancer]['average_duration_standard_deviation']
                if len(balancer_std_data) > 0:
                    fig_balancer_std.add_trace(go.Violin(
                        y=balancer_std_data,
                        name=balancer,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor='#7F9FFF',
                        opacity=0.6,
                        line_color='#7F9FFF'
                    ))

            fig_balancer_std.update_layout(
                xaxis_title='Balancer',
                yaxis_title='Duration StdDev',
                height=350,
                showlegend=False
            )
            fig_balancer_std.update_yaxes(range=[0, None])

            st.plotly_chart(fig_balancer_std, use_container_width=True)

    # Bar charts for duration standard deviation
    with st.expander("Duration Standard Deviation Averages", expanded=False):
        col3, col4 = st.columns(2)

        with col3:
            # Create bar chart for clusterers - duration std dev
            clusterer_std_avg = summary_pd.groupby('clusterer')['average_duration_standard_deviation'].mean().sort_values()
            fig_clusterer_std_bar = go.Figure()
            fig_clusterer_std_bar.add_trace(go.Bar(
                x=clusterer_std_avg.index,
                y=clusterer_std_avg.values,
                marker_color=[clusterer_colors.get(c, '#cccccc') for c in clusterer_std_avg.index],
                text=[f"{v:.2f}" for v in clusterer_std_avg.values],
                textposition='auto'
            ))

            fig_clusterer_std_bar.update_layout(
                xaxis_title='Clusterer',
                yaxis_title='Average Duration StdDev',
                height=300,
                showlegend=False
            )
            fig_clusterer_std_bar.update_yaxes(range=[0, None])

            st.plotly_chart(fig_clusterer_std_bar, use_container_width=True)

        with col4:
            # Create bar chart for balancers - duration std dev
            balancer_std_avg = summary_pd.groupby('balancer')['average_duration_standard_deviation'].mean().sort_values()
            fig_balancer_std_bar = go.Figure()
            fig_balancer_std_bar.add_trace(go.Bar(
                x=balancer_std_avg.index,
                y=balancer_std_avg.values,
                marker_color='#7F9FFF',
                text=[f"{v:.2f}" for v in balancer_std_avg.values],
                textposition='auto'
            ))

            fig_balancer_std_bar.update_layout(
                xaxis_title='Balancer',
                yaxis_title='Average Duration StdDev',
                height=300,
                showlegend=False
            )
            fig_balancer_std_bar.update_yaxes(range=[0, None])

            st.plotly_chart(fig_balancer_std_bar, use_container_width=True)

    # ADDITIONAL METRICS TABLE
    st.subheader("Additional Metrics Overview")

    # Calculate and display additional metrics averages
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_utilization = summary_pd['average_utilization'].mean()
        st.metric("Average Utilization", f"{avg_utilization:.1f}%")
    with col2:
        avg_overutilized = summary_pd['average_overutilized_days'].mean()
        st.metric("Average Overutilized Days", f"{avg_overutilized:.1f}")
    with col3:
        avg_underutilized = summary_pd['average_underutilized_days'].mean()
        st.metric("Average Underutilized Days", f"{avg_underutilized:.1f}")

    col4, col5, col6 = st.columns(3)
    with col4:
        avg_pos_time = summary_pd['average_daily_pos_time'].mean()
        st.metric("Average Daily POS Time", f"{avg_pos_time:.2f} hrs")
    with col5:
        avg_drive_time = summary_pd['average_daily_drive_time'].mean()
        st.metric("Average Daily Drive Time", f"{avg_drive_time:.2f} hrs")
    with col6:
        # Add timestamp info
        latest_timestamp = summary_pd['created_on'].max() if 'created_on' in summary_pd.columns else 'Unknown'
        st.metric("Latest Data", str(latest_timestamp)[:10] if latest_timestamp != 'Unknown' else 'Unknown')

    # select secondary columns to display
    secondary_columns = [
        'clusterer',
        'balancer',
        'average_utilization',
        'average_overutilized_days',
        'average_underutilized_days',
        'average_daily_pos_time',
        'average_daily_drive_time'
    ]

    # rename columns for better display
    secondary_column_mapping = {
        'clusterer': 'Clusterer',
        'balancer': 'Balancer',
        'average_utilization': 'Utilization (%)',
        'average_overutilized_days': 'Overutilized Days',
        'average_underutilized_days': 'Underutilized Days',
        'average_daily_pos_time': 'Daily POS Time (hrs)',
        'average_daily_drive_time': 'Daily Drive Time (hrs)'
    }

    secondary_display = summary_display[secondary_columns].rename(columns=secondary_column_mapping)

    # format numeric columns
    secondary_display['Utilization (%)'] = (secondary_display['Utilization (%)']).round(1)
    secondary_display['Overutilized Days'] = secondary_display['Overutilized Days'].round(1)
    secondary_display['Underutilized Days'] = secondary_display['Underutilized Days'].round(1)
    secondary_display['Daily POS Time (hrs)'] = secondary_display['Daily POS Time (hrs)'].round(2)
    secondary_display['Daily Drive Time (hrs)'] = secondary_display['Daily Drive Time (hrs)'].round(2)

    # sort by clusterer and balancer
    secondary_display = secondary_display.sort_values(['Clusterer', 'Balancer'])

    # display secondary table with formatting
    st.dataframe(
        secondary_display,
        width='stretch',
        hide_index=True,
        column_config={
            'Utilization (%)': st.column_config.NumberColumn(format="%.1f"),
            'Overutilized Days': st.column_config.NumberColumn(format="%.1f"),
            'Underutilized Days': st.column_config.NumberColumn(format="%.1f"),
            'Daily POS Time (hrs)': st.column_config.NumberColumn(format="%.2f"),
            'Daily Drive Time (hrs)': st.column_config.NumberColumn(format="%.2f")
        }
    )

    # Bar charts for utilization metrics
    with st.expander("Average Utilization", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for clusterers - utilization
            clusterer_util_avg = summary_pd.groupby('clusterer')['average_utilization'].mean().sort_values()
            fig_clusterer_util = go.Figure()
            fig_clusterer_util.add_trace(go.Bar(
                x=clusterer_util_avg.index,
                y=clusterer_util_avg.values,
                marker_color=[clusterer_colors.get(c, '#cccccc') for c in clusterer_util_avg.index],
                text=[f"{v:.1f}%" for v in clusterer_util_avg.values],
                textposition='auto'
            ))

            fig_clusterer_util.update_layout(
                xaxis_title='Clusterer',
                yaxis_title='Average Utilization',
                height=300,
                showlegend=False
            )
            fig_clusterer_util.update_yaxes(range=[0, None])

            st.plotly_chart(fig_clusterer_util, use_container_width=True)

        with col2:
            # Create bar chart for balancers - utilization
            balancer_util_avg = summary_pd.groupby('balancer')['average_utilization'].mean().sort_values()
            fig_balancer_util = go.Figure()
            fig_balancer_util.add_trace(go.Bar(
                x=balancer_util_avg.index,
                y=balancer_util_avg.values,
                marker_color='#7F9FFF',
                text=[f"{v:.1f}%" for v in balancer_util_avg.values],
                textposition='auto'
            ))

            fig_balancer_util.update_layout(
                xaxis_title='Balancer',
                yaxis_title='Average Utilization',
                height=300,
                showlegend=False
            )
            fig_balancer_util.update_yaxes(range=[0, None])

            st.plotly_chart(fig_balancer_util, use_container_width=True)

    # Bar charts for over/under utilized days
    with st.expander("Average Overutilized Days", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for clusterers - overutilized days
            clusterer_over_avg = summary_pd.groupby('clusterer')['average_overutilized_days'].mean().sort_values()
            fig_clusterer_over = go.Figure()
            fig_clusterer_over.add_trace(go.Bar(
                x=clusterer_over_avg.index,
                y=clusterer_over_avg.values,
                marker_color=[clusterer_colors.get(c, '#cccccc') for c in clusterer_over_avg.index],
                text=[f"{v:.1f}" for v in clusterer_over_avg.values],
                textposition='auto'
            ))

            fig_clusterer_over.update_layout(
                xaxis_title='Clusterer',
                yaxis_title='Average Overutilized Days',
                height=300,
                showlegend=False
            )
            fig_clusterer_over.update_yaxes(range=[0, None])

            st.plotly_chart(fig_clusterer_over, use_container_width=True)

        with col2:
            # Create bar chart for balancers - overutilized days
            balancer_over_avg = summary_pd.groupby('balancer')['average_overutilized_days'].mean().sort_values()
            fig_balancer_over = go.Figure()
            fig_balancer_over.add_trace(go.Bar(
                x=balancer_over_avg.index,
                y=balancer_over_avg.values,
                marker_color='#7F9FFF',
                text=[f"{v:.1f}" for v in balancer_over_avg.values],
                textposition='auto'
            ))

            fig_balancer_over.update_layout(
                xaxis_title='Balancer',
                yaxis_title='Average Overutilized Days',
                height=300,
                showlegend=False
            )
            fig_balancer_over.update_yaxes(range=[0, None])

            st.plotly_chart(fig_balancer_over, use_container_width=True)

    # Bar charts for daily time metrics
    with st.expander("Average Daily POS Time", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for clusterers - daily POS time
            clusterer_pos_avg = summary_pd.groupby('clusterer')['average_daily_pos_time'].mean().sort_values()
            fig_clusterer_pos = go.Figure()
            fig_clusterer_pos.add_trace(go.Bar(
                x=clusterer_pos_avg.index,
                y=clusterer_pos_avg.values,
                marker_color=[clusterer_colors.get(c, '#cccccc') for c in clusterer_pos_avg.index],
                text=[f"{v:.2f}h" for v in clusterer_pos_avg.values],
                textposition='auto'
            ))

            fig_clusterer_pos.update_layout(
                xaxis_title='Clusterer',
                yaxis_title='Average Daily POS Time (hrs)',
                height=300,
                showlegend=False
            )
            fig_clusterer_pos.update_yaxes(range=[0, None])

            st.plotly_chart(fig_clusterer_pos, use_container_width=True)

        with col2:
            # Create bar chart for balancers - daily POS time
            balancer_pos_avg = summary_pd.groupby('balancer')['average_daily_pos_time'].mean().sort_values()
            fig_balancer_pos = go.Figure()
            fig_balancer_pos.add_trace(go.Bar(
                x=balancer_pos_avg.index,
                y=balancer_pos_avg.values,
                marker_color='#7F9FFF',
                text=[f"{v:.2f}h" for v in balancer_pos_avg.values],
                textposition='auto'
            ))

            fig_balancer_pos.update_layout(
                xaxis_title='Balancer',
                yaxis_title='Average Daily POS Time (hrs)',
                height=300,
                showlegend=False
            )
            fig_balancer_pos.update_yaxes(range=[0, None])

            st.plotly_chart(fig_balancer_pos, use_container_width=True)

    # Bar charts for underutilized days
    with st.expander("Average Underutilized Days", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for clusterers - underutilized days
            clusterer_under_avg = summary_pd.groupby('clusterer')['average_underutilized_days'].mean().sort_values()
            fig_clusterer_under = go.Figure()
            fig_clusterer_under.add_trace(go.Bar(
                x=clusterer_under_avg.index,
                y=clusterer_under_avg.values,
                marker_color=[clusterer_colors.get(c, '#cccccc') for c in clusterer_under_avg.index],
                text=[f"{v:.1f}" for v in clusterer_under_avg.values],
                textposition='auto'
            ))

            fig_clusterer_under.update_layout(
                xaxis_title='Clusterer',
                yaxis_title='Average Underutilized Days',
                height=300,
                showlegend=False
            )
            fig_clusterer_under.update_yaxes(range=[0, None])

            st.plotly_chart(fig_clusterer_under, use_container_width=True)

        with col2:
            # Create bar chart for balancers - underutilized days
            balancer_under_avg = summary_pd.groupby('balancer')['average_underutilized_days'].mean().sort_values()
            fig_balancer_under = go.Figure()
            fig_balancer_under.add_trace(go.Bar(
                x=balancer_under_avg.index,
                y=balancer_under_avg.values,
                marker_color='#7F9FFF',
                text=[f"{v:.1f}" for v in balancer_under_avg.values],
                textposition='auto'
            ))

            fig_balancer_under.update_layout(
                xaxis_title='Balancer',
                yaxis_title='Average Underutilized Days',
                height=300,
                showlegend=False
            )
            fig_balancer_under.update_yaxes(range=[0, None])

            st.plotly_chart(fig_balancer_under, use_container_width=True)

    # Bar charts for daily drive time
    with st.expander("Average Daily Drive Time", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for clusterers - daily drive time
            clusterer_drive_avg = summary_pd.groupby('clusterer')['average_daily_drive_time'].mean().sort_values()
            fig_clusterer_drive = go.Figure()
            fig_clusterer_drive.add_trace(go.Bar(
                x=clusterer_drive_avg.index,
                y=clusterer_drive_avg.values,
                marker_color=[clusterer_colors.get(c, '#cccccc') for c in clusterer_drive_avg.index],
                text=[f"{v:.2f}h" for v in clusterer_drive_avg.values],
                textposition='auto'
            ))

            fig_clusterer_drive.update_layout(
                xaxis_title='Clusterer',
                yaxis_title='Average Daily Drive Time (hrs)',
                height=300,
                showlegend=False
            )
            fig_clusterer_drive.update_yaxes(range=[0, None])

            st.plotly_chart(fig_clusterer_drive, use_container_width=True)

        with col2:
            # Create bar chart for balancers - daily drive time
            balancer_drive_avg = summary_pd.groupby('balancer')['average_daily_drive_time'].mean().sort_values()
            fig_balancer_drive = go.Figure()
            fig_balancer_drive.add_trace(go.Bar(
                x=balancer_drive_avg.index,
                y=balancer_drive_avg.values,
                marker_color='#7F9FFF',
                text=[f"{v:.2f}h" for v in balancer_drive_avg.values],
                textposition='auto'
            ))

            fig_balancer_drive.update_layout(
                xaxis_title='Balancer',
                yaxis_title='Average Daily Drive Time (hrs)',
                height=300,
                showlegend=False
            )
            fig_balancer_drive.update_yaxes(range=[0, None])

            st.plotly_chart(fig_balancer_drive, use_container_width=True)

    # data info
    st.subheader("Data Information")
    st.info(f"""
    **Total Algorithm Combinations**: {len(summary_pd)}

    **Data Source**: aggregate-summary.parquet

    **Last Updated**: {summary_pd['created_on'].max() if 'created_on' in summary_pd.columns else 'Unknown'}

    This page shows system-wide performance metrics aggregated across all zones for each algorithm combination.
    """)