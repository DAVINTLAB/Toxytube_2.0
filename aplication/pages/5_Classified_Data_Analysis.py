"""
Classified Data Analysis Page
Visualizes and analyzes the global dataset with classification results
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from components.navigation import render_navigation

# =============================================================================
# Initialize Session State - Analysis Data
# =============================================================================

if 'analysisData' not in st.session_state:
    st.session_state.analysisData = {
        'labelColumn': None,                # Column with classification labels
        'datetimeColumn': None,             # Datetime column (optional)
        'aggregation': 'day',               # Time aggregation level
    }

# Initialize global data if not exists
if 'globalData' not in st.session_state:
    st.session_state.globalData = {
        'dataset': None,
        'textColumn': None,
        'outputDirectory': os.path.expanduser('~/Downloads'),
        'outputFileName': '',
        'outputFormat': 'csv',
        'datasetLoaded': False,
        'originalFileName': ''
    }

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Classified Data Analysis - Toxicytube",
    page_icon="📊",
    layout="wide",
)

# Render navigation sidebar
render_navigation('analysis')

# =============================================================================
# Helper Functions
# =============================================================================

def parse_datetime_column(df, column_name):
    """Parse datetime column and handle various formats"""
    try:
        df = df.copy()
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')

        null_count = df[column_name].isnull().sum()
        if null_count > 0:
            st.warning(f"⚠️ {null_count} datetime values could not be converted and were ignored.")

        df = df.dropna(subset=[column_name])
        return df
    except Exception as e:
        st.error(f"❌ Error processing datetime column: {str(e)}")
        return None

def create_class_distribution_chart(df, label_column):
    """Create pie chart showing class distribution"""
    try:
        class_counts = df[label_column].value_counts()

        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title=f"Class Distribution - {label_column}",
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )

        fig.update_layout(
            showlegend=True,
            height=500,
            font=dict(size=12)
        )

        return fig, class_counts
    except Exception as e:
        st.error(f"❌ Error creating distribution chart: {str(e)}")
        return None, None

def create_temporal_analysis_chart(df, datetime_column, label_column, aggregation='day'):
    """Create temporal analysis chart showing class variation over time"""
    try:
        df_temp = df.copy()
        df_temp = df_temp.set_index(datetime_column)

        freq_map = {
            'hour': 'H',
            'day': 'D',
            'week': 'W',
            'month': 'M'
        }

        freq = freq_map.get(aggregation, 'D')
        grouped = df_temp.groupby([pd.Grouper(freq=freq), label_column]).size().unstack(fill_value=0)

        fig = go.Figure()

        colors = px.colors.qualitative.Set3
        for i, column in enumerate(grouped.columns):
            fig.add_trace(
                go.Scatter(
                    x=grouped.index,
                    y=grouped[column],
                    mode='lines+markers',
                    name=str(column),
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{column}</b><br>Date: %{{x}}<br>Count: %{{y}}<extra></extra>'
                )
            )

        fig.update_layout(
            title=f"Temporal Class Variation - Aggregation by {aggregation.title()}",
            xaxis_title="Date/Time",
            yaxis_title="Number of Records",
            height=500,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig, grouped
    except Exception as e:
        st.error(f"❌ Error creating temporal analysis: {str(e)}")
        return None, None

# =============================================================================
# Main Content
# =============================================================================

st.markdown("# 📊 Classified Data Analysis")
st.markdown("Visualize and analyze the global dataset with classification results.")

st.markdown("---")

# =============================================================================
# Global Dataset Preview
# =============================================================================

with st.container(border=True):
    st.markdown("### 📁 Global Dataset Preview")
    
    if st.session_state.globalData['datasetLoaded'] and st.session_state.globalData['dataset'] is not None:
        dataset = st.session_state.globalData['dataset']
        text_column = st.session_state.globalData['textColumn']
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Rows", f"{len(dataset):,}")
        with col2:
            st.metric("📋 Columns", len(dataset.columns))
        with col3:
            st.metric("📝 Text Column", text_column if text_column else "Not selected")
        with col4:
            st.metric("📄 Source", st.session_state.globalData['originalFileName'])
        
        # Preview
        st.markdown("**Dataset Preview (first 5 rows):**")
        st.dataframe(dataset.head(5), use_container_width=True)
        
        # Show all columns
        st.markdown("**Available Columns:**")
        st.write(", ".join([f"`{col}`" for col in dataset.columns]))
    else:
        st.warning("⚠️ **No dataset loaded.** Please upload a dataset in the Home page first.")
        st.info("👈 Go to **Home** to upload and configure your dataset.")

st.markdown("")

# =============================================================================
# Column Selection for Analysis
# =============================================================================

if st.session_state.globalData['datasetLoaded'] and st.session_state.globalData['dataset'] is not None:
    dataset = st.session_state.globalData['dataset']
    
    with st.container(border=True):
        st.markdown("### ⚙️ Analysis Configuration")
        st.markdown("Select the columns to use for analysis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Label/Classification Column:**")
            label_columns = dataset.columns.tolist()
            
            # Try to auto-detect label columns
            suggested_labels = [col for col in label_columns if any(
                keyword in col.lower() for keyword in ['label', 'class', 'predict', 'classification', 'toxic', 'sentiment']
            )]
            
            label_column = st.selectbox(
                "Select the column containing classification labels:",
                options=label_columns,
                index=label_columns.index(suggested_labels[0]) if suggested_labels else 0,
                help="This column should contain the classification results"
            )
            
            if label_column:
                st.session_state.analysisData['labelColumn'] = label_column
                unique_values = dataset[label_column].unique()
                st.info(f"📋 Found classes: {', '.join(map(str, unique_values[:10]))}{'...' if len(unique_values) > 10 else ''}")
        
        with col2:
            st.markdown("**Date/Time Column (Optional):**")
            
            # Try to auto-detect datetime columns
            datetime_columns = ['None'] + dataset.columns.tolist()
            
            datetime_column = st.selectbox(
                "Select the column containing date/time (optional for temporal analysis):",
                options=datetime_columns,
                index=0,
                help="This column should contain date/time information"
            )
            
            if datetime_column != 'None':
                st.session_state.analysisData['datetimeColumn'] = datetime_column
                sample_values = dataset[datetime_column].head(3).tolist()
                st.info(f"📅 Sample values: {', '.join(map(str, sample_values))}")
            else:
                st.session_state.analysisData['datetimeColumn'] = None

    st.markdown("")
    
    # =============================================================================
    # Class Distribution Analysis
    # =============================================================================
    
    if st.session_state.analysisData['labelColumn']:
        label_column = st.session_state.analysisData['labelColumn']
        
        with st.container(border=True):
            st.markdown("### 📊 Class Distribution")
            st.markdown("Visualize the proportion of each class in the dataset.")
            
            fig_pie, class_counts = create_class_distribution_chart(dataset, label_column)
            
            if fig_pie is not None:
                col_chart, col_stats = st.columns([3, 1])
                
                with col_chart:
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_stats:
                    st.markdown("**📊 Statistics:**")
                    total_samples = len(dataset)
                    
                    for label, count in class_counts.items():
                        percentage = (count / total_samples) * 100
                        st.metric(
                            label=f"🏷️ {label}",
                            value=f"{count:,}",
                            delta=f"{percentage:.1f}%"
                        )
        
        st.markdown("")
        
        # =============================================================================
        # Temporal Analysis (if datetime column selected)
        # =============================================================================
        
        if st.session_state.analysisData['datetimeColumn']:
            datetime_column = st.session_state.analysisData['datetimeColumn']
            
            with st.container(border=True):
                st.markdown("### ⏰ Temporal Analysis")
                st.markdown("Visualize how classes vary over time.")
                
                # Aggregation selector
                col_agg, col_info = st.columns([2, 8])
                
                with col_agg:
                    aggregation = st.selectbox(
                        "Time aggregation:",
                        options=['hour', 'day', 'week', 'month'],
                        index=1,
                        help="How to group data over time"
                    )
                    st.session_state.analysisData['aggregation'] = aggregation
                
                # Process datetime and create chart
                dataset_processed = parse_datetime_column(dataset.copy(), datetime_column)
                
                if dataset_processed is not None and len(dataset_processed) > 0:
                    with col_info:
                        min_date = dataset_processed[datetime_column].min()
                        max_date = dataset_processed[datetime_column].max()
                        date_range = max_date - min_date
                        st.info(f"📅 Data period: {min_date.strftime('%Y-%m-%d %H:%M')} to {max_date.strftime('%Y-%m-%d %H:%M')} ({date_range.days} days)")
                    
                    fig_temporal, grouped_data = create_temporal_analysis_chart(
                        dataset_processed, datetime_column, label_column, aggregation
                    )
                    
                    if fig_temporal is not None:
                        st.plotly_chart(fig_temporal, use_container_width=True)
                else:
                    st.warning("⚠️ Could not process datetime column for temporal analysis.")
        
        st.markdown("")
        
        # =============================================================================
        # Detailed Statistics
        # =============================================================================
        
        with st.container(border=True):
            st.markdown("### 📈 Detailed Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Class Distribution Table:**")
                stats_df = pd.DataFrame({
                    'Class': class_counts.index,
                    'Count': class_counts.values,
                    'Percentage': [f"{(c/len(dataset)*100):.2f}%" for c in class_counts.values]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Dataset Summary:**")
                st.metric("Total Records", f"{len(dataset):,}")
                st.metric("Unique Classes", len(class_counts))
                st.metric("Most Common Class", f"{class_counts.index[0]} ({class_counts.values[0]:,})")
                if len(class_counts) > 1:
                    st.metric("Least Common Class", f"{class_counts.index[-1]} ({class_counts.values[-1]:,})")

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("### 📋 About This Tool")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Classified Data Analysis** allows you to:
    
    - 📊 **Class Distribution**: Visualize proportions of each category
    - ⏰ **Temporal Analysis**: See how classes vary over time
    - 📈 **Trends**: Identify growing or declining patterns
    - 🔍 **Insights**: Discover activity peaks and time patterns
    """)

with col2:
    st.markdown("""
    **Tips:**
    
    - Use the global dataset configured in the Home page
    - Select the appropriate label column for analysis
    - Add a datetime column for temporal analysis
    - Results update automatically when the dataset changes
    """)
