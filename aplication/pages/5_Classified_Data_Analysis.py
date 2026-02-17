"""
Classified Data Analysis Page
Automatic visualization and analysis of classification results from all classifiers
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from components.navigation import render_navigation, get_configuration_status

# Initialize global data if not exists
if 'globalData' not in st.session_state:
    st.session_state.globalData = {
        'dataset': None,
        'textColumn': None,
        'outputDirectory': os.path.expanduser('~/Downloads'),
        'outputFileName': '',
        'outputFormat': None,
        'datasetLoaded': False,
        'originalFileName': ''
    }

# Initialize analysis state
if 'analysisState' not in st.session_state:
    st.session_state.analysisState = {
        'bert_threshold': 0.5,
        'detoxify_threshold': 0.5,
        'llm_confidence_filter': ['high', 'medium', 'low']
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
# Helper Functions - Column Parsing
# =============================================================================

def get_bert_columns(df):
    """
    Parse BERT classifier columns from dataframe.
    Returns dict with label names and their probability columns.
    """
    bert_cols = {}
    for col in df.columns:
        if col.startswith('bert_prob_'):
            label = col.replace('bert_prob_', '')
            bert_cols[label] = col
    return bert_cols

def get_detoxify_columns(df):
    """
    Parse Detoxify classifier columns from dataframe.
    Returns dict with label names and their probability columns.
    """
    detoxify_cols = {}
    for col in df.columns:
        if col.startswith('detoxify_prob_'):
            label = col.replace('detoxify_prob_', '')
            detoxify_cols[label] = col
    return detoxify_cols

def get_llm_columns(df):
    """
    Parse LLM classifier columns from dataframe.
    Returns dict with column names if they exist.
    """
    llm_cols = {}
    if 'llm_classification' in df.columns:
        llm_cols['classification'] = 'llm_classification'
    if 'llm_confidence' in df.columns:
        llm_cols['confidence'] = 'llm_confidence'
    if 'llm_reasoning' in df.columns:
        llm_cols['reasoning'] = 'llm_reasoning'
    return llm_cols

def has_classifier_data(df, classifier_type):
    """Check if dataframe has data from a specific classifier."""
    if classifier_type == 'bert':
        return len(get_bert_columns(df)) > 0
    elif classifier_type == 'detoxify':
        return len(get_detoxify_columns(df)) > 0
    elif classifier_type == 'llm':
        return len(get_llm_columns(df)) > 0
    return False

# =============================================================================
# Helper Functions - Threshold Curves
# =============================================================================

def create_threshold_chart(df, prob_columns, title, height=350):
    """
    Create threshold-based class distribution chart.
    Shows how many texts would be classified as each class at different probability thresholds.
    """
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
    
    # Generate thresholds from 0.0 to 1.0
    thresholds = np.linspace(0.0, 1.0, 101)
    
    for idx, (label, col) in enumerate(prob_columns.items()):
        counts_at_threshold = []
        
        for threshold in thresholds:
            count = (df[col] >= threshold).sum()
            counts_at_threshold.append(count)
        
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=counts_at_threshold,
            mode='lines',
            name=label,
            line=dict(color=color, width=2),
            hovertemplate=f'<b>{label}</b><br>Threshold: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title='Probability Threshold',
        yaxis_title='Number of Texts',
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        xaxis=dict(range=[0, 1])
    )
    
    return fig

def create_confidence_donut(df, llm_cols, title, height=350):
    """
    Create donut chart showing LLM confidence distribution.
    """
    if 'confidence' not in llm_cols:
        return None
    
    confidence_counts = df[llm_cols['confidence']].value_counts()
    
    # Define colors for confidence levels
    color_map = {
        'high': '#2ecc71',
        'medium': '#f39c12', 
        'low': '#e74c3c'
    }
    colors = [color_map.get(str(c).lower(), '#95a5a6') for c in confidence_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=confidence_counts.index,
        values=confidence_counts.values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo='percent+label',
        textfont=dict(size=11)
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        annotations=[dict(
            text=f'{len(df)}<br>texts',
            x=0.5, y=0.5,
            font_size=12,
            showarrow=False
        )]
    )
    
    return fig

def create_class_distribution_donut(df, prob_columns, threshold, title, height=350):
    """
    Create donut chart showing class distribution based on threshold.
    Classes are determined by probability >= threshold.
    """
    class_counts = {}
    
    for label, col in prob_columns.items():
        count = (df[col] >= threshold).sum()
        if count > 0:
            class_counts[label] = count
    
    if not class_counts:
        # If no class meets threshold, show "Below threshold"
        class_counts = {'Below threshold': len(df)}
    
    labels = list(class_counts.keys())
    values = list(class_counts.values())
    
    colors = px.colors.qualitative.Set3[:len(labels)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo='percent+label',
        textfont=dict(size=10)
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        annotations=[dict(
            text=f'≥{threshold:.2f}',
            x=0.5, y=0.5,
            font_size=12,
            showarrow=False
        )]
    )
    
    return fig

def create_llm_class_distribution_donut(df, llm_cols, confidence_filter, title, height=350):
    """
    Create donut chart showing LLM classification distribution filtered by confidence.
    """
    if 'classification' not in llm_cols or 'confidence' not in llm_cols:
        return None
    
    # Filter by confidence
    filtered_df = df[df[llm_cols['confidence']].str.lower().isin([c.lower() for c in confidence_filter])]
    
    if len(filtered_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data matches filter",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            height=height,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    
    class_counts = filtered_df[llm_cols['classification']].value_counts()
    
    colors = px.colors.qualitative.Set3[:len(class_counts)]
    
    fig = go.Figure(data=[go.Pie(
        labels=class_counts.index,
        values=class_counts.values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo='percent+label',
        textfont=dict(size=10)
    )])
    
    confidence_str = ', '.join(confidence_filter)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        annotations=[dict(
            text=f'{len(filtered_df)}<br>texts',
            x=0.5, y=0.5,
            font_size=12,
            showarrow=False
        )]
    )
    
    return fig

# =============================================================================
# Main Content
# =============================================================================

st.markdown("# 📊 Classified Data Analysis")
st.markdown("Automatic visualization and comparison of classification results from BERT, Detoxify, and LLM classifiers.")

st.markdown("---")

# =============================================================================
# Global Dataset Preview
# =============================================================================

with st.container(border=True):
    st.markdown("### 📁 Dataset Preview")

    config_status = get_configuration_status()
    if config_status['complete']:
        dataset = st.session_state.globalData['dataset']
        text_column = st.session_state.globalData['textColumn']

        # Custom CSS for smaller metrics
        st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 20px;
            }
            [data-testid="stMetricLabel"] {
                font-size: 14px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Dataset statistics
        col1, col2, col3, spacer, col4, col5 = st.columns([1, 1, 1, 0.5, 1.5, 1.5])

        with col1:
            st.metric("📊 Rows", f"{len(dataset):,}")
        with col2:
            st.metric("📋 Columns", len(dataset.columns))
        with col3:
            # Calculate dataset size in memory
            size_bytes = dataset.memory_usage(deep=True).sum()
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024**2:
                size_str = f"{size_bytes/1024:.2f} KB"
            elif size_bytes < 1024**3:
                size_str = f"{size_bytes/(1024**2):.2f} MB"
            else:
                size_str = f"{size_bytes/(1024**3):.2f} GB"
            st.metric("💾 Size", size_str)

        with col4:
            st.metric("📄 Source", st.session_state.globalData['originalFileName'])
        with col5:
            st.metric("📝 Text Column", text_column if text_column else "Not selected")

        # Preview
        st.markdown("**Dataset Preview (first 5 rows):**")
        st.dataframe(dataset.head(5), use_container_width=True)

        # Show all columns
        st.markdown("**Available Columns:**")
        st.write(", ".join([f"`{col}`" for col in dataset.columns]))
    else:
        st.warning("⚠️ **No dataset loaded.** Please upload a dataset in the Home page first.")

st.markdown("")

# =============================================================================
# Classification Comparison Section
# =============================================================================

config_status = get_configuration_status()
if config_status['complete']:
    dataset = st.session_state.globalData['dataset']
    
    # Parse columns for each classifier
    bert_cols = get_bert_columns(dataset)
    detoxify_cols = get_detoxify_columns(dataset)
    llm_cols = get_llm_columns(dataset)
    
    has_bert = len(bert_cols) > 0
    has_detoxify = len(detoxify_cols) > 0
    has_llm = len(llm_cols) > 0
    
    # Check if any classifier data exists
    if not has_bert and not has_detoxify and not has_llm:
        st.warning("⚠️ **No classification data found.** Please run at least one classifier (BERT, Detoxify, or LLM) before using this analysis page.")
    else:
        with st.container(border=True):
            st.markdown("### 📈 Classification Comparison")
            st.markdown("Compare results from all available classifiers side by side.")
            
            # Show which classifiers are available
            available_classifiers = []
            if has_bert:
                available_classifiers.append("🤖 BERT")
            if has_detoxify:
                available_classifiers.append("🛡️ Detoxify")
            if has_llm:
                available_classifiers.append("🧠 LLM")
            
            st.info(f"**Available classifiers:** {', '.join(available_classifiers)}")
            
            st.markdown("---")
            
            # =============================================================================
            # Row 1: Threshold Curves / Confidence Distribution
            # =============================================================================
            
            st.markdown("#### 📊 Row 1: Threshold-based Class Distribution")
            
            col_bert, col_detoxify, col_llm = st.columns(3)
            
            with col_bert:
                st.markdown("##### 🤖 BERT Classifier")
                if has_bert:
                    fig_threshold_bert = create_threshold_chart(
                        dataset, bert_cols, 
                        "Texts per Class at Threshold - BERT"
                    )
                    st.plotly_chart(fig_threshold_bert, use_container_width=True)
                else:
                    st.info("No BERT classification data available.")
            
            with col_detoxify:
                st.markdown("##### 🛡️ Detoxify Classifier")
                if has_detoxify:
                    fig_threshold_detoxify = create_threshold_chart(
                        dataset, detoxify_cols,
                        "Texts per Class at Threshold - Detoxify"
                    )
                    st.plotly_chart(fig_threshold_detoxify, use_container_width=True)
                else:
                    st.info("No Detoxify classification data available.")
            
            with col_llm:
                st.markdown("##### 🧠 LLM Classifier")
                if has_llm and 'confidence' in llm_cols:
                    fig_confidence = create_confidence_donut(
                        dataset, llm_cols,
                        "Confidence Distribution - LLM"
                    )
                    if fig_confidence:
                        st.plotly_chart(fig_confidence, use_container_width=True)
                else:
                    st.info("No LLM classification data available.")
            
            st.markdown("---")
            
            # =============================================================================
            # Row 2: Threshold / Filter Selection
            # =============================================================================
            
            st.markdown("#### 🎚️ Row 2: Threshold & Filter Selection")
            
            col_bert_thresh, col_detoxify_thresh, col_llm_filter = st.columns(3)
            
            with col_bert_thresh:
                st.markdown("##### 🤖 BERT Threshold")
                if has_bert:
                    bert_threshold = st.slider(
                        "Select probability threshold:",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.analysisState.get('bert_threshold', 0.5),
                        step=0.05,
                        key="bert_threshold_slider",
                        help="Only samples with probability >= threshold will be shown"
                    )
                    st.session_state.analysisState['bert_threshold'] = bert_threshold
                else:
                    st.info("No BERT data to filter.")
                    bert_threshold = 0.5
            
            with col_detoxify_thresh:
                st.markdown("##### 🛡️ Detoxify Threshold")
                if has_detoxify:
                    detoxify_threshold = st.slider(
                        "Select probability threshold:",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.analysisState.get('detoxify_threshold', 0.5),
                        step=0.05,
                        key="detoxify_threshold_slider",
                        help="Only samples with probability >= threshold will be shown"
                    )
                    st.session_state.analysisState['detoxify_threshold'] = detoxify_threshold
                else:
                    st.info("No Detoxify data to filter.")
                    detoxify_threshold = 0.5
            
            with col_llm_filter:
                st.markdown("##### 🧠 LLM Confidence Filter")
                if has_llm and 'confidence' in llm_cols:
                    # Get unique confidence levels
                    unique_confidences = dataset[llm_cols['confidence']].dropna().unique()
                    confidence_options = [str(c) for c in unique_confidences]
                    
                    # Default to all selected
                    default_selection = st.session_state.analysisState.get('llm_confidence_filter', confidence_options)
                    
                    # Multi-select buttons using checkboxes
                    st.markdown("Select confidence levels:")
                    
                    selected_confidences = []
                    cols_check = st.columns(len(confidence_options) if len(confidence_options) <= 3 else 3)
                    
                    for i, conf in enumerate(confidence_options):
                        col_idx = i % 3
                        with cols_check[col_idx]:
                            # Color coding for confidence
                            if conf.lower() == 'high':
                                emoji = "🟢"
                            elif conf.lower() == 'medium':
                                emoji = "🟡"
                            else:
                                emoji = "🔴"
                            
                            is_selected = st.checkbox(
                                f"{emoji} {conf.capitalize()}",
                                value=conf in default_selection,
                                key=f"llm_conf_{conf}"
                            )
                            if is_selected:
                                selected_confidences.append(conf)
                    
                    st.session_state.analysisState['llm_confidence_filter'] = selected_confidences
                    llm_confidence_filter = selected_confidences
                else:
                    st.info("No LLM data to filter.")
                    llm_confidence_filter = []
            
            st.markdown("---")
            
            # =============================================================================
            # Row 3: Class Distribution Donuts (Filtered)
            # =============================================================================
            
            st.markdown("#### 🍩 Row 3: Class Distribution (Filtered)")
            
            col_bert_dist, col_detoxify_dist, col_llm_dist = st.columns(3)
            
            with col_bert_dist:
                st.markdown("##### 🤖 BERT Classes")
                if has_bert:
                    fig_bert_dist = create_class_distribution_donut(
                        dataset, bert_cols, bert_threshold,
                        f"BERT Classes (≥{bert_threshold:.2f})"
                    )
                    st.plotly_chart(fig_bert_dist, use_container_width=True)
                    
                    # Show count details
                    total_above = sum((dataset[col] >= bert_threshold).sum() for col in bert_cols.values())
                    st.caption(f"Total samples above threshold: {total_above:,}")
                else:
                    st.info("No BERT classification data available.")
            
            with col_detoxify_dist:
                st.markdown("##### 🛡️ Detoxify Classes")
                if has_detoxify:
                    fig_detoxify_dist = create_class_distribution_donut(
                        dataset, detoxify_cols, detoxify_threshold,
                        f"Detoxify Classes (≥{detoxify_threshold:.2f})"
                    )
                    st.plotly_chart(fig_detoxify_dist, use_container_width=True)
                    
                    # Show count details
                    total_above = sum((dataset[col] >= detoxify_threshold).sum() for col in detoxify_cols.values())
                    st.caption(f"Total samples above threshold: {total_above:,}")
                else:
                    st.info("No Detoxify classification data available.")
            
            with col_llm_dist:
                st.markdown("##### 🧠 LLM Classes")
                if has_llm and 'classification' in llm_cols and llm_confidence_filter:
                    fig_llm_dist = create_llm_class_distribution_donut(
                        dataset, llm_cols, llm_confidence_filter,
                        f"LLM Classes ({', '.join(llm_confidence_filter)})"
                    )
                    if fig_llm_dist:
                        st.plotly_chart(fig_llm_dist, use_container_width=True)
                    
                    # Show count details
                    filtered_count = len(dataset[dataset[llm_cols['confidence']].str.lower().isin([c.lower() for c in llm_confidence_filter])])
                    st.caption(f"Samples matching filter: {filtered_count:,}")
                elif has_llm and not llm_confidence_filter:
                    st.warning("Please select at least one confidence level.")
                else:
                    st.info("No LLM classification data available.")

