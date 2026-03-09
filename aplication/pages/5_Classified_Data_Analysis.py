"""
Classified Data Analysis Page
Automatic visualization and analysis of classification results from all classifiers
"""
import streamlit as st
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.graph_objects as go

# Garantir que stopwords em português estejam disponíveis
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from components.navigation import render_navigation, get_configuration_status
from components.analysis_backend import (
    filter_fully_classified_texts,
    get_bert_columns,
    get_detoxify_columns,
    get_llm_columns,
    has_classifier_data,
    create_threshold_chart,
    create_confidence_donut,
    create_class_distribution_donut,
    create_bert_argmax_distribution_donut,
    create_llm_class_distribution_donut,
    detect_likes_column,
    detect_author_column,
    detect_reply_count_column,
    get_probability_value_color,
    style_probability_columns,
    create_top_comments_table,
    create_bert_llm_agreement_table,
    parse_datetime_column,
    create_time_series_chart,
    default_palette,
    _label_color_map,
    _color_idx,
    sentiment_color_map
)

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
        'llm_confidence_filter': ['high', 'medium', 'low'],
        'bert_use_argmax': False,
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
# Main Content
# =============================================================================

st.markdown("# 📊 Classified Data Analysis")
st.markdown("Comparative visualization of texts classified by **all three models** (BERT, Detoxify, and LLM).")

st.markdown("---")

# =============================================================================
# Dataset Filtering and Statistics
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

    # Check if all three classifiers have data
    if not (has_bert and has_detoxify and has_llm):
        st.warning("⚠️ **This analysis page requires data from all three classifiers (BERT, Detoxify, and LLM).**")
        st.info("Please run all three classifiers before accessing this page.")

        missing = []
        if not has_bert:
            missing.append("🤖 BERT Classifier")
        if not has_detoxify:
            missing.append("🛡️ Detoxify Classifier")
        if not has_llm:
            missing.append("🧠 LLM Classifier")

        if missing:
            st.markdown("**Missing classifiers:**")
            for m in missing:
                st.markdown(f"- {m}")
    else:
        # Filter dataset to include only fully classified texts
        filtered_dataset, stats = filter_fully_classified_texts(dataset, bert_cols, detoxify_cols, llm_cols)

        # Display statistics
        with st.container(border=True):
            st.markdown("### 📋 Classification Coverage")
            st.markdown("This page analyzes only texts that were classified by **all three models**.")

            # Custom CSS for metrics
            st.markdown("""
                <style>
                [data-testid="stMetricValue"] {
                    font-size: 22px;
                }
                [data-testid="stMetricLabel"] {
                    font-size: 14px;
                }
                </style>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("📊 Total Texts", f"{stats['total']:,}")

            with col2:
                bert_pct = (stats['bert_classified'] / stats['total'] * 100) if stats['total'] > 0 else 0
                st.metric("🤖 BERT", f"{stats['bert_classified']:,} ({bert_pct:.1f}%)")

            with col3:
                detox_pct = (stats['detoxify_classified'] / stats['total'] * 100) if stats['total'] > 0 else 0
                st.metric("🛡️ Detoxify", f"{stats['detoxify_classified']:,} ({detox_pct:.1f}%)")

            with col4:
                llm_pct = (stats['llm_classified'] / stats['total'] * 100) if stats['total'] > 0 else 0
                st.metric("🧠 LLM", f"{stats['llm_classified']:,} ({llm_pct:.1f}%)")

            with col5:
                all_pct = (stats['all_classified'] / stats['total'] * 100) if stats['total'] > 0 else 0
                st.metric("All 3 Models", f"{stats['all_classified']:,} ({all_pct:.1f}%)")

            # Warning if coverage is low
            if all_pct < 50:
                st.warning(f"⚠️ Only {all_pct:.1f}% of texts were classified by all three models. Consider classifying more texts for better analysis coverage.")
            elif stats['all_classified'] == 0:
                st.error("❌ No texts were classified by all three models. Please ensure you run all classifiers on the same dataset.")
            else:
                st.success(f"✅ {stats['all_classified']:,} texts available for comparative analysis.")

        st.markdown("")

        # Only proceed if we have texts classified by all models
        if stats['all_classified'] > 0:
            # Use filtered dataset for all subsequent visualizations
            dataset = filtered_dataset

        # =============================================================================
        # Agreement: BERT vs LLM Section
        # =============================================================================

        st.markdown("")

        with st.container(border=True):
            st.markdown("### 🤝 BERT & LLM Agreement")
            st.markdown("Show comments where BERT's final class (argmax) matches the LLM classification.")

            # Use same detection as before
            likes_col = detect_likes_column(dataset)
            reply_col = detect_reply_count_column(dataset)
            author_col = detect_author_column(dataset)
            text_column = st.session_state.globalData['textColumn']

            # Compute agreement over full dataset for metrics and donuts
            try:
                df_agree_full, agree_err_full = create_bert_llm_agreement_table(
                    dataset,
                    text_column,
                    likes_col,
                    author_col,
                    reply_col,
                    bert_cols,
                    detoxify_cols,
                    llm_cols,
                    sort_by='likes',
                    top_n=len(dataset)
                )

                if agree_err_full:
                    agreement_pct = None
                    total_agree = 0
                    total_rows = 0
                    total_dataset_rows = len(dataset) if dataset is not None else 0
                else:
                    # All texts in filtered dataset are fully classified
                    total_dataset_rows = len(dataset) if dataset is not None else 0
                    total_rows = total_dataset_rows

                    total_agree = len(df_agree_full) if df_agree_full is not None else 0
                    agreement_pct = (total_agree / total_rows * 100) if total_rows > 0 else 0.0
            except Exception:
                agreement_pct = None
                total_agree = 0
                total_rows = 0
                total_dataset_rows = len(dataset) if dataset is not None else 0

            # --- Class distribution donuts (two donuts: LLM and BERT) ---
            try:
                # Agreement counts per class (where BERT == LLM)
                agree_counts = None
                if df_agree_full is not None and not df_agree_full.empty:
                    agree_counts = df_agree_full['🤖 BERT Final'].dropna().astype(str).value_counts()

                # LLM class distribution (all dataset)
                llm_counts = None
                if 'classification' in llm_cols and llm_cols['classification'] in dataset.columns:
                    llm_counts = dataset[llm_cols['classification']].dropna().astype(str).value_counts()

                # BERT class distribution (compute final class for all rows)
                bert_counts = None
                bert_prob_cols = [col for col in bert_cols.values() if col in dataset.columns]
                if bert_prob_cols:
                    # map column name to label
                    col_to_label = {col: label for label, col in bert_cols.items() if col in dataset.columns}
                    # idxmax returns the column name with highest value per row
                    try:
                        bert_argmax_cols = dataset[bert_prob_cols].idxmax(axis=1)
                        bert_final_all = bert_argmax_cols.map(lambda c: col_to_label.get(c) if c in col_to_label else None)
                        bert_counts = bert_final_all.dropna().astype(str).value_counts()
                    except Exception:
                        bert_counts = None

                # Render two donuts side-by-side (LLM, BERT) — Agreement-by-Class donut removed
                try:
                    if (llm_counts is not None and not llm_counts.empty) or (bert_counts is not None and not bert_counts.empty):
                        col_left, col_right = st.columns(2)

                        def _make_donut(counts, title, col):
                            if counts is None or counts.empty:
                                with col:
                                    st.info(f"No data for {title}")
                                return
                            labels = counts.index.tolist()
                            values = counts.values.tolist()

                            # Build colors using global maps defined at module level if present
                            colors = []
                            for lbl in labels:
                                key = str(lbl).strip()
                                low = key.lower()
                                if low in sentiment_color_map:
                                    colors.append(sentiment_color_map[low])
                                else:
                                    if key in _label_color_map:
                                        colors.append(_label_color_map[key])
                                    else:
                                        c = default_palette[_color_idx['i'] % len(default_palette)]
                                        _label_color_map[key] = c
                                        colors.append(c)
                                        _color_idx['i'] += 1

                            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, marker=dict(colors=colors), textinfo='percent+label')])
                            fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=True)
                            with col:
                                st.plotly_chart(fig, use_container_width=True)

                        _make_donut(bert_counts, 'BERT Class Distribution', col_left)
                        _make_donut(llm_counts, 'LLM Class Distribution', col_right)
                except Exception:
                    pass
            except Exception:
                pass

            # --- Controls: Agreement metric, Show Top, Sort By, Class filter (placed BELOW the donuts) ---
            col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])

            with col_a:
                # Show agreement as percentage and absolute counts
                if agreement_pct is None:
                    st.metric("Agreement", "N/A")
                else:
                    st.metric("Agreement*", f"{agreement_pct:.2f}% ({total_agree:,}/{total_rows:,})")

            with col_b:
                agree_top_n = st.selectbox(
                    "Show top:",
                    options=[10, 20, 50, 100],
                    index=1,
                    key="agree_top_n"
                )

            with col_c:
                agree_sort_options = []
                if likes_col:
                    agree_sort_options.append('Likes')
                if reply_col:
                    agree_sort_options.append('Replies')

                if len(agree_sort_options) > 0:
                    agree_sort_choice = st.selectbox(
                        "Sort by:",
                        options=agree_sort_options,
                        index=0,
                        key="agree_sort_by"
                    )
                    agree_sort_by = 'likes' if agree_sort_choice == 'Likes' else 'replies'
                else:
                    agree_sort_by = 'likes'

            with col_d:
                # Class filter for the agreement table
                class_options = ['All']
                if df_agree_full is not None and not df_agree_full.empty:
                    classes = df_agree_full['🤖 BERT Final'].dropna().astype(str).unique().tolist()
                    class_options += sorted(classes)
                selected_agree_class = st.selectbox('Filter class:', options=class_options, index=0, key='agree_class_filter')

            # Create the agreement table using the chosen top_n and sort_by
            df_agree, agree_error = create_bert_llm_agreement_table(
                dataset,
                text_column,
                likes_col,
                author_col,
                reply_col,
                bert_cols,
                detoxify_cols,
                llm_cols,
                sort_by=agree_sort_by,
                top_n=agree_top_n
            )

            if agree_error:
                st.warning(f"⚠️ {agree_error}")
            else:
                if df_agree is None or df_agree.empty:
                    st.info("No comments where BERT final class matches LLM classification were found.")
                else:
                    # Apply class filter if selected
                    if selected_agree_class and selected_agree_class != 'All':
                        if '🤖 BERT Final' in df_agree.columns:
                            df_agree = df_agree[df_agree['🤖 BERT Final'].astype(str) == selected_agree_class]

                    # Display simple table
                    st.markdown("**Comments with agreement (sample):**")

                    # Apply heatmap styling to BERT and Detoxify probability columns (if present)
                    bert_prob_cols = [col for col in df_agree.columns if col.startswith('🤖 ') and col != '🤖 BERT Final']
                    detox_prob_cols = [col for col in df_agree.columns if col.startswith('🛡️ ')]
                    all_prob_cols = bert_prob_cols + detox_prob_cols

                    if all_prob_cols:
                        # Formatting for numeric probability columns (2 decimal places)
                        def format_prob(val):
                            if isinstance(val, (int, float)) and not pd.isna(val):
                                return f'{val:.2f}'
                            return val

                        format_dict = {col: format_prob for col in all_prob_cols if col in df_agree.columns}

                        # Prepare style DataFrame (same shape) to apply a single unified styling pass.
                        styles_df = pd.DataFrame('', index=df_agree.index, columns=df_agree.columns)

                        # Blue gradient for all probability columns (BERT + Detoxify)
                        for col in all_prob_cols:
                            if col in df_agree.columns:
                                # apply color per cell using get_probability_value_color
                                for idx, val in df_agree[col].items():
                                    try:
                                        if isinstance(val, (int, float)) and 0 <= val <= 1:
                                            styles_df.at[idx, col] = get_probability_value_color(val)
                                        else:
                                            styles_df.at[idx, col] = ''
                                    except Exception:
                                        styles_df.at[idx, col] = ''

                        # Detect sentiment labels (case-insensitive) in BERT and LLM columns
                        bert_col = '🤖 BERT Final'
                        llm_col = '🧠 LLM Class'
                        sentiment_vals = {'negative', 'neutral', 'positive'}

                        try:
                            bert_vals = set(v.lower() for v in df_agree[bert_col].dropna().astype(str).unique()) if bert_col in df_agree.columns else set()
                        except Exception:
                            bert_vals = set()
                        try:
                            llm_vals = set(v.lower() for v in df_agree[llm_col].dropna().astype(str).unique()) if llm_col in df_agree.columns else set()
                        except Exception:
                            llm_vals = set()

                        # Decide if sentiment coloring should be applied (if either column contains sentiment labels)
                        apply_sentiment = (bert_vals and bert_vals.issubset(sentiment_vals)) or (llm_vals and llm_vals.issubset(sentiment_vals))

                        sentiment_color_map = {
                            'negative': 'background-color: #e74c3c; color: white',
                            'neutral':  'background-color: #f1c40f; color: black',
                            'positive': 'background-color: #2ecc71; color: white'
                        }

                        if apply_sentiment:
                            # Compute per-row sentiment preference: prefer BERT, fallback to LLM
                            row_sentiments = []
                            for _, row in df_agree.iterrows():
                                s = None
                                if bert_col in df_agree.columns and pd.notna(row.get(bert_col)):
                                    v = str(row.get(bert_col)).lower()
                                    if v in sentiment_vals:
                                        s = v
                                if s is None and llm_col in df_agree.columns and pd.notna(row.get(llm_col)):
                                    v = str(row.get(llm_col)).lower()
                                    if v in sentiment_vals:
                                        s = v
                                row_sentiments.append(s)

                            # Apply sentiment color to ALL non-probability cells
                            non_prob_cols = [c for c in df_agree.columns if c not in all_prob_cols]
                            for col in non_prob_cols:
                                for idx, s in enumerate(row_sentiments):
                                    if s:
                                        styles_df.iat[idx, df_agree.columns.get_loc(col)] = sentiment_color_map.get(s, '')

                        # Apply unified styles and formatting
                        styled_df = df_agree.style.apply(lambda _: styles_df, axis=None).format(format_dict)

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            height=min(400 + (agree_top_n * 10), 800),
                            hide_index=True
                        )
                    else:
                        st.dataframe(df_agree, use_container_width=True, height=min(400 + (agree_top_n * 10), 800), hide_index=True)

        # =============================================================================
        # Detoxify Analysis Section
        # =============================================================================

        st.markdown("")

        with st.container(border=True):
            st.markdown("### 🛡️ Detoxify Classification Analysis")
            st.markdown("Analyze how Detoxify classifications vary with different probability thresholds.")

            if has_detoxify and detoxify_cols:
                # 1. Threshold variation chart (line chart)
                st.markdown("#### 📈 Class Distribution by Threshold")
                st.markdown("Shows how many texts are classified in each class at different probability thresholds.")

                # Create threshold chart
                threshold_fig = create_threshold_chart(
                    dataset,
                    detoxify_cols,
                    "Detoxify: Number of Texts per Class vs Threshold",
                    height=400
                )
                st.plotly_chart(threshold_fig, use_container_width=True)

                st.markdown("---")

                # 2. Threshold slider
                st.markdown("#### 🎚️ Select Threshold for Distribution")
                detoxify_threshold = st.slider(
                    "Probability Threshold:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="detoxify_analysis_threshold",
                    help="Texts with probability >= this threshold will be counted in each class"
                )

                st.markdown("---")

                # 3. Bar chart showing distribution at selected threshold
                st.markdown(f"#### 📊 Class Distribution at Threshold = {detoxify_threshold:.2f}")

                # Calculate counts for each class at the selected threshold
                class_counts = {}
                for label, col in detoxify_cols.items():
                    if col in dataset.columns:
                        # Convert column to numeric to handle string values
                        numeric_col = pd.to_numeric(dataset[col], errors='coerce')
                        count = (numeric_col >= detoxify_threshold).sum()
                        if count > 0:
                            class_counts[label] = count

                if class_counts:
                    # Create bar chart — sorted by count descending
                    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                    labels = [item[0] for item in sorted_items]
                    values = [item[1] for item in sorted_items]

                    # Build colors using shared palette and sentiment overrides
                    colors = []
                    for lbl in labels:
                        key = str(lbl).strip()
                        low = key.lower()
                        if low in sentiment_color_map:
                            colors.append(sentiment_color_map[low])
                            _label_color_map[key] = sentiment_color_map[low]
                        else:
                            if key in _label_color_map:
                                colors.append(_label_color_map[key])
                            else:
                                c = default_palette[_color_idx['i'] % len(default_palette)]
                                _label_color_map[key] = c
                                colors.append(c)
                                _color_idx['i'] += 1

                    bar_fig = go.Figure(data=[
                        go.Bar(
                            x=labels,
                            y=values,
                            marker=dict(color=colors),
                            text=values,
                            textposition='auto',
                            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                        )
                    ])

                    bar_fig.update_layout(
                        title=dict(text=f"Number of Texts per Class (Threshold ≥ {detoxify_threshold:.2f})", font=dict(size=14)),
                        xaxis_title="Class",
                        yaxis_title="Number of Texts",
                        height=400,
                        margin=dict(l=40, r=20, t=60, b=40),
                        showlegend=False
                    )

                    st.plotly_chart(bar_fig, use_container_width=True)

                else:
                    st.info(f"No texts have probabilities ≥ {detoxify_threshold:.2f} for any class.")

            else:
                st.warning("⚠️ No Detoxify classification data available.")

        # =============================================================================
        # Word Cloud Section
        # =============================================================================

        st.markdown("")

        with st.container(border=True):
            st.markdown("### ☁️ Word Cloud")
            st.markdown("Visualize the most frequent words in the comments.")

            text_column = st.session_state.globalData['textColumn']

            if text_column and text_column in dataset.columns:
                # Filter options
                col1, col2 = st.columns(2)

                with col1:
                    max_words = st.slider(
                        "Maximum words:",
                        min_value=10,
                        max_value=200,
                        value=100,
                        step=10,
                        key="wordcloud_max_words"
                    )

                with col2:
                    min_word_length = st.slider(
                        "Minimum word length:",
                        min_value=1,
                        max_value=10,
                        value=3,
                        step=1,
                        key="wordcloud_min_length"
                    )

                # Checkbox to use global dataset
                st.markdown("")
                use_global_dataset = st.checkbox(
                    "Use complete dataset (all texts, not just those classified by all 3 models)",
                    value=True,
                    key="wordcloud_use_global",
                    help="When checked, uses all texts from the original dataset instead of only those classified by BERT, Detoxify, and LLM."
                )

                # Select which dataset to use
                if use_global_dataset:
                    wordcloud_dataset = st.session_state.globalData['dataset']
                    st.info(f"💡 Using complete dataset with {len(wordcloud_dataset):,} texts.")
                else:
                    wordcloud_dataset = dataset
                    st.info(f"💡 Using filtered dataset with {len(wordcloud_dataset):,} texts (classified by all 3 models).")

                # Palavras extras a remover (complementa NLTK)
                with st.expander("🚫 Words to remove from cloud (optional)", expanded=False):
                    st.markdown("Enter comma-separated words to exclude from the word cloud (in addition to NLTK stopwords).")
                    custom_stopwords_input = st.text_area(
                        "Words to remove:",
                        placeholder="E.g.: video, channel, like, subscribe, thanks, ...",
                        height=80,
                        key="wordcloud_custom_stopwords"
                    )
                    custom_stopwords_set = set(
                        w.strip().lower() for w in custom_stopwords_input.split(",") if w.strip()
                    ) if custom_stopwords_input else set()

                # Initialize filtered texts
                filtered_texts = wordcloud_dataset[text_column].astype(str).tolist()
                filter_applied = False
                combined_mask = None

                # Only show classification filters if using filtered dataset
                if not use_global_dataset:
                    # Optional: Filter by classification
                    st.markdown("**Optional: Filter by classification**")

                    filter_col1, filter_col2, filter_col3 = st.columns(3)

                    with filter_col1:
                        if has_bert and bert_cols:
                            bert_filter_classes = st.multiselect(
                                "🤖 BERT class filter:",
                                options=list(bert_cols.keys()),
                                default=[],
                                key="wordcloud_bert_filter"
                            )
                            if bert_filter_classes:
                                bert_threshold_wc = st.slider(
                                    "BERT threshold:",
                                    0.0, 1.0, 0.5, 0.05,
                                    key="wordcloud_bert_threshold"
                                )
                                # Create mask for any selected class meeting threshold
                                bert_mask = pd.Series([False] * len(wordcloud_dataset))
                                for cls in bert_filter_classes:
                                    col = bert_cols[cls]
                                    # Convert column to numeric to handle string values
                                    numeric_col = pd.to_numeric(wordcloud_dataset[col], errors='coerce')
                                    class_mask = numeric_col >= bert_threshold_wc
                                    bert_mask = bert_mask | class_mask
                                combined_mask = bert_mask if combined_mask is None else combined_mask & bert_mask
                                filter_applied = True

                    with filter_col2:
                        if has_detoxify and detoxify_cols:
                            detox_filter_classes = st.multiselect(
                                "🛡️ Detoxify class filter:",
                                options=list(detoxify_cols.keys()),
                                default=[],
                                key="wordcloud_detox_filter"
                            )
                            if detox_filter_classes:
                                detox_threshold_wc = st.slider(
                                    "Detoxify threshold:",
                                    0.0, 1.0, 0.5, 0.05,
                                    key="wordcloud_detox_threshold"
                                )
                                # Create mask for any selected class meeting threshold
                                detox_mask = pd.Series([False] * len(wordcloud_dataset))
                                for cls in detox_filter_classes:
                                    col = detoxify_cols[cls]
                                    # Convert column to numeric to handle string values
                                    numeric_col = pd.to_numeric(wordcloud_dataset[col], errors='coerce')
                                    class_mask = numeric_col >= detox_threshold_wc
                                    detox_mask = detox_mask | class_mask
                                combined_mask = detox_mask if combined_mask is None else combined_mask & detox_mask
                                filter_applied = True

                    with filter_col3:
                        if has_llm and 'classification' in llm_cols:
                            llm_classes_available = list(wordcloud_dataset[llm_cols['classification']].dropna().unique())
                            llm_filter_classes = st.multiselect(
                                "🧠 LLM class filter:",
                                options=llm_classes_available,
                                default=[],
                                key="wordcloud_llm_filter"
                            )
                            if llm_filter_classes:
                                llm_mask = wordcloud_dataset[llm_cols['classification']].isin(llm_filter_classes)
                                combined_mask = llm_mask if combined_mask is None else combined_mask & llm_mask
                                filter_applied = True

                    # Apply combined filter
                    if filter_applied and combined_mask is not None:
                        filtered_texts = wordcloud_dataset[combined_mask][text_column].astype(str).tolist()

                # Generate word cloud
                if st.button("Generate Word Cloud", use_container_width=True, type="primary", key="generate_wordcloud"):
                    if len(filtered_texts) > 0:
                        with st.spinner("Generating word cloud..."):
                            # Stopwords em português + palavras customizadas do usuário
                            stop_words = set(stopwords.words('portuguese')) | custom_stopwords_set
                            palavras = []

                            for msg in filtered_texts:
                                msg = str(msg).lower()
                                msg = re.sub(r':[a-zA-Z0-9_]+:', '', msg)  # remove emojis :nome:
                                msg = re.sub(r'http\S+|www\S+|https\S+', '', msg)
                                msg = re.sub(r'@\w+', '', msg)
                                msg = re.sub(r'#\w+', '', msg)
                                msg = re.sub(r'[^\w\s]', ' ', msg)
                                msg = re.sub(r'\s+', ' ', msg)
                                for palavra in msg.split():
                                    if palavra not in stop_words and len(palavra) >= min_word_length:
                                        palavras.append(palavra)

                            all_text = ' '.join(palavras)

                            if all_text.strip():
                                # Generate word cloud (com stopwords também no WordCloud)
                                wordcloud = WordCloud(
                                    width=1200,
                                    height=600,
                                    max_words=max_words,
                                    background_color='white',
                                    colormap='viridis',
                                    stopwords=stop_words,
                                    min_word_length=min_word_length,
                                    collocations=False,
                                    random_state=42
                                ).generate(all_text)

                                # Create figure
                                fig, ax = plt.subplots(figsize=(15, 8))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                plt.tight_layout(pad=0)

                                # Save to buffer
                                buf = BytesIO()
                                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                buf.seek(0)
                                plt.close()

                                # Display
                                st.image(buf, use_container_width=True)
                            else:
                                st.warning("⚠️ No text available after filtering.")
                    else:
                        st.warning("⚠️ No texts match the selected filter.")
            else:
                st.warning("⚠️ **No text column configured.** Please select a text column in the Home page.")


# =============================================================================
# Time Series Analysis Section (UI)
# =============================================================================

# Display Time Series Analysis Section
if st.session_state.globalData.get('datasetLoaded') and st.session_state.globalData.get('dataset') is not None:
    df_global_ts = st.session_state.globalData['dataset']

    # Check if there's any classification data
    has_bert_ts = len(get_bert_columns(df_global_ts)) > 0
    has_detoxify_ts = len(get_detoxify_columns(df_global_ts)) > 0
    has_llm_ts = 'llm_classification' in df_global_ts.columns

    if has_bert_ts or has_detoxify_ts or has_llm_ts:
        st.markdown("")

        with st.container(border=True):
            st.markdown("### ⏱️ Time Series Analysis")
            st.markdown("Visualize how classifications vary over time")

            # Checkbox to use complete dataset
            use_complete_dataset = st.checkbox(
                "Use complete dataset (show only total comment count over time)",
                value=True,
                key="ts_use_complete_dataset",
                help="When checked, shows total comment count over time. When unchecked, shows class distribution over time."
            )

            # Select which dataset to use based on checkbox
            if use_complete_dataset:
                df = df_global_ts
                # Show dataset info
                st.info(f"💡 Using complete dataset with {len(df):,} texts.")
            else:
                # Filter to only texts classified by all 3 models
                bert_cols_ts = get_bert_columns(df_global_ts)
                detoxify_cols_ts = get_detoxify_columns(df_global_ts)
                llm_cols_ts = get_llm_columns(df_global_ts)

                if bert_cols_ts and detoxify_cols_ts and llm_cols_ts:
                    filtered_dataset_ts, _ = filter_fully_classified_texts(df_global_ts, bert_cols_ts, detoxify_cols_ts, llm_cols_ts)
                    df = filtered_dataset_ts
                else:
                    # Fallback to global if filtering not possible
                    df = df_global_ts
                # Show dataset info
                st.info(f"💡 Using filtered dataset with {len(df):,} texts (classified by all 3 models).")

            if use_complete_dataset:
                # Simple mode: just show total count over time
                col_date, col_granularity = st.columns([2, 1])

                with col_date:
                    date_columns = list(df.columns)
                    selected_date_col = st.selectbox(
                        "📅 Select Date/Time Column",
                        options=date_columns,
                        key="ts_date_col_simple",
                        help="Select the column containing date/time information"
                    )

                with col_granularity:
                    time_granularity_simple = st.selectbox(
                        "⏱️ Time Granularity",
                        options=['Minutes', 'Hours', 'Days'],
                        index=1,
                        key="ts_granularity_simple",
                        help="Select the time unit for grouping data"
                    )
                    # Map to pandas frequency strings
                    granularity_map = {'Minutes': 'T', 'Hours': 'H', 'Days': 'D'}
                    time_freq_simple = granularity_map[time_granularity_simple]

                if st.button("Generate Time Series", use_container_width=True, type="primary", key="generate_ts_simple"):
                    if selected_date_col:
                        # Parse datetime column
                        datetime_series = parse_datetime_column(df, selected_date_col)

                        if datetime_series is None or datetime_series.isna().all():
                            st.error(f"❌ Could not parse '{selected_date_col}' as a datetime column.")
                        else:
                            # Create working dataframe
                            df_work = df.copy()
                            df_work['parsed_datetime'] = datetime_series
                            df_work = df_work.dropna(subset=['parsed_datetime'])
                            df_work = df_work.sort_values('parsed_datetime')

                            if len(df_work) == 0:
                                st.error("❌ No valid datetime values found.")
                            else:
                                # Group by selected time granularity
                                df_work['date_hour'] = df_work['parsed_datetime'].dt.floor(time_freq_simple)

                                # Count total comments per time period
                                time_series_data = df_work.groupby('date_hour').size().reset_index(name='count')

                                # Create simple line chart
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=time_series_data['date_hour'],
                                    y=time_series_data['count'],
                                    mode='lines+markers',
                                    name='Total Comments',
                                    line=dict(color='#636EFA', width=2),
                                    marker=dict(size=4)
                                ))

                                fig.update_layout(
                                    title=dict(text="Total Comments Over Time", font=dict(size=16)),
                                    xaxis_title="Date/Time",
                                    yaxis_title="Number of Comments",
                                    height=500,
                                    hovermode='x unified',
                                    margin=dict(l=40, r=40, t=60, b=40)
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # Show summary
                                col_sum1, col_sum2, col_sum3 = st.columns(3)
                                with col_sum1:
                                    st.metric("Total Comments", f"{len(df_work):,}")
                                with col_sum2:
                                    st.metric("Time Range", f"{time_series_data['date_hour'].min().strftime('%Y-%m-%d')} to {time_series_data['date_hour'].max().strftime('%Y-%m-%d')}")
                                with col_sum3:
                                    avg_per_period = time_series_data['count'].mean()
                                    period_label = time_granularity_simple.rstrip('s')  # Remove 's' for singular
                                    st.metric(f"Avg per {period_label}", f"{avg_per_period:.1f}")
                    else:
                        st.warning("⚠️ Please select a date/time column.")

            else:
                # Advanced mode: show class distribution over time
                col1, col2, col3, col4 = st.columns([2, 1, 2, 2])

                with col1:
                    # Select datetime column
                    date_columns = list(df.columns)
                    selected_date_col = st.selectbox(
                        "📅 Select Date/Time Column",
                        options=date_columns,
                        key="ts_date_col_advanced",
                        help="Select the column containing date/time information"
                    )

                with col2:
                    # Select time granularity
                    time_granularity_advanced = st.selectbox(
                        "⏱️ Granularity",
                        options=['Minutes', 'Hours', 'Days'],
                        index=1,
                        key="ts_granularity_advanced",
                        help="Time unit for grouping"
                    )
                    granularity_map = {'Minutes': 'T', 'Hours': 'H', 'Days': 'D'}
                    time_freq_advanced = granularity_map[time_granularity_advanced]

                with col3:
                    # Select classifier type
                    classifier_options = []
                    if has_bert_ts:
                        classifier_options.append('BERT')
                    if has_detoxify_ts:
                        classifier_options.append('Detoxify')
                    if has_llm_ts:
                        classifier_options.append('LLM')

                    selected_classifier = st.selectbox(
                        "🤖 Select Classifier",
                        options=classifier_options,
                        help="Choose which classifier to visualize"
                    )

                with col4:
                    # Threshold selector (only for Detoxify, not BERT)
                    if selected_classifier == 'Detoxify':
                        ts_threshold = st.slider(
                            "🎚️ Probability Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.05,
                            help="Minimum probability to consider a text as belonging to a class"
                        )
                    else:
                        ts_threshold = None

                # Get available classes based on classifier
                classifier_lower = selected_classifier.lower()

                if classifier_lower == 'bert':
                    prob_cols_dict = get_bert_columns(df)
                    available_classes = list(prob_cols_dict.keys())
                elif classifier_lower == 'detoxify':
                    prob_cols_dict = get_detoxify_columns(df)
                    available_classes = list(prob_cols_dict.keys())
                else:  # LLM
                    if 'llm_classification' in df.columns:
                        available_classes = df['llm_classification'].dropna().unique().tolist()
                    else:
                        available_classes = []

                if available_classes:
                    # Select classes to display
                    selected_ts_classes = st.multiselect(
                        "📊 Select Classes to Display",
                        options=available_classes,
                        default=available_classes[:3] if len(available_classes) >= 3 else available_classes,
                        help="Choose which classes to show in the time series"
                    )

                    if st.button("Generate Time Series", use_container_width=True, type="primary", key="generate_ts_advanced"):
                        if selected_ts_classes and selected_date_col:
                            # Create and display chart
                            fig, error = create_time_series_chart(
                                df=df,
                                datetime_col=selected_date_col,
                                classifier_type=classifier_lower,
                                threshold=ts_threshold if ts_threshold is not None else 0.5,
                                selected_classes=selected_ts_classes,
                                time_granularity=time_freq_advanced
                            )

                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(f"❌ {error}")
                        else:
                            st.warning("⚠️ Please select at least one class and a date/time column.")
                else:
                    st.warning(f"⚠️ No classes found for {selected_classifier} classifier.")

