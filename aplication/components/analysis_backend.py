"""
Analysis Backend
Contains all business logic for classified data analysis and visualizations
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =============================================================================
# Shared Color Palettes and Mappings
# =============================================================================

# Shared palette and sentiment mapping for donuts across the page
# We keep a persistent mapping so the same label gets the same color across charts
default_palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Dark24
_label_color_map = {}
_color_idx = {'i': 0}
sentiment_color_map = {
    'positive': '#2ecc71',  # green
    'negative': '#e74c3c',  # red
    'neutral': '#f1c40f'    # yellow
}


# =============================================================================
# Dataset Filtering
# =============================================================================

def filter_fully_classified_texts(df, bert_cols, detoxify_cols, llm_cols):
    """
    Filter dataset to include only texts that were classified by all three models.
    Returns filtered dataframe and statistics dict.
    """
    stats = {
        'total': len(df),
        'bert_classified': 0,
        'detoxify_classified': 0,
        'llm_classified': 0,
        'all_classified': 0
    }

    # Create masks for each classifier
    bert_mask = pd.Series([True] * len(df), index=df.index)
    if bert_cols:
        # Check if all BERT probability columns have numeric values (not 'not_classified')
        for col in bert_cols.values():
            if col in df.columns:
                numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
                bert_mask = bert_mask & numeric_mask

    detoxify_mask = pd.Series([True] * len(df), index=df.index)
    if detoxify_cols:
        # Check if all Detoxify probability columns have numeric values
        for col in detoxify_cols.values():
            if col in df.columns:
                numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
                detoxify_mask = detoxify_mask & numeric_mask

    llm_mask = pd.Series([True] * len(df), index=df.index)
    if 'classification' in llm_cols and llm_cols['classification'] in df.columns:
        # Check if LLM classification is not null and not 'not classified'
        llm_mask = (
            df[llm_cols['classification']].notna() &
            (df[llm_cols['classification']].astype(str).str.lower() != 'not classified')
        )

    # Calculate statistics
    stats['bert_classified'] = bert_mask.sum() if bert_cols else 0
    stats['detoxify_classified'] = detoxify_mask.sum() if detoxify_cols else 0
    stats['llm_classified'] = llm_mask.sum() if llm_cols else 0

    # Combined mask: all three classifiers must have classified the text
    combined_mask = bert_mask & detoxify_mask & llm_mask
    stats['all_classified'] = combined_mask.sum()

    # Filter dataset
    filtered_df = df[combined_mask].copy()

    return filtered_df, stats


# =============================================================================
# Column Parsing
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
# Threshold Curves and Charts
# =============================================================================

def create_threshold_chart(df, prob_columns, title, height=350):
    """
    Create threshold-based class distribution chart.
    Shows how many texts would be classified as each class at different probability thresholds.
    All texts in the dataframe are already classified (no 'not_classified' values).
    """
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']

    # Generate thresholds from 0.0 to 1.0
    thresholds = np.linspace(0.0, 1.0, 101)

    for idx, (label, col) in enumerate(prob_columns.items()):
        counts_at_threshold = []

        for threshold in thresholds:
            # Convert column to numeric to handle string values
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            count = (numeric_col >= threshold).sum()
            counts_at_threshold.append(count)

        # Choose color: prefer sentiment overrides for common labels
        key = str(label).strip()
        low = key.lower()
        if low in sentiment_color_map:
            color = sentiment_color_map[low]
            _label_color_map[key] = sentiment_color_map[low]
        else:
            # Reuse persistent mapping when available, otherwise pick next palette color
            if key in _label_color_map:
                color = _label_color_map[key]
            else:
                color = colors[idx % len(colors)]
                _label_color_map[key] = color
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

    # Normalize confidence strings to avoid duplicates like "High" vs "high"
    conf_series = (
        df[llm_cols['confidence']]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
    )
    confidence_counts = conf_series.value_counts()

    # Default colors for confidence levels (fallback)
    color_map = {
        'high': '#2ecc71',
        'medium': '#f39c12',
        'low': '#e74c3c'
    }

    # Blue palette to match probability styling used in the top comments table
    blue_map = {
        'low': '#e3f2fd',    # light blue
        'medium': '#64b5f6', # medium blue
        'high': '#0d47a1'    # dark blue
    }

    # If the confidence levels are the common set (low/medium/high), prefer the blue palette
    idxs = [str(c).lower() for c in confidence_counts.index]
    if set(idxs).issubset({'low', 'medium', 'high'}):
        colors = [blue_map.get(str(c).lower(), '#95a5a6') for c in confidence_counts.index]
        # persist mapping for consistency across charts
        for lbl, col in zip(confidence_counts.index, colors):
            _label_color_map[str(lbl).strip()] = col
    else:
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
        showlegend=True,
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
    All texts in the dataframe are already classified (no 'not_classified' values).
    """
    class_counts = {}

    for label, col in prob_columns.items():
        # Convert column to numeric to handle string values
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        count = (numeric_col >= threshold).sum()
        if count > 0:
            class_counts[label] = count

    labels = list(class_counts.keys())
    values = list(class_counts.values())

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
        showlegend=True,
        annotations=[dict(
            text=f'{len(df)}<br>texts',
            x=0.5, y=0.5,
            font_size=11,
            showarrow=False
        )]
    )

    return fig


def create_bert_argmax_distribution_donut(df, bert_cols, threshold, title, height=350):
    """
    Create donut chart for BERT where each text is assigned to a single
    final class based on the highest probability (argmax). Only texts whose
    maximum BERT probability is >= threshold are counted.
    All texts in the dataframe are already classified (no 'not_classified' values).
    """
    # Keep only probability columns that exist in the dataframe
    available_bert_cols = {label: col for label, col in bert_cols.items() if col in df.columns}
    if not available_bert_cols:
        # Fallback: behave like empty distribution
        labels = ['Below threshold']
        values = [len(df)]
    else:
        prob_cols = list(available_bert_cols.values())
        sub = df[prob_cols].copy()

        if sub.empty:
            labels = ['Below threshold']
            values = [len(df)]
        else:
            # Replace NaNs with -inf so they don't win the argmax
            sub_filled = sub.fillna(float('-inf'))

            # Compute max probability and argmax column per row
            max_probs = sub_filled.max(axis=1)
            argmax_cols = sub_filled.idxmax(axis=1)

            # Map column name to label
            col_to_label = {col: label for label, col in available_bert_cols.items()}
            bert_final = argmax_cols.map(col_to_label)

            # Apply threshold on the max probability
            mask = max_probs >= threshold
            bert_final_filtered = bert_final[mask]

            class_counts_series = bert_final_filtered.value_counts(dropna=True)

            if class_counts_series.empty:
                labels = ['Below threshold']
                values = [len(df)]
            else:
                labels = list(class_counts_series.index.astype(str))
                values = list(class_counts_series.values)

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
        showlegend=True,
        annotations=[dict(
            text=f'{len(df)}<br>texts',
            x=0.5, y=0.5,
            font_size=11,
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

    # Build colors using shared palette and sentiment overrides
    labels = list(class_counts.index.astype(str))
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
        showlegend=True,
        annotations=[dict(
            text=f'{len(filtered_df)}<br>texts',
            x=0.5, y=0.5,
            font_size=12,
            showarrow=False
        )]
    )

    return fig


# =============================================================================
# Column Detection
# =============================================================================

def detect_likes_column(df):
    """Detect the column containing like counts."""
    possible_names = ['like_count', 'likes', 'likeCount', 'like_counts', 'num_likes']
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def detect_author_column(df):
    """Detect the column containing author/user names."""
    possible_names = ['author', 'user', 'username', 'authorDisplayName', 'user_name', 'channel']
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def detect_reply_count_column(df):
    """Detect the column containing reply counts."""
    possible_names = ['reply_count', 'replies', 'replyCount', 'reply_counts', 'num_replies', 'totalReplyCount']
    for name in possible_names:
        if name in df.columns:
            return name
    return None


# =============================================================================
# Table Styling
# =============================================================================

def get_probability_value_color(value):
    """
    Get background color for probability value (light blue to dark blue gradient).
    Low probability = light blue, High probability = dark blue.
    """
    if pd.isna(value):
        return 'background-color: #f0f0f0'

    # Clamp value between 0 and 1
    value = max(0, min(1, float(value)))

    # Light blue (low) to Dark blue (high)
    # Light blue: #e3f2fd (227, 242, 253)
    # Dark blue: #0d47a1 (13, 71, 161)

    r = int(227 - (227 - 13) * value)   # 227 to 13
    g = int(242 - (242 - 71) * value)   # 242 to 71
    b = int(253 - (253 - 161) * value)  # 253 to 161

    # Use white text for darker blues (value > 0.5)
    text_color = 'white' if value > 0.5 else 'black'

    return f'background-color: rgba({r}, {g}, {b}, 0.85); color: {text_color}'


def style_probability_columns(df, prob_columns):
    """
    Apply color styling to probability columns in a dataframe.
    """
    def apply_color(val):
        return get_probability_value_color(val)

    # Create a styler
    styler = df.style

    for col in prob_columns:
        if col in df.columns:
            styler = styler.applymap(apply_color, subset=[col])

    return styler


# =============================================================================
# Top Comments Tables
# =============================================================================

def create_top_comments_table(df, text_column, likes_col, author_col, reply_col, bert_cols, detoxify_cols, llm_cols, sort_by='likes', top_n=20):
    """
    Create a styled dataframe with top comments.
    
    Args:
        sort_by: 'likes' or 'replies' to determine sorting column
    """
    # Determine sort column
    if sort_by == 'replies' and reply_col:
        sort_col = reply_col
    elif likes_col:
        sort_col = likes_col
    else:
        return None, "No sorting column found in dataset"

    # Sort by selected column
    df_sorted = df.sort_values(by=sort_col, ascending=False).head(top_n).copy()

    # Build display columns
    display_cols = []
    rename_map = {}
    separator_cols = []

    # Likes column
    if likes_col and likes_col in df.columns:
        display_cols.append(likes_col)
        rename_map[likes_col] = '👍 Likes'

    # Reply count column
    if reply_col and reply_col in df.columns:
        display_cols.append(reply_col)
        rename_map[reply_col] = '💬 Replies'

    # Author column (if exists)
    if author_col and author_col in df.columns:
        display_cols.append(author_col)
        rename_map[author_col] = '👤 Author'

    # Text column
    if text_column and text_column in df.columns:
        display_cols.append(text_column)
        rename_map[text_column] = '📝 Comment'

    # Probability columns
    prob_columns = []

    # BERT columns
    has_bert = False
    for label, col in bert_cols.items():
        if col in df.columns:
            display_cols.append(col)
            rename_map[col] = f'🤖 {label}'
            prob_columns.append(f'🤖 {label}')
            has_bert = True

    # Add separator after BERT if exists and before Detoxify
    if has_bert and len(detoxify_cols) > 0:
        separator_col_name = '│ (1)'
        df_sorted[separator_col_name] = ''
        display_cols.append(separator_col_name)
        separator_cols.append(separator_col_name)

    # Detoxify columns
    has_detoxify = False
    for label, col in detoxify_cols.items():
        if col in df.columns:
            display_cols.append(col)
            rename_map[col] = f'🛡️ {label}'
            prob_columns.append(f'🛡️ {label}')
            has_detoxify = True

    # Add separator after Detoxify if exists and before LLM
    if has_detoxify and 'classification' in llm_cols and llm_cols['classification'] in df.columns:
        separator_col_name = '│ (2)'
        df_sorted[separator_col_name] = ''
        display_cols.append(separator_col_name)
        separator_cols.append(separator_col_name)

    # LLM columns
    if 'classification' in llm_cols and llm_cols['classification'] in df.columns:
        display_cols.append(llm_cols['classification'])
        rename_map[llm_cols['classification']] = '🧠 LLM Class'

    if 'confidence' in llm_cols and llm_cols['confidence'] in df.columns:
        display_cols.append(llm_cols['confidence'])
        rename_map[llm_cols['confidence']] = '🧠 Confidence'

    # Filter to only existing columns
    display_cols = [col for col in display_cols if col in df_sorted.columns]

    # Create display dataframe
    df_display = df_sorted[display_cols].copy()

    # Round probability columns to 2 decimal places
    for old_col in bert_cols.values():
        if old_col in df_display.columns:
            df_display[old_col] = pd.to_numeric(df_display[old_col], errors='coerce').round(2)

    for old_col in detoxify_cols.values():
        if old_col in df_display.columns:
            df_display[old_col] = pd.to_numeric(df_display[old_col], errors='coerce').round(2)

    df_display = df_display.rename(columns=rename_map)

    # Reset index
    df_display = df_display.reset_index(drop=True)

    return df_display, prob_columns, separator_cols


def create_bert_llm_agreement_table(df, text_column, likes_col, author_col, reply_col, bert_cols, detoxify_cols, llm_cols, sort_by='likes', top_n=20):
    """
    Create a table containing comments where BERT's final class (argmax over bert_prob_ columns)
    matches the LLM classification.

    Returns a dataframe with columns: likes, replies, author, text, bert_final, llm_classification
    """
    # Ensure required columns exist
    if not bert_cols or 'classification' not in llm_cols or llm_cols['classification'] not in df.columns:
        return None, "No BERT or LLM classification columns available."

    # Only consider bert probability columns that exist in df
    available_bert_cols = {label: col for label, col in bert_cols.items() if col in df.columns}
    if not available_bert_cols:
        return None, "No BERT probability columns found in the dataset."

    # Reverse mapping col -> label
    col_to_label = {col: label for label, col in available_bert_cols.items()}

    df_work = df.copy()

    # Compute BERT final class (label with highest probability)
    def _bert_final(row):
        best_col = None
        best_val = -float('inf')
        for col in col_to_label:
            try:
                v = row[col]
                if pd.isna(v):
                    continue
                if float(v) > best_val:
                    best_val = float(v)
                    best_col = col
            except Exception:
                continue
        return col_to_label.get(best_col) if best_col else None

    df_work['bert_final'] = df_work.apply(_bert_final, axis=1)

    # Build display columns
    display_cols = []
    rename_map = {}

    if likes_col and likes_col in df_work.columns:
        display_cols.append(likes_col)
        rename_map[likes_col] = '👍 Likes'

    if reply_col and reply_col in df_work.columns:
        display_cols.append(reply_col)
        rename_map[reply_col] = '💬 Replies'

    if author_col and author_col in df_work.columns:
        display_cols.append(author_col)
        rename_map[author_col] = '👤 Author'

    if text_column and text_column in df_work.columns:
        display_cols.append(text_column)
        rename_map[text_column] = '📝 Comment'

    # Add computed BERT final and LLM classification columns side-by-side
    display_cols.append('bert_final')
    display_cols.append(llm_cols['classification'])
    rename_map['bert_final'] = '🤖 BERT Final'
    rename_map[llm_cols['classification']] = '🧠 LLM Class'

    # Add BERT probability columns after BERT Final
    if bert_cols:
        for label, col in bert_cols.items():
            if col in df_work.columns:
                display_cols.append(col)
                rename_map[col] = f'🤖 {label}'

    # Add Detoxify probability columns (if provided) after BERT probabilities
    if detoxify_cols:
        for label, col in detoxify_cols.items():
            if col in df_work.columns:
                display_cols.append(col)
                rename_map[col] = f'🛡️ {label}'

    # Filter to existing
    display_cols = [c for c in display_cols if c in df_work.columns]

    df_table = df_work[display_cols].copy()
    df_table = df_table.rename(columns=rename_map)

    # Filter rows where bert_final equals llm classification (case-insensitive)
    bert_col_name = '🤖 BERT Final'
    llm_col_name = '🧠 LLM Class'

    if bert_col_name not in df_table.columns or llm_col_name not in df_table.columns:
        return None, 'Required columns not present after processing.'

    # Prepare lowercase LLM classification for comparison
    llm_lower = df_table[llm_col_name].astype(str).str.lower()

    # Only consider rows where:
    # - Both BERT and LLM classifications are not null
    # - LLM actually classified the text (value different from "not classified")
    # - BERT and LLM agree on the class label (case-insensitive)
    mask = (
        df_table[bert_col_name].notna()
    ) & (
        df_table[llm_col_name].notna()
    ) & (
        llm_lower != 'not classified'
    ) & (
        df_table[bert_col_name].astype(str).str.lower() == llm_lower
    )

    df_agree = df_table[mask].copy()

    if df_agree.empty:
        return pd.DataFrame(columns=rename_map.values()), None

    # Round probability columns to 2 decimal places
    for renamed_col in df_agree.columns:
        if renamed_col.startswith('🤖 ') and renamed_col != '🤖 BERT Final':
            df_agree[renamed_col] = pd.to_numeric(df_agree[renamed_col], errors='coerce').round(2)
        if renamed_col.startswith('🛡️ '):
            df_agree[renamed_col] = pd.to_numeric(df_agree[renamed_col], errors='coerce').round(2)

    # Sorting
    if sort_by == 'likes' and '👍 Likes' in df_agree.columns:
        df_agree = df_agree.sort_values(by='👍 Likes', ascending=False)
    elif sort_by == 'replies' and '💬 Replies' in df_agree.columns:
        df_agree = df_agree.sort_values(by='💬 Replies', ascending=False)

    df_agree = df_agree.head(top_n).reset_index(drop=True)

    return df_agree, None


# =============================================================================
# Time Series Analysis
# =============================================================================

def parse_datetime_column(df, column_name):
    """
    Try to parse a column as datetime using multiple formats.
    Returns a Series with parsed datetime values or None if parsing fails.
    """
    if column_name not in df.columns:
        return None

    column_data = df[column_name].copy()

    # List of datetime formats to try
    datetime_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d',
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d',
        '%d-%m-%Y %H:%M:%S',
        '%d-%m-%Y',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%fZ',
    ]

    # Try pandas to_datetime with infer_datetime_format first
    try:
        parsed = pd.to_datetime(column_data, infer_datetime_format=True, errors='coerce')
        if parsed.notna().sum() > 0:
            return parsed
    except:
        pass

    # Try each format explicitly
    for fmt in datetime_formats:
        try:
            parsed = pd.to_datetime(column_data, format=fmt, errors='coerce')
            if parsed.notna().sum() > 0:
                return parsed
        except:
            continue

    # Try parsing as unix timestamp (seconds)
    try:
        if pd.api.types.is_numeric_dtype(column_data):
            parsed = pd.to_datetime(column_data, unit='s', errors='coerce')
            if parsed.notna().sum() > 0:
                return parsed
    except:
        pass

    # Try parsing as unix timestamp (milliseconds)
    try:
        if pd.api.types.is_numeric_dtype(column_data):
            parsed = pd.to_datetime(column_data, unit='ms', errors='coerce')
            if parsed.notna().sum() > 0:
                return parsed
    except:
        pass

    return None


def create_time_series_chart(df, datetime_col, classifier_type, threshold, selected_classes, time_granularity='H'):
    """
    Create a time series chart showing class distribution over time.
    
    Parameters:
    - df: DataFrame with data
    - datetime_col: Column name with datetime data
    - classifier_type: 'bert', 'detoxify', or 'llm'
    - threshold: Probability threshold (for bert and detoxify)
    - selected_classes: List of classes to display
    - time_granularity: Time granularity for grouping ('T' for minutes, 'H' for hours, 'D' for days)
    """
    # Parse datetime column
    datetime_series = parse_datetime_column(df, datetime_col)

    if datetime_series is None or datetime_series.isna().all():
        return None, "Could not parse datetime column"

    # Create working dataframe
    df_work = df.copy()
    df_work['parsed_datetime'] = datetime_series
    df_work = df_work.dropna(subset=['parsed_datetime'])
    df_work = df_work.sort_values('parsed_datetime')

    if len(df_work) == 0:
        return None, "No valid datetime values found"

    # Get probability columns based on classifier type
    if classifier_type == 'bert':
        prob_cols_dict = get_bert_columns(df_work)
        if not prob_cols_dict:
            return None, "No BERT classification columns found"
    elif classifier_type == 'detoxify':
        prob_cols_dict = get_detoxify_columns(df_work)
        if not prob_cols_dict:
            return None, "No Detoxify classification columns found"
    elif classifier_type == 'llm':
        if 'llm_classification' not in df_work.columns:
            return None, "No LLM classification column found"
    else:
        return None, "Invalid classifier type"

    # Prepare data for time series
    if classifier_type == 'bert':
        # For BERT: use argmax (each text counted once in the class with highest probability)
        df_work['date_hour'] = df_work['parsed_datetime'].dt.floor(time_granularity)

        # Get probability columns
        prob_cols = [col for col in prob_cols_dict.values() if col in df_work.columns]
        col_to_label = {col: label for label, col in prob_cols_dict.items() if col in df_work.columns}

        if not prob_cols:
            return None, "No valid BERT probability columns found"

        # Convert to numeric
        for col in prob_cols:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

        # Compute argmax class for each row
        bert_argmax_cols = df_work[prob_cols].idxmax(axis=1)
        df_work['bert_final_class'] = bert_argmax_cols.map(lambda c: col_to_label.get(c) if c in col_to_label else None)

        # Count texts per class per time period
        time_series_data = []
        for date_hour in sorted(df_work['date_hour'].unique()):
            df_hour = df_work[df_work['date_hour'] == date_hour]

            for class_name in selected_classes:
                count = (df_hour['bert_final_class'] == class_name).sum()
                time_series_data.append({
                    'datetime': date_hour,
                    'class': class_name,
                    'count': count
                })

        df_ts = pd.DataFrame(time_series_data)

        if len(df_ts) == 0:
            return None, "No data matches the selected criteria"

        # Create line chart
        fig = go.Figure()

        # Total comments bar (always shown as background reference)
        df_total_bert = df_work.groupby('date_hour').size().reset_index(name='total_count')
        fig.add_trace(go.Bar(
            x=df_total_bert['date_hour'],
            y=df_total_bert['total_count'],
            name='Total Comments',
            marker=dict(color='rgba(30, 136, 229, 0.35)'),
            hovertemplate='<b>Total Comments</b><br>Count: %{y}<extra></extra>'
        ))

        use_sentiment_colors = all(c.strip().lower() in sentiment_color_map for c in selected_classes)

        for class_name in selected_classes:
            df_class = df_ts[df_ts['class'] == class_name]
            if len(df_class) > 0:
                color = sentiment_color_map.get(class_name.strip().lower()) if use_sentiment_colors else None
                fig.add_trace(go.Scatter(
                    x=df_class['datetime'],
                    y=df_class['count'],
                    mode='lines+markers',
                    name=class_name,
                    line=dict(color=color, width=2) if color else dict(width=2),
                    marker=dict(color=color, size=6) if color else dict(size=6)
                ))

        fig.update_layout(
            title=f"BERT Classification Over Time (Argmax)",
            xaxis_title="Date/Time",
            yaxis_title="Number of Texts",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

    elif classifier_type == 'detoxify':
        # For Detoxify: count texts above threshold for each class
        df_work['date_hour'] = df_work['parsed_datetime'].dt.floor(time_granularity)

        time_series_data = []
        for date_hour in sorted(df_work['date_hour'].unique()):
            df_hour = df_work[df_work['date_hour'] == date_hour]

            # Iterate over dictionary items (class_name: column_name)
            for class_name, prob_col in prob_cols_dict.items():
                if class_name in selected_classes:
                    # Count texts above threshold for this class
                    # Convert column to numeric to handle string values
                    numeric_col = pd.to_numeric(df_hour[prob_col], errors='coerce')
                    count = (numeric_col >= threshold).sum()
                    time_series_data.append({
                        'datetime': date_hour,
                        'class': class_name,
                        'count': count
                    })

        df_ts = pd.DataFrame(time_series_data)

        if len(df_ts) == 0:
            return None, "No data matches the selected criteria"

        # Create line chart
        fig = go.Figure()

        # Total comments bar (always shown as background reference)
        df_total_detox = df_work.groupby('date_hour').size().reset_index(name='total_count')
        fig.add_trace(go.Bar(
            x=df_total_detox['date_hour'],
            y=df_total_detox['total_count'],
            name='Total Comments',
            marker=dict(color='rgba(30, 136, 229, 0.35)'),
            hovertemplate='<b>Total Comments</b><br>Count: %{y}<extra></extra>'
        ))

        use_sentiment_colors = all(c.strip().lower() in sentiment_color_map for c in selected_classes)

        for class_name in selected_classes:
            df_class = df_ts[df_ts['class'] == class_name]
            if len(df_class) > 0:
                color = sentiment_color_map.get(class_name.strip().lower()) if use_sentiment_colors else None
                fig.add_trace(go.Scatter(
                    x=df_class['datetime'],
                    y=df_class['count'],
                    mode='lines+markers',
                    name=class_name,
                    line=dict(color=color, width=2) if color else dict(width=2),
                    marker=dict(color=color, size=6) if color else dict(size=6)
                ))

        fig.update_layout(
            title=f"DETOXIFY Classification Over Time (Threshold ≥ {threshold:.2f})",
            xaxis_title="Date/Time",
            yaxis_title="Number of Texts",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

    else:  # LLM
        # For LLM: count texts for each classification
        df_work['date_hour'] = df_work['parsed_datetime'].dt.floor(time_granularity)

        time_series_data = []
        for date_hour in sorted(df_work['date_hour'].unique()):
            df_hour = df_work[df_work['date_hour'] == date_hour]

            for class_name in selected_classes:
                count = (df_hour['llm_classification'] == class_name).sum()
                time_series_data.append({
                    'datetime': date_hour,
                    'class': class_name,
                    'count': count
                })

        df_ts = pd.DataFrame(time_series_data)

        if len(df_ts) == 0:
            return None, "No data matches the selected criteria"

        # Create line chart
        fig = go.Figure()

        # Total comments bar (always shown as background reference)
        df_total_llm = df_work.groupby('date_hour').size().reset_index(name='total_count')
        fig.add_trace(go.Bar(
            x=df_total_llm['date_hour'],
            y=df_total_llm['total_count'],
            name='Total Comments',
            marker=dict(color='rgba(30, 136, 229, 0.35)'),
            hovertemplate='<b>Total Comments</b><br>Count: %{y}<extra></extra>'
        ))

        use_sentiment_colors = all(c.strip().lower() in sentiment_color_map for c in selected_classes)

        for class_name in selected_classes:
            df_class = df_ts[df_ts['class'] == class_name]
            if len(df_class) > 0:
                color = sentiment_color_map.get(class_name.strip().lower()) if use_sentiment_colors else None
                fig.add_trace(go.Scatter(
                    x=df_class['datetime'],
                    y=df_class['count'],
                    mode='lines+markers',
                    name=class_name,
                    line=dict(color=color, width=2) if color else dict(width=2),
                    marker=dict(color=color, size=6) if color else dict(size=6)
                ))

        fig.update_layout(
            title=f"LLM Classification Over Time",
            xaxis_title="Date/Time",
            yaxis_title="Number of Texts",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

    return fig, None
