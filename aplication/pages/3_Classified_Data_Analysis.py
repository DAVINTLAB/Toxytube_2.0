"""
Página de Análise de Dados Classificados
Visualiza e analisa datasets que já foram classificados
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

from components.navigation import render_navigation

# =============================================================================
# Initialize Session State - Dados do Processo de Análise
# =============================================================================

# Dicionário com todos os dados necessários para o processo de análise de dados classificados
if 'analysisData' not in st.session_state:
    st.session_state.analysisData = {
        'dataset': None,                    # Dataset classificado carregado
        'labelColumn': None,                # Coluna com as classificações/labels
        'datetimeColumn': None,             # Coluna de data/hora (opcional)
        'datasetLoaded': False,             # Status do carregamento
        'aggregation': 'day',               # Nível de agregação temporal (hour, day, week, month)
        'selectedFilters': [],              # Filtros aplicados
        'chartType': 'pie'                  # Tipo de gráfico selecionado
    }

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Análise de Dados Classificados - ML Hub",
    page_icon="📊",
    layout="wide",
)

# Render navigation sidebar
render_navigation('analysis')

# =============================================================================
# Main Content
# =============================================================================

# Page header
st.markdown("# 📊 Análise de Dados Classificados")
st.markdown("Visualize e analise datasets que já foram classificados por modelos de Machine Learning.")

st.markdown("---")

# =============================================================================
# Helper Functions
# =============================================================================

def load_dataset(uploaded_file):
    """Load dataset from uploaded file"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
        else:
            st.error(f"❌ Formato de arquivo não suportado: {file_extension}")
            return None

        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
        return None

def parse_datetime_column(df, column_name):
    """Parse datetime column and handle various formats"""
    try:
        # Try common datetime formats
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')

        # Check for any failed conversions
        null_count = df[column_name].isnull().sum()
        if null_count > 0:
            st.warning(f"⚠️ {null_count} valores de data/hora não puderam ser convertidos e foram ignorados.")

        # Remove rows with invalid datetime
        df = df.dropna(subset=[column_name])

        return df
    except Exception as e:
        st.error(f"❌ Erro ao processar coluna de data/hora: {str(e)}")
        return None

def create_class_distribution_chart(df, label_column):
    """Create pie chart showing class distribution"""
    try:
        # Count occurrences of each class
        class_counts = df[label_column].value_counts()

        # Create pie chart
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title=f"Distribuição de Classes - {label_column}",
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Quantidade: %{value}<br>Percentual: %{percent}<extra></extra>'
        )

        fig.update_layout(
            showlegend=True,
            height=500,
            font=dict(size=12)
        )

        return fig, class_counts
    except Exception as e:
        st.error(f"❌ Erro ao criar gráfico de distribuição: {str(e)}")
        return None, None

def create_temporal_analysis_chart(df, datetime_column, label_column, aggregation='day'):
    """Create temporal analysis chart showing class variation over time"""
    try:
        # Create a copy to avoid modifying original data
        df_temp = df.copy()

        # Set datetime as index
        df_temp = df_temp.set_index(datetime_column)

        # Determine aggregation frequency
        freq_map = {
            'hour': 'H',
            'day': 'D',
            'week': 'W',
            'month': 'M'
        }

        freq = freq_map.get(aggregation, 'D')

        # Group by time period and class, then count
        grouped = df_temp.groupby([pd.Grouper(freq=freq), label_column]).size().unstack(fill_value=0)

        # Create line chart
        fig = go.Figure()

        # Add a line for each class
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
                    hovertemplate=f'<b>{column}</b><br>Data: %{{x}}<br>Quantidade: %{{y}}<extra></extra>'
                )
            )

        fig.update_layout(
            title=f"Variação Temporal das Classes - Agregação por {aggregation.title()}",
            xaxis_title="Data/Hora",
            yaxis_title="Número de Registros",
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
        st.error(f"❌ Erro ao criar análise temporal: {str(e)}")
        return None, None

# =============================================================================
# Step 1: Dataset Loading
# =============================================================================

st.markdown("### Etapa 1: Carregamento do Dataset")
st.markdown("Faça upload de um dataset que já foi classificado por um modelo de Machine Learning.")

uploaded_file = st.file_uploader(
    "Selecione o arquivo com dados classificados:",
    type=['csv', 'json', 'xlsx', 'xls', 'parquet'],
    help="Formatos suportados: CSV, Excel (XLSX/XLS), JSON, Parquet"
)

if uploaded_file is not None:
    # Load dataset
    with st.spinner("🔄 Carregando dataset..."):
        dataset = load_dataset(uploaded_file)

    if dataset is not None:
        st.session_state.analysisData['dataset'] = dataset
        st.session_state.analysisData['dataset']_loaded = True

        # Show basic information
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📊 Total de Linhas", f"{len(dataset):,}")

        with col2:
            st.metric("📋 Total de Colunas", len(dataset.columns))

        with col3:
            file_size = uploaded_file.size / 1024  # Convert to KB
            if file_size > 1024:
                file_size = file_size / 1024  # Convert to MB
                st.metric("📏 Tamanho do Arquivo", f"{file_size:.2f} MB")
            else:
                st.metric("📏 Tamanho do Arquivo", f"{file_size:.2f} KB")

# Show dataset preview and column selection if loaded
if st.session_state.analysisData['dataset']_loaded and st.session_state.analysisData['dataset'] is not None:
    dataset = st.session_state.analysisData['dataset']

    # Dataset preview
    st.dataframe(dataset.head(10), use_container_width=True)



st.markdown("---")

# =============================================================================
# Step 2: Column Selection
# =============================================================================

if st.session_state.analysisData['dataset']_loaded and st.session_state.analysisData['dataset'] is not None:
    dataset = st.session_state.analysisData['dataset']

    st.markdown("### Etapa 2: Seleção de Colunas")
    st.markdown("Selecione as colunas que contêm os labels classificados e as informações de data/hora.")

    col1, col2 = st.columns(2)

    with col1:
        # Label column selection
        st.markdown("**Coluna de Labels/Classes:**")
        label_columns = dataset.columns.tolist()

        label_column = st.selectbox(
            "Selecione a coluna que contém os labels finais da classificação:",
            options=label_columns,
            help="Esta coluna deve conter os resultados da classificação (ex: POSITIVE, NEGATIVE, NEUTRAL)"
        )

        if label_column:
            st.session_state.analysisData['labelColumn'] = label_column

            # Show unique values in selected column
            unique_values = dataset[label_column].unique()
            st.info(f"📋 Classes encontradas: {', '.join(map(str, unique_values))}")

    with col2:
        # Datetime column selection
        st.markdown("**Coluna de Data/Hora:**")

        # Try to identify datetime columns
        potential_datetime_cols = []
        for col in dataset.columns:
            if dataset[col].dtype == 'object':
                # Check if column contains datetime-like strings
                sample_value = str(dataset[col].iloc[0])
                if any(char.isdigit() for char in sample_value) and any(sep in sample_value for sep in ['-', '/', ':', ' ']):
                    potential_datetime_cols.append(col)

        datetime_column = st.selectbox(
            "Selecione a coluna que contém data/hora:",
            options=dataset.columns.tolist(),
            help="Esta coluna deve conter informações de data e hora (formato: YYYY-MM-DD HH:MM:SS)"
        )

        if datetime_column:
            st.session_state.analysisData['datetimeColumn'] = datetime_column

            # Show sample values
            sample_values = dataset[datetime_column].head(3).tolist()
            st.info(f"📅 Exemplos de valores: {', '.join(map(str, sample_values))}")

st.markdown("---")

# =============================================================================
# Step 3: Data Visualization
# =============================================================================

if (st.session_state.analysisData['dataset']_loaded and
    st.session_state.analysisData['labelColumn'] and
    st.session_state.analysisData['datetimeColumn']):

    dataset = st.session_state.analysisData['dataset']
    label_column = st.session_state.analysisData['labelColumn']
    datetime_column = st.session_state.analysisData['datetimeColumn']

    st.markdown("### Etapa 3: Visualizações e Análises")

    # Parse datetime column
    with st.spinner("🔄 Processando dados temporais..."):
        dataset_processed = parse_datetime_column(dataset.copy(), datetime_column)

    if dataset_processed is not None:
        st.success("✅ Dados processados com sucesso!")

        # =============================================================================
        # Class Distribution Chart
        # =============================================================================

        st.markdown("#### Distribuição de Classes")
        st.markdown("Visualize a proporção de cada classe no dataset completo.")

        fig_pie, class_counts = create_class_distribution_chart(dataset_processed, label_column)

        if fig_pie is not None:
            col_chart, col_stats = st.columns([3, 1])

            with col_chart:
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_stats:
                st.markdown("**📊 Estatísticas:**")
                total_samples = len(dataset_processed)

                for label, count in class_counts.items():
                    percentage = (count / total_samples) * 100
                    st.metric(
                        label=f"🏷️ {label}",
                        value=f"{count:,}",
                        delta=f"{percentage:.1f}%"
                    )

        st.markdown("---")

        # =============================================================================
        # Temporal Analysis Chart
        # =============================================================================

        st.markdown("#### Análise Temporal")
        st.markdown("Visualize como as classes variam ao longo do tempo.")

        # Aggregation selector
        col_agg, col_info = st.columns([2, 8])

        with col_agg:
            aggregation = st.selectbox(
                "Agregação temporal:",
                options=['hour', 'day', 'week', 'month'],
                index=1,  # Default to 'day'
                help="Como agrupar os dados no tempo"
            )

        with col_info:
            # Show date range
            min_date = dataset_processed[datetime_column].min()
            max_date = dataset_processed[datetime_column].max()
            date_range = max_date - min_date

            st.info(f"📅 Período dos dados: {min_date.strftime('%d/%m/%Y %H:%M')} até {max_date.strftime('%d/%m/%Y %H:%M')} ({date_range.days} dias)")

        # Create temporal chart
        fig_temporal, grouped_data = create_temporal_analysis_chart(
            dataset_processed, datetime_column, label_column, aggregation
        )

        if fig_temporal is not None:
            st.plotly_chart(fig_temporal, use_container_width=True)

        st.markdown("---")

# =============================================================================
# Footer Information
# =============================================================================

st.markdown("### 📋 Sobre Esta Ferramenta")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Análise de Dados Classificados** permite:

    - 📊 **Distribuição de Classes**: Visualizar proporções de cada categoria
    - ⏰ **Análise Temporal**: Ver como classes variam no tempo
    - 📈 **Tendências**: Identificar padrões crescentes ou decrescentes
    - 🔍 **Insights**: Descobrir picos de atividade e padrões horários
    - 📋 **Dados Detalhados**: Exportar análises temporais completas
    """)

with col2:
    st.markdown("""
    **Formatos Suportados:**

    - 📄 **CSV**: Arquivos separados por vírgula
    - 📊 **Excel**: Formatos XLSX e XLS
    - 🔧 **JSON**: Dados estruturados em JSON
    - ⚡ **Parquet**: Formato otimizado para análise
    
    **Requisitos do Dataset:**
    
    - 🏷️ Coluna com labels/classes da classificação
    - 📅 Coluna com informações de data/hora
    - 📊 Dados já processados e classificados
    """)

st.markdown("---")
