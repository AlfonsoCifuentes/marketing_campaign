import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from datetime import datetime
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats



# Page configuration
st.set_page_config(
    page_title="Análisis de Campañas de Marketing",
    page_icon="📊",
    layout="wide"
)

# Custom color palette (based on banner colors)
PRIMARY_COLOR = "#1a1a2e"  # Dark blue
SECONDARY_COLOR = "#16213e"  # Medium blue
ACCENT_COLOR = "#4adede"  # Cyan
TEXT_COLOR = "#ffffff"  # White
DARK_BG = "#121212"  # Dark background
MEDIUM_BG = "#2c2c2c"  # Medium dark background

# Custom CSS
st.markdown(f"""
<style>
    /* General text and background colors */
    .st-emotion-cache-18ni7ap {{
        background-color: {DARK_BG};
    }}
    .st-emotion-cache-uf99v8 {{
        background-color: {DARK_BG};
    }}
    .st-emotion-cache-z5fcl4 {{
        padding-top: 0;
    }}
    body {{
        color: {TEXT_COLOR};
        background-color: {DARK_BG};
    }}
    
    /* Main header */
    h1, h2, h3, h4, h5, h6 {{
        color: {ACCENT_COLOR} !important;
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        width: 100%;
        background-color: {DARK_BG};
        border-bottom: 1px solid {SECONDARY_COLOR};
        display: flex;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {PRIMARY_COLOR};
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        color: {TEXT_COLOR};
        flex-grow: 1;
        text-align: center;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {DARK_BG};
        border-bottom: 1px solid {SECONDARY_COLOR};
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {ACCENT_COLOR};
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: {SECONDARY_COLOR};
        color: {ACCENT_COLOR};
    }}
    
    /* Tab hover effect */
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {SECONDARY_COLOR};
        color: {ACCENT_COLOR};
        box-shadow: 0 0 10px {ACCENT_COLOR};
        transition: all 0.3s ease;
    }}
    
    /* Card-like container for sections */
    .insight-card {{
        background-color: {MEDIUM_BG};
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid {ACCENT_COLOR};
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    /* Hero section styling */
    .hero {{
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid {SECONDARY_COLOR};
    }}
    
    /* Data visualization containers */
    .chart-container {{
        background-color: {MEDIUM_BG};
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }}
    
    /* Metric styling */
    .metric-container {{
        background-color: {SECONDARY_COLOR};
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid {ACCENT_COLOR};
    }}
    .metric-value {{
        font-size: 24px;
        font-weight: bold;
        color: {ACCENT_COLOR};
    }}
    .metric-label {{
        font-size: 14px;
        color: {TEXT_COLOR};
    }}
    
    /* Dataframe styling */
    .dataframe {{
        border-radius: 10px;
        overflow: hidden;
    }}
    .dataframe th {{
        background-color: {SECONDARY_COLOR};
        color: {ACCENT_COLOR};
        text-align: center !important;
        padding: 12px !important;
    }}
    .dataframe td {{
        text-align: center !important;
        padding: 10px !important;
    }}
    
    /* Conclusion highlights */
    .conclusion {{
        font-style: italic;
        color: {ACCENT_COLOR};
        border-left: 4px solid {ACCENT_COLOR};
        padding-left: 15px;
        margin: 20px 0;
    }}
    
    /* Custom button */
    .custom-button {{
        background-color: {SECONDARY_COLOR};
        color: {ACCENT_COLOR};
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 5px;
        border: 1px solid {ACCENT_COLOR};
        cursor: pointer;
        transition-duration: 0.4s;
    }}
    .custom-button:hover {{
        background-color: {ACCENT_COLOR};
        color: {DARK_BG};
        box-shadow: 0 0 10px {ACCENT_COLOR};
    }}
</style>
""", unsafe_allow_html=True)

# Load the banner image
try:
    banner_path = os.path.join("img", "banner_logo.jpg")
    banner = Image.open(banner_path)
except FileNotFoundError:
    # Try alternative path if first attempt fails
    try:
        banner_path = os.path.join("..", "img", "banner_logo.jpg")
        banner = Image.open(banner_path)
    except FileNotFoundError:
        banner = None

# Function to load data with caching
@st.cache_data
def load_data():
    try:
        # Try to load from the dataset directory with error handling
        df = pd.read_csv(
            os.path.join("dataset", "marketingcampaigns.csv"),
            # Use Python engine which is more forgiving
            engine='python',
            # Skip bad lines or use on_bad_lines='warn' to see warnings but continue
            on_bad_lines='skip'
        )
        return df
    except FileNotFoundError:
        try:
            # Try alternative path with the same error handling
            df = pd.read_csv(
                os.path.join("..", "dataset", "marketingcampaigns.csv"),
                engine='python',
                on_bad_lines='skip'
            )
            return df
        except FileNotFoundError:
            # Fallback to embedded data
            st.warning("Using embedded dataset as fallback. File paths might not be correctly configured.")
            
            # Create sample data based on observed structure in the notebook
            try:
                data = pd.read_csv(
                    "marketingcampaigns.csv",
                    engine='python',
                    on_bad_lines='skip'
                )
                return data
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                # Return empty DataFrame with expected columns as last resort
                return pd.DataFrame(columns=[
                    'campaign_name', 'start_date', 'end_date', 'budget', 
                    'roi', 'conversion_rate', 'revenue', 'channel', 
                    'type', 'target_audience'
                ])

# Load the data
data = load_data()

# Clean and preprocess data (similar to the notebook)
def preprocess_data(df):
    # Tratar casos específicos de datos negativos como valores faltantes
    mask_negative_budget = df['campaign_name'] == "Negative ROI test"
    mask_negative_revenue = df['campaign_name'] == "Negative Revenue test"

    # Reemplazar valores negativos por NaN
    if mask_negative_budget.any():
        df.loc[mask_negative_budget, 'budget'] = np.nan
        print(f"Reemplazado presupuesto negativo en campaña 'Negative ROI test' por NaN")

    if mask_negative_revenue.any():
        df.loc[mask_negative_revenue, 'revenue'] = np.nan
        print(f"Reemplazado ingreso negativo en campaña 'Negative Revenue test' por NaN")

    # Convert date columns to datetime
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
    
    # Convert numeric columns
    numeric_cols = ['budget', 'roi', 'conversion_rate', 'revenue']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate campaign duration
    df['campaign_duration'] = (df['end_date'] - df['start_date']).dt.days
    
    # Calculate efficiency metrics
    df['revenue_efficiency'] = df['revenue'] / df['budget']
    df['net_profit'] = df['revenue'] - df['budget']
    df['profit_margin'] = (df['net_profit'] / df['revenue'] * 100).round(2)
    
    # Identificar y tratar el caso específico de "Too many conversions"
    mask_too_many_conversions = df['campaign_name'] == "Too many conversions"

# Reemplazar valores de conversión excesivos
    if mask_too_many_conversions.any():
        df.loc[mask_too_many_conversions, 'conversion_rate'] = 1.0
        print(f"Ajustada tasa de conversión para la campaña 'Too many conversions' al valor máximo de 1.0")


    # Handle missing values with median imputation
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill category missing values
    category_cols = ['type', 'target_audience', 'channel']
    for col in category_cols:
        df[col] = df[col].fillna('unknown')
    
    # Fix dates with median duration
    median_duration = pd.to_timedelta(df['campaign_duration'].median(), unit='D')
    for idx, row in df[df['start_date'].isna() | df['end_date'].isna()].iterrows():
        if pd.isna(row['start_date']) and not pd.isna(row['end_date']):
            df.at[idx, 'start_date'] = row['end_date'] - median_duration
        elif pd.isna(row['end_date']) and not pd.isna(row['start_date']):
            df.at[idx, 'end_date'] = row['start_date'] + median_duration
    
    # Fix negative campaign durations by swapping dates
    mask = df['campaign_duration'] < 0
    df.loc[mask, ['start_date', 'end_date']] = df.loc[mask, ['end_date', 'start_date']].values
    df['campaign_duration'] = (df['end_date'] - df['start_date']).dt.days
    
    # Cap conversion rate to logical maximum
    df['conversion_rate'] = df['conversion_rate'].clip(upper=1.0)
    
    # Fix channel naming inconsistencies
    df['channel'] = df['channel'].replace({
        'referal': 'referral', 
        'promoton': 'promotion', 
        'orgnic': 'organic'
    })
    
    # Normalize audience values
    df['target_audience'] = df['target_audience'].replace({
        'b2b': 'B2B', 'b2c': 'B2C', 
        'BtoB': 'B2B', 'BtoC': 'B2C'
    })

    df = df[~((df['campaign_name'] == 'Outlier Budget') & (df['channel'] == 'promotion'))]
    print(f"Se ha eliminado la campaña 'Outlier Budget' del canal 'promotion' por tener un presupuesto atípico extremo.")

    # Limit negative values and percentage ranges

    df['budget'] = df['budget'].clip(lower=0)
    df['revenue'] = df['revenue'].clip(lower=0)
    df['roi'] = df['roi'].clip(lower=0)
    df['conversion_rate'] = df['conversion_rate'].clip(lower=0, upper=1)
    df['net_profit'] = df['net_profit'].clip(lower=0)
    df['profit_margin'] = df['profit_margin'].clip(lower=0, upper=100)
    df['revenue_efficiency'] = df['revenue_efficiency'].clip(lower=0)
    df['campaign_duration'] = df['campaign_duration'].clip(lower=0)
  




    return df

# Quitando outliers extremos usando el método IQR

def remove_outliers(df):
    """Detecta y elimina outliers usando el método IQR en variables clave"""
    
    # Columnas numéricas a analizar
    numerical_cols = ['budget', 'revenue', 'roi', 'conversion_rate', 'net_profit']
    
    # Cantidad de filas antes de eliminar outliers
    rows_before = len(df)
    
    print("Eliminando outliers extremos...")
    
    # Para cada columna, detectar y eliminar outliers usando IQR
    for col in numerical_cols:
        # Verificar que la columna es numérica
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Columna {col} no es numérica. Tipo: {df[col].dtype}. Omitiendo...")
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Consideramos outliers extremos (más de 3 veces el IQR)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Filtrar outliers extremos
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if len(outliers) > 0:
            print(f"  • {len(outliers)} outliers extremos detectados en '{col}'")
            for _, row in outliers.iterrows():
                print(f"    - '{row['campaign_name']}': {col} = {row[col]}")
            
            # Filtrar el dataframe para excluir estos outliers
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Cantidad de filas eliminadas
    rows_removed = rows_before - len(df)
    rows_percentage = (rows_removed / rows_before) * 100
    print(f"Se eliminaron {rows_removed} registros ({rows_percentage:.2f}% del total) por contener valores atípicos extremos.")
    
    return df


# Primero, cargar los datos
data = load_data()

# Apply preprocessing
data = preprocess_data(data)

# Para usar la función, añade esta línea después del preprocesamiento inicial
data = remove_outliers(data)

# Create a positive size column for scatter plots
data['budget_size'] = data['budget'].abs() + 1  # Adding 1 ensures no zeros

# Define color maps for consistent visualization
channel_colors = {
    'Social Media': '#3498db',   # Blue
    'Email': '#2ecc71',          # Green
    'Webinar': '#e74c3c',        # Red
    'Podcast': '#f39c12',        # Orange
    'Organic': '#9b59b6',        # Purple
    'Paid': '#1abc9c',           # Turquoise
    'Promotion': '#d35400',      # Burnt Orange
    'Referral': '#34495e'        # Dark Blue
}

# Create a function to determine colors for channels
def get_channel_color(channel):
    for key in channel_colors:
        if key.lower() in channel.lower():
            return channel_colors[key]
    return '#95a5a6'  # Default gray for unknown channels

# Hero section with banner
if banner is not None:
    st.image(banner, use_container_width=True)

st.markdown("""
<div class="hero">
    <h1>Análisis de Campañas de Marketing</h1>
    <p>Exploración, limpieza y análisis avanzado de datos de marketing para descubrir insights clave sobre presupuestos, ingresos, ROI y audiencias.</p>
</div>
""", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Datos y Limpieza", 
    "📊 Análisis Exploratorio", 
    "📈 Patrones y Tendencias", 
    "💡 Conclusiones y Recomendaciones"
])

# Tab 1: Data and Cleaning
with tab1:
    # Create subtabs
    subtab1_1, subtab1_2, subtab1_3 = st.tabs(["📋 Resumen de Datos", "🔍 Detección y Tratamiento", "✅ Resultados de Limpieza"])
    
    with subtab1_1:
        st.markdown("""
        <div class="insight-card">
            <h3>Exploración Inicial del Dataset</h3>
            <p>El conjunto de datos contiene información sobre campañas de marketing, incluyendo nombres, fechas, presupuestos, ROI y métricas de rendimiento.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### Primeras filas del dataset")
            st.dataframe(data.head(5), height=250)
        
        with col2:
            st.markdown("### Estructura del dataset")
            st.markdown(f"""
            - 📊 **Filas**: {data.shape[0]}
            - 📋 **Columnas**: {data.shape[1]}
            - 📅 **Rango temporal**: {data['start_date'].min().strftime('%Y-%m-%d')} a {data['end_date'].max().strftime('%Y-%m-%d')}
            """)
            
            st.markdown("### Tipos de datos")
            for col, dtype in zip(data.columns, data.dtypes):
                st.markdown(f"- **{col}**: `{dtype}`")
    
    with subtab1_2:
        st.markdown("""
        <div class="insight-card">
            <h3>Problemas Identificados y Soluciones Aplicadas</h3>
            <p>Durante la exploración inicial, se detectaron varios problemas en los datos que requerían corrección para garantizar un análisis preciso.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for missing values visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Valores Faltantes por Columna (Antes de la limpieza)")
            # Calculate missing values before cleaning
            missing_values_before = {
            'start_date': 3,
            'end_date': 3,
            'budget': 4,
            'roi': 4,
            'type': 1,
            'target_audience': 2,
            'channel': 1,
            'conversion_rate': 4,
            'revenue': 3,
            'duracion_campaña': 5,
            'eficiencia_ingresos': 6
            }
            
            missing_df = pd.DataFrame({
            'Columna': list(missing_values_before.keys()),
            'Valores Faltantes': list(missing_values_before.values())
            })
            
            # Plot missing values before - increased figure width
            fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
            ax.set_facecolor(MEDIUM_BG)
            bars = ax.barh(missing_df['Columna'], missing_df['Valores Faltantes'], color=ACCENT_COLOR)
            ax.set_xlabel('Cantidad de Valores Faltantes', color='white')
            ax.set_title(' ', color='white', fontweight='bold')
            
            # Set colors for spines, ticks, labels
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            
            # Add value annotations
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width}', ha='left', va='center', color='white')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Adding empty space between columns to maximize separation
        st.markdown("<div style='width:100%; padding:10px;'></div>", unsafe_allow_html=True)
        
        with col2:
            # Adding padding to improve separation
            st.markdown("""
            <div style="padding-left:25px;">
            <h3>Problemas Identificados:</h3>
            
            <ul>
                <li>❗ <strong>Valores Faltantes</strong>: Presentes en múltiples columnas como presupuesto, ROI, fechas</li>
                <li>🔄 <strong>Tipos de Datos Incorrectos</strong>: Fechas y números almacenados como texto</li>
                <li>⚠️ <strong>Inconsistencias</strong>: Fechas invertidas, tasas de conversión mayores a 1, etc.</li>
                <li>📉 <strong>Valores Atípicos</strong>: Presupuestos e ingresos extremadamente altos</li>
                <li>📌 <strong>**Nota**</strong>: Se ha excluido la campaña 'Outlier Budget' del canal 'promotion' debido a un presupuesto extremadamente alto (9,999,999) que distorsionaba los análisis.</li>        
            </ul>

            <h3>Proceso de Limpieza:</h3>

            <ul>
                <li>✅ <strong>Formateo de Tipos</strong>: Conversión a los tipos de datos correctos:
                <ul>
                    <li><code>datetime64[ns]</code> para fechas</li>
                    <li><code>float64</code> para métricas numéricas</li>
                </ul>
                </li>
                <li>🔄 <strong>Corrección de Fechas Invertidas</strong>: Intercambio de fechas para asegurar duraciones positivas</li>
                <li>📊 <strong>Imputación de Valores Faltantes</strong>:
                <ul>
                    <li>Columnas numéricas: Imputadas con la mediana para minimizar el impacto de valores atípicos</li>
                    <li>Columnas categóricas: Asignado "unknown" a los campos vacíos</li>
                    <li>Fechas: Estimadas usando la duración mediana de campañas</li>
                </ul>
                </li>
                <li>📈 <strong>Limitación de Valores Ilógicos</strong>:
                <ul>
                    <li>Tasas de conversión limitadas a un máximo de 1.0 (100%)</li>
                </ul>
                </li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
  

    with subtab1_3:
        st.markdown("""
        <div class="insight-card">
            <h3>Resultados del Proceso de Limpieza</h3>
            <p>Tras el proceso de limpieza y transformación, los datos están listos para el análisis con las siguientes mejoras y características:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">1032</div>
                <div class="metric-label">Campañas Analizadas</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${data['budget'].sum():,.0f}</div>
                <div class="metric-label">Presupuesto Total</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${data['net_profit'].sum():,.0f}</div>
                <div class="metric-label">Beneficio Neto Total</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{data['roi'].mean():.2f}</div>
                <div class="metric-label">ROI Medio</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Variables Calculadas")
        
        # Show calculated features
        calculated_features = pd.DataFrame({
            'Variable': ['campaign_duration', 'revenue_efficiency', 'net_profit', 'profit_margin'],
            'Descripción': [
                'Duración de la campaña en días',
                'Eficiencia de ingresos (Ingresos/Presupuesto)',
                'Beneficio neto (Ingresos - Presupuesto)',
                'Margen de beneficio en %'
            ],
            'Ejemplo': [
                f"{data['campaign_duration'].mean():.1f} días (promedio)",
                f"{data['revenue_efficiency'].mean():.2f} (promedio)",
                f"${data['net_profit'].iloc[0]:,.2f} (primera campaña)",
                f"{data['profit_margin'].iloc[0]:.1f}% (primera campaña)"
            ]
        })
        
        st.dataframe(calculated_features, use_container_width=True, hide_index=True)

        st.markdown("### Muestra del dataset una vez limpio y procesado (10 filas)")
        st.dataframe(data.head(10), use_container_width=True)
        
        st.markdown("""
        ### Mejoras Conseguidas:

        - ✅ **Conjunto de datos completo**: Sin valores faltantes en ninguna columna
        - 📆 **Fechas consistentes**: Todas las campañas tienen ahora duración positiva
        - 📊 **Métricas coherentes**: Tasas de conversión en el rango lógico (0-100%)
        - 📈 **Análisis enriquecido**: Nuevas métricas calculadas para evaluar rendimiento
        - 🔍 **Control de valores atípicos**: Eliminación de campañas con presupuestos extremos que distorsionaban el análisis
        - 📉 **Corrección de valores negativos**: Tratamiento de ingresos y presupuestos negativos para mantener la consistencia de datos
        """)

# Tab 2: Exploratory Data Analysis
with tab2:
    subtab2_1, subtab2_2, subtab2_3, subtab2_4 = st.tabs([
        "📊 Canales de Marketing", 
        "👥 Tipos de Audiencia", 
        "📋 Tipos de Campaña", 
        "💰 Análisis de Rentabilidad"
    ])
    
    with subtab2_1:
        st.markdown("""
        <div class="insight-card">
            <h3>Análisis de Canales de Marketing</h3>
            <p>Exploramos qué canales son más utilizados y cuáles generan mejores resultados en términos de ROI, ingresos y eficiencia.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # First analysis: most used channels
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Canales de Marketing más Utilizados")
            
            # Count frequency of each channel
            channel_counts = data['channel'].value_counts().reset_index()
            channel_counts.columns = ['channel', 'count']
            
            # Plot with Plotly for better interactivity
            fig = px.bar(
                channel_counts,
                x='channel',
                y='count',
                color='channel',
                title=' ',
                labels={'count': 'Número de Campañas', 'channel': 'Canal de Marketing'},
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            
            # Update layout for dark theme
            fig.update_layout(
                plot_bgcolor=MEDIUM_BG,
                paper_bgcolor=MEDIUM_BG,
                font_color='white',
                title_font_color=ACCENT_COLOR,
                legend_title_font_color='white',
                hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
                title={
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            
            # Add percentage labels
            total_campaigns = channel_counts['count'].sum()
            
            for i, row in channel_counts.iterrows():
                percentage = (row['count'] / total_campaigns) * 100
                fig.add_annotation(
                    x=row['channel'],
                    y=row['count'],
                    text=f"{row['count']} ({percentage:.1f}%)",
                    showarrow=False,
                    yshift=10,
                    font_color='white'
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("### ROI Promedio por Canal")
            
            # Calculate average ROI by channel
            channel_roi = data.groupby('channel')['roi'].mean().sort_values(ascending=False).reset_index()
            
            # Create horizontal bar chart with Plotly
            fig = px.bar(
                channel_roi,
                y='channel',
                x='roi',
                orientation='h',
                color='roi',
                title=' ',
                labels={'roi': 'ROI Promedio', 'channel': 'Canal de Marketing'},
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            # Update layout for dark theme
            fig.update_layout(
                plot_bgcolor=MEDIUM_BG,
                paper_bgcolor=MEDIUM_BG,
                font_color='white',
                title_font_color=ACCENT_COLOR,
                legend_title_font_color='white',
                hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
                title={
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            
            # Add value annotations
            for i, row in channel_roi.iterrows():
                fig.add_annotation(
                    y=row['channel'],
                    x=row['roi'],
                    text=f"{row['roi']:.2f}",
                    showarrow=False,
                    xshift=10,
                    font_color='white'
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Second part of the analysis: budget vs revenue by channel
        st.markdown("<h3 style='text-align: center;'>Presupuesto e Ingresos Medios por Canal</h3>", unsafe_allow_html=True)
        
        # Prepare data
        channel_stats = data.groupby('channel').agg({
            'budget': 'mean',
            'revenue': 'mean',
            'conversion_rate': 'mean',
            'roi': 'mean',
            'revenue_efficiency': 'mean'
        }).reset_index()
        
        # Create figure with matplotlib for more control over styling
        fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK_BG)
        ax.set_facecolor(MEDIUM_BG)
        
        # Bar width and positions
        x = np.arange(len(channel_stats))
        width = 0.35
        
        # Create bars
        budget_bars = ax.bar(x - width/2, channel_stats['budget'], width, 
                             label='Presupuesto Medio ($)', color='#00bc8c')
        revenue_bars = ax.bar(x + width/2, channel_stats['revenue'], width, 
                              label='Ingreso Medio ($)', color='#f39c12')
        
        # Add title and labels
        ax.set_title(' ', 
                     fontsize=16, color='white', fontweight='bold')
        ax.set_xlabel('Canal de Marketing', fontsize=12, color='white')
        ax.set_ylabel('Monto ($)', fontsize=12, color='white')
        
        # Set up X ticks
        ax.set_xticks(x)
        ax.set_xticklabels(channel_stats['channel'], color='white', rotation=45, ha='right')
        
        # Set Y axis labels color
        ax.tick_params(colors='white')
        
        # Add legend
        legend = ax.legend(facecolor=MEDIUM_BG, edgecolor='gray', loc='upper right')
        for text in legend.get_texts():
            text.set_color('white')
        
        # Add value labels to bars
        def add_value_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'${height:,.0f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color='white')
        
        add_value_labels(budget_bars)
        add_value_labels(revenue_bars)
        
        # Set spines color
        for spine in ax.spines.values():
            spine.set_color('white')
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        # Show plot
        st.pyplot(fig)

        # Calcular estadísticas actualizadas
        email_roi = data[data['channel'] == 'Email']['roi'].mean()
        social_media_revenue = data[data['channel'] == 'Social Media']['revenue'].mean()
        social_media_budget = data[data['channel'] == 'Social Media']['budget'].mean()
        tv_revenue = data[data['channel'] == 'TV']['revenue'].mean()
        tv_budget = data[data['channel'] == 'TV']['budget'].mean()

        # Insights
        st.markdown(f"""
        <div class="conclusion">
            <h3>📈 Hallazgos sobre Canales</h3>
            <ul>
                <li><strong>Email Marketing</strong> destaca como el canal más <strong>costo-efectivo</strong>, generando altos ingresos con inversiones relativamente bajas (ROI promedio: {email_roi:.2f})</li>
                <li><strong>Social Media</strong> genera los <strong>mayores ingresos medios</strong> (${social_media_revenue:,.0f}) pero también utiliza el <strong>mayor presupuesto medio</strong> (${social_media_budget:,.0f})</li>
                <li><strong>TV</strong> presenta el <strong>menor retorno</strong> en términos de ingresos absolutos (${tv_revenue:,.0f})</li>
                <li>Existe una <strong>correlación positiva</strong> entre presupuesto e ingresos, pero no es proporcional en todos los canales</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with subtab2_2:
        b2b_mean = data[data['target_audience'] == 'B2B']['conversion_rate'].mean()
        b2c_mean = data[data['target_audience'] == 'B2C']['conversion_rate'].mean()
        professionals_revenue = data[data['target_audience'] == 'B2B']['revenue'].mean()
        students_efficiency = data[data['target_audience'] == 'Students']['revenue_efficiency'].mean() if 'Students' in data['target_audience'].unique() else 0
        young_adults_roi = data[data['target_audience'] == 'Young Adults']['roi'].mean() if 'Young Adults' in data['target_audience'].unique() else 0
        seniors_revenue = data[data['target_audience'] == 'Seniors']['revenue'].mean() if 'Seniors' in data['target_audience'].unique() else 0

        # Actualizar el markdown de hallazgos para usar valores dinámicos
        st.markdown(f"""                    
        <div class="conclusion">
            <h3>📈 Hallazgos sobre Audiencias</h3>
            <ul>
                <li><strong>B2B</strong> muestra mayor consistencia en resultados (menor desviación estándar)</li>
                <li><strong>B2C</strong> tiene una tasa de conversión ligeramente superior (aproximadamente {b2c_mean:.2f} vs {b2b_mean:.2f})</li>
                <li>La diferencia entre ambos segmentos no es estadísticamente significativa</li>
                <li><strong>Profesionales (B2B)</strong> generan los <strong>ingresos más altos</strong> (${professionals_revenue:,.0f})</li>
                <li><strong>Estudiantes</strong> muestran la <strong>mejor eficiencia de ingresos</strong> ({students_efficiency:.2f})</li>
                <li><strong>Adultos jóvenes</strong> presentan el <strong>mejor ROI</strong> ({young_adults_roi:.2f})</li>
                <li><strong>Seniors</strong> generan los <strong>menores ingresos</strong> (${seniors_revenue:,.0f}) a pesar de recibir presupuestos considerables</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
            
        # First section: metrics by audience
        audience_stats = data.groupby('target_audience').agg({
            'budget': 'mean',
            'revenue': 'mean',
            'roi': 'mean',
            'conversion_rate': 'mean',
            'revenue_efficiency': 'mean',
            'net_profit': 'mean',
            'campaign_name': 'count'
        }).reset_index()
        
        audience_stats.columns = ['Audiencia Objetivo', 'Presupuesto Medio', 'Ingreso Medio', 'ROI Medio', 
                                 'Tasa Conversión Media', 'Eficiencia Ingresos', 'Beneficio Neto Medio', 'Número Campañas']
        
        # Format the dataframe for display
        audience_stats_display = audience_stats.copy()
        audience_stats_display['Presupuesto Medio'] = audience_stats_display['Presupuesto Medio'].map('${:,.2f}'.format)
        audience_stats_display['Ingreso Medio'] = audience_stats_display['Ingreso Medio'].map('${:,.2f}'.format)
        audience_stats_display['ROI Medio'] = audience_stats_display['ROI Medio'].map('{:.2f}'.format)
        audience_stats_display['Tasa Conversión Media'] = audience_stats_display['Tasa Conversión Media'].map('{:.1%}'.format)
        audience_stats_display['Eficiencia Ingresos'] = audience_stats_display['Eficiencia Ingresos'].map('{:.2f}'.format)
        audience_stats_display['Beneficio Neto Medio'] = audience_stats_display['Beneficio Neto Medio'].map('${:,.2f}'.format)
        
        st.markdown("### Métricas por Audiencia Objetivo")
        st.dataframe(audience_stats_display, use_container_width=True, hide_index=True)
        
        # Second section: conversion rates comparison
        st.markdown("### Comparativa de Tasas de Conversión: B2B vs B2C")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Filter data
            b2b_data = data[data['target_audience'] == 'B2B']
            b2c_data = data[data['target_audience'] == 'B2C']
            
            # Calculate statistics
            b2b_stats = {
                'mean': b2b_data['conversion_rate'].mean(),
                'median': b2b_data['conversion_rate'].median(),
                'std': b2b_data['conversion_rate'].std(),
                'count': len(b2b_data)
            }
            
            b2c_stats = {
                'mean': b2c_data['conversion_rate'].mean(),
                'median': b2c_data['conversion_rate'].median(),
                'std': b2c_data['conversion_rate'].std(),
                'count': len(b2c_data)
            }
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
            ax.set_facecolor(MEDIUM_BG)
            
            segments = ['B2B', 'B2C']
            means = [b2b_stats['mean'], b2c_stats['mean']]
            stds = [b2b_stats['std'], b2c_stats['std']]
            
            colors = ['#4adeee', '#f39c12']  # Eliminando el carácter 'b' extra al final
            bar_positions = np.arange(len(segments))
            bars = ax.bar(bar_positions, means, yerr=stds, capsize=10, 
                         color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            
            # Add labels and formatting
            ax.set_title('Tasa de Conversión Media por Segmento', fontsize=14, color='white')
            ax.set_ylabel('Tasa de Conversión', fontsize=12, color='white')
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(segments, fontsize=12, color='white')
            
            # Set colors for axes
            ax.tick_params(axis='y', colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            
            # Add grid
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                count = b2b_stats['count'] if i == 0 else b2c_stats['count']
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                     f'{height:.3f}\n(n={count})',
                     ha='center', va='bottom', fontsize=11, color='white',
                     fontweight='bold')
            
            # Add global mean reference line
            global_mean = data['conversion_rate'].mean()
            ax.axhline(y=global_mean, color='#f39c12', linestyle='--', alpha=0.7)
            ax.text(0.5, global_mean + 0.02, f'Media Global: {global_mean:.3f}', 
                   ha='center', color='#f39c12', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Create boxplot for distribution comparison
            fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
            ax.set_facecolor(MEDIUM_BG)
            
            boxplot_data = [b2b_data['conversion_rate'], b2c_data['conversion_rate']]
            
            box = ax.boxplot(boxplot_data, patch_artist=True, 
                tick_labels=segments,  # Cambiado de 'labels' a 'tick_labels'
                widths=0.6,
                flierprops={'marker': 'o', 'markersize': 8, 'markerfacecolor': 'white'})
            
            # Customize boxplot colors
            for i, patch in enumerate(box['boxes']):
                patch.set_facecolor(colors[i])
                
            for i, element in enumerate(['whiskers', 'caps', 'medians']):
                for item in box[element]:
                    item.set_color('white')
            
            # Add title and labels
            ax.set_title('Distribución de Tasas de Conversión', fontsize=14, color='white')
            ax.set_ylabel('Tasa de Conversión', fontsize=12, color='white')
            
            # Set colors for axes
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            
            # Add grid
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Add global mean reference line
            ax.axhline(y=global_mean, color='#f39c12', linestyle='--', alpha=0.7, 
                      label=f'Media Global: {global_mean:.3f}')
            
            # Add legend
            ax.legend(loc='upper right', facecolor=MEDIUM_BG, edgecolor='gray')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        # Statistical comparison
        from scipy import stats
        
        t_stat, p_value = stats.ttest_ind(b2b_data['conversion_rate'], b2c_data['conversion_rate'], equal_var=False)
        
        st.markdown(f"""
        <div class="insight-card">
            <h4>Análisis Estadístico de la Diferencia</h4>
            <ul>
                <li><strong>Diferencia de medias</strong>: {b2c_stats['mean'] - b2b_stats['mean']:.3f}</li>
                <li><strong>Prueba t</strong>: t={t_stat:.3f}, p={p_value:.3f}</li>
                <li><strong>{'Diferencia estadísticamente significativa' if p_value < 0.05 else 'No hay diferencia estadísticamente significativa'}</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Insights
        st.markdown("""
        <div class="conclusion">
            <h3>📈 Hallazgos sobre Audiencias</h3>
            <ul>
                <li><strong>B2B</strong> muestra mayor consistencia en resultados (menor desviación estándar)</li>
                <li><strong>B2C</strong> tiene una tasa de conversión ligeramente superior (aproximadamente 0.58 vs 0.52)</li>
                <li>La diferencia entre ambos segmentos no es estadísticamente significativa</li>
                <li><strong>Profesionales (B2B)</strong> generan los <strong>ingresos más altos</strong> ($52,271)</li>
                <li><strong>Estudiantes</strong> muestran la <strong>mejor eficiencia de ingresos</strong> (3.80)</li>
                <li><strong>Adultos jóvenes</strong> presentan el <strong>mejor ROI</strong> (1.98)</li>
                <li><strong>Seniors</strong> generan los <strong>menores ingresos</strong> ($44,616) a pesar de recibir presupuestos considerables</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with subtab2_3:
        # Calcular métricas para cada tipo de campaña
        awareness_stats = data[data['type'] == 'Awareness'].agg({
            'revenue': 'mean',
            'net_profit': 'mean',
            'roi': 'mean',
            'conversion_rate': 'mean',
            'profit_margin': 'mean'
        }).to_dict()
        
        conversion_stats = data[data['type'] == 'Conversion'].agg({
            'revenue': 'mean',
            'net_profit': 'mean',
            'roi': 'mean',
            'conversion_rate': 'mean',
            'profit_margin': 'mean'
        }).to_dict()
        
        retention_stats = data[data['type'] == 'Retention'].agg({
            'revenue': 'mean',
            'net_profit': 'mean',
            'roi': 'mean',
            'conversion_rate': 'mean',
            'profit_margin': 'mean'
        }).to_dict()
        
        # Actualizar el markdown con valores dinámicos
        st.markdown(f"""
        <div class="conclusion">
        <h3>📈 Hallazgos por Tipo de Campaña</h3>
        <h4>🥇 Campañas de Awareness (${awareness_stats['revenue']:,.0f})</h4>
        <ul>
        <li><strong>Mayor generador de ingresos promedio</strong></li>
        <li><strong>Mayor beneficio neto medio</strong>: ${awareness_stats['net_profit']:,.0f}</li>
        <li>ROI de {awareness_stats['roi']:.2f}</li>
        <li>Tasa de conversión del {awareness_stats['conversion_rate']*100:.1f}% (la más baja entre las categorías)</li>
        <li>Margen de beneficio del {awareness_stats['profit_margin']:.1f}%</li>
        </ul>
        
        <h4>🥈 Campañas de Conversion (${conversion_stats['revenue']:,.0f})</h4>
        <ul>
        <li>Segundo lugar en ingresos promedio</li>
        <li>Beneficio neto medio: ${conversion_stats['net_profit']:,.0f}</li>
        <li><strong>Mejor ROI</strong> ({conversion_stats['roi']:.2f})</li>
        <li><strong>Mayor tasa de conversión</strong> ({conversion_stats['conversion_rate']*100:.1f}%)</li>
        <li><strong>Mejor margen de beneficio</strong>: {conversion_stats['profit_margin']:.1f}%</li>
        </ul>
        
        <h4>🥉 Campañas de Retention (${retention_stats['revenue']:,.0f})</h4>
        <ul>
        <li>Tercer lugar en ingresos promedio</li>
        <li>Beneficio neto medio: ${retention_stats['net_profit']:,.0f} (el más bajo)</li>
        <li>ROI intermedio de {retention_stats['roi']:.2f}</li>
        <li>Buena tasa de conversión ({retention_stats['conversion_rate']*100:.1f}%)</li>
        <li><strong>Mayor número de campañas</strong></li>
        <li>Margen de beneficio más bajo: {retention_stats['profit_margin']:.1f}%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare data
        campaign_type_stats = data.groupby('type').agg({
            'revenue': 'mean',
            'budget': 'mean',
            'net_profit': 'mean',
            'roi': 'mean',
            'conversion_rate': 'mean',
            'profit_margin': 'mean',
            'campaign_name': 'count'
        }).reset_index()
        
        # Sort by revenue
        campaign_type_stats = campaign_type_stats.sort_values('revenue', ascending=False)
        
        # Create visualization
        st.markdown("### Ingresos, Presupuestos y Beneficio Neto por Tipo de Campaña")
        
        fig = go.Figure()
        
        # Add bars for revenue, budget, and profit
        fig.add_trace(go.Bar(
            x=campaign_type_stats['type'],
            y=campaign_type_stats['revenue'],
            name='Ingreso Medio ($)',
            marker_color='#f39c12',
            text=[f'${val:,.0f}' for val in campaign_type_stats['revenue']],
            textposition='outside',
            width=0.2  # Reduced from 0.25 for more space between bars
        ))
        
        fig.add_trace(go.Bar(
            x=campaign_type_stats['type'],
            y=campaign_type_stats['budget'],
            name='Presupuesto Medio ($)',
            marker_color='#00bc8c',
            text=[f'${val:,.0f}' for val in campaign_type_stats['budget']],
            textposition='outside',
            width=0.2  # Reduced from 0.25 for more space between bars
        ))
        
        fig.add_trace(go.Bar(
            x=campaign_type_stats['type'],
            y=campaign_type_stats['net_profit'],
            name='Beneficio Neto Medio ($)',
            marker_color='#3498db',
            text=[f'${val:,.0f}' for val in campaign_type_stats['net_profit']],
            textposition='outside',
            width=0.2  # Reduced from 0.25 for more space between bars
        ))
        
        # Update layout with increased margins
        fig.update_layout(
            title=' ',
            xaxis_title='Tipo de Campaña',
            yaxis_title='Monto ($)',
            barmode='group',
            plot_bgcolor=MEDIUM_BG,
            paper_bgcolor=MEDIUM_BG,
            font_color='white',
            title_font_color=ACCENT_COLOR,
            legend_title_font_color='white',
            hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
            margin=dict(l=50, r=50, t=30, b=50),  # Added explicit margins
            xaxis=dict(
                tickangle=-45,  # Angled labels for better readability
                tickfont=dict(size=12),  # Adjusted text size
                automargin=True  # Ensures margin is automatically adjusted if needed
            ),
            yaxis=dict(
                tickfont=dict(size=12),  # Adjusted text size
                automargin=True  # Ensures margin is automatically adjusted if needed
            )
        )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics table
        st.markdown("### Métricas Detalladas por Tipo de Campaña")
        
        # Format the dataframe for display
        type_stats_display = campaign_type_stats.copy()
        type_stats_display.columns = ['Tipo de Campaña', 'Ingreso Medio', 'Presupuesto Medio', 
                                     'Beneficio Neto Medio', 'ROI Medio', 'Tasa Conversión Media',
                                     'Margen Beneficio', 'Cantidad Campañas']
        
        type_stats_display['Ingreso Medio'] = type_stats_display['Ingreso Medio'].map('${:,.0f}'.format)
        type_stats_display['Presupuesto Medio'] = type_stats_display['Presupuesto Medio'].map('${:,.0f}'.format)
        type_stats_display['Beneficio Neto Medio'] = type_stats_display['Beneficio Neto Medio'].map('${:,.0f}'.format)
        type_stats_display['ROI Medio'] = type_stats_display['ROI Medio'].map('{:.2f}'.format)
        type_stats_display['Tasa Conversión Media'] = type_stats_display['Tasa Conversión Media'].map('{:.1%}'.format)
        type_stats_display['Margen Beneficio'] = type_stats_display['Margen Beneficio'].map('{:.1f}%'.format)
        
        st.dataframe(type_stats_display, use_container_width=True, hide_index=True)
        
        # Campaigns by Type visualization with findings side by side
        st.markdown("### Distribución de Campañas por Tipo")
        
        # Create a custom container with CSS flexbox for alignment
        st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: flex-end ; gap: 20px;">
            <div style="flex: 1;">
                <div id="pie-chart-container"></div>
            </div>
            <div style="flex: 1;">
                <div id="findings-container"></div>
            </div>
            </div>
        """, unsafe_allow_html=True)

        # Create column layout (we'll still use these for the content)
        col_chart, col_findings = st.columns([1, 1])
        
        with col_chart:
            # Create a pie chart
            type_counts = data['type'].value_counts().reset_index()
            type_counts.columns = ['type', 'count']
            
            fig = px.pie(
            type_counts,
            values='count',
            names='type',
            title=' ',
            color_discrete_sequence=px.colors.sequential.Viridis,
            hole=0.4,
            height=400  # Set explicit height to match findings text box
            )
            
            # Update layout for dark theme
            fig.update_layout(
            plot_bgcolor=MEDIUM_BG,
            paper_bgcolor=MEDIUM_BG,
            font_color='white',
            title_font_color=ACCENT_COLOR,
            legend_title_font_color='white',
            hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
            margin=dict(t=20, b=20, l=20, r=20)  # Add margins around the chart
            )
            
            # Update traces for better readability
            fig.update_traces(
            textinfo='percent+label',
            textfont_color='white'
            )
            
            # Wrap with a container for consistent height
            with st.container():
                st.plotly_chart(fig, use_container_width=True)
        
         
        # Insights in the right column
        with col_findings:
            st.markdown("""
            <div class="conclusion">
            <h3>📈 Hallazgos por Tipo de Campaña</h3>
            <h4>🥇 Campañas de Awareness ($53,909)</h4>
            <ul>
            <li><strong>Mayor generador de ingresos promedio</strong></li>
            <li><strong>Mayor beneficio neto medio</strong>: $27,470</li>
            <li>ROI de 1.63</li>
            <li>Tasa de conversión del 47.7% (la más baja entre las categorías)</li>
            <li>Margen de beneficio del 23.2%</li>
            </ul>
            
            <h4>🥈 Campañas de Conversion ($49,844)</h4>
            <ul>
            <li>Segundo lugar en ingresos promedio</li>
            <li>Beneficio neto medio: $26,734</li>
            <li><strong>Mejor ROI</strong> (1.94)</li>
            <li><strong>Mayor tasa de conversión</strong> (59.7%)</li>
            <li><strong>Mejor margen de beneficio</strong>: 33.4%</li>
            </ul>
            
            <h4>🥉 Campañas de Retention ($45,361)</h4>
            <ul>
            <li>Tercer lugar en ingresos promedio</li>
            <li>Beneficio neto medio: $20,098 (el más bajo)</li>
            <li>ROI intermedio de 1.79</li>
            <li>Buena tasa de conversión (59.3%)</li>
            <li><strong>Mayor número de campañas</strong></li>
            <li>Margen de beneficio más bajo: 6.7%</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    with subtab2_4:
        st.markdown("""
        <div class="insight-card">
            <h3>Análisis de Rentabilidad</h3>
            <p>Examinamos qué campañas generan mayor beneficio neto y qué factores contribuyen al éxito financiero de las campañas.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top and bottom campaigns by profit
        st.markdown("### Campañas con Mayor y Menor Beneficio Neto")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Top 10 most profitable campaigns
            st.markdown("#### Top 10 Campañas Más Rentables")
            
            top_campaigns = data.sort_values('net_profit', ascending=False).head(10)
            top_campaigns_display = top_campaigns[['campaign_name', 'channel', 'target_audience', 'net_profit', 'profit_margin']].copy()
            top_campaigns_display.columns = ['Campaña', 'Canal', 'Audiencia', 'Beneficio Neto', 'Margen']
            
            top_campaigns_display['Beneficio Neto'] = top_campaigns_display['Beneficio Neto'].map('${:,.0f}'.format)
            top_campaigns_display['Margen'] = top_campaigns_display['Margen'].map('{:.1f}%'.format)
            
            st.dataframe(top_campaigns_display, use_container_width=True, hide_index=True)
            
        with col2:
            # Bottom 5 least profitable campaigns
            st.markdown("#### 5 Campañas Menos Rentables")
            
            bottom_campaigns = data.sort_values('net_profit', ascending=True).head(5)
            bottom_campaigns_display = bottom_campaigns[['campaign_name', 'channel', 'target_audience', 'net_profit', 'profit_margin']].copy()
            bottom_campaigns_display.columns = ['Campaña', 'Canal', 'Audiencia', 'Beneficio Neto', 'Margen']
            
            bottom_campaigns_display['Beneficio Neto'] = bottom_campaigns_display['Beneficio Neto'].map('${:,.0f}'.format)
            bottom_campaigns_display['Margen'] = bottom_campaigns_display['Margen'].map('{:.1f}%'.format)
            
            st.dataframe(bottom_campaigns_display, use_container_width=True, hide_index=True)
        
        # Profit distribution visualization
        st.markdown("### Distribución del Beneficio Neto")
        
        # Create profit bins
        profit_bins = [-1000, 0, 20000, 40000, 60000, 80000, 100000]  # Ajustado para datos sin outliers extremos
        profit_labels = ['Pérdida', '0-20K', '20K-40K', '40K-60K', '60K-80K', '80K+']
        
        data['profit_category'] = pd.cut(data['net_profit'], bins=profit_bins, labels=profit_labels)
        profit_distribution = data['profit_category'].value_counts().reset_index()
        profit_distribution.columns = ['Categoría', 'Número de Campañas']
        
        # Calculate percentages
        total_campaigns = profit_distribution['Número de Campañas'].sum()
        profit_distribution['Porcentaje'] = (profit_distribution['Número de Campañas'] / total_campaigns * 100).round(1)
        
        # Sort by category for better visualization
        category_order = ['Pérdida', '0-20K', '20K-40K', '40K-60K', '60K-80K', '80K+']
        profit_distribution['Categoría'] = pd.Categorical(profit_distribution['Categoría'], categories=category_order)
        profit_distribution = profit_distribution.sort_values('Categoría')
        
        # Create bar chart
        fig = px.bar(
            profit_distribution,
            x='Categoría',
            y='Número de Campañas',
            color='Categoría',
            text='Porcentaje',
            title='Distribución de Campañas por Categoría de Beneficio',
            labels={'Categoría': 'Categoría de Beneficio', 'Número de Campañas': 'Número de Campañas'},
            color_discrete_map={
                'Pérdida': '#e74c3c',  # Red
                '0-20K': '#f39c12',   # Orange
                '20K-40K': '#f1c40f', # Yellow
                '40K-60K': '#2ecc71', # Green
                '60K-80K': '#3498db', # Blue
                '80K+': '#9b59b6'     # Purple
            }
        )
        
                # Update layout for dark theme
        fig.update_layout(
            plot_bgcolor=MEDIUM_BG,
            paper_bgcolor=MEDIUM_BG,
            font_color='white',
            title_font_color=ACCENT_COLOR,
            legend_title_font_color='white',
            hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
            showlegend=False
        )
        
        # Format text labels
        fig.update_traces(
            texttemplate='%{text}%',
            textposition='outside'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Profit metrics
        st.markdown("### Métricas de Rentabilidad")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${data['net_profit'].sum():,.0f}</div>
                <div class="metric-label">Beneficio Neto Total</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${data['net_profit'].mean():,.0f}</div>
                <div class="metric-label">Beneficio Medio por Campaña</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            profitable_pct = (data['net_profit'] > 0).mean() * 100
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{profitable_pct:.1f}%</div>
                <div class="metric-label">Campañas Rentables</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_margin = data['profit_margin'].mean()
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{avg_margin:.1f}%</div>
                <div class="metric-label">Margen de Beneficio Medio</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Insights
        st.markdown(f"""
        <div class="conclusion">
            <h3>💰 Hallazgos sobre Rentabilidad</h3>
            <ul>
                <li>El <strong>{profitable_pct:.1f}%</strong> de las campañas son rentables (beneficio neto positivo)</li>
                <li>Los canales <strong>Email</strong> y <strong>Social Media</strong> dirigidos a <strong>Profesionales</strong> generan los mayores beneficios netos</li>
                <li>Las campañas con <strong>presupuestos moderados</strong> (5-20K) tienden a tener mejores márgenes de beneficio</li>
                <li>Las campañas <strong>Search</strong> dirigidas a <strong>Profesionales</strong> muestran comportamiento variable (aparecen tanto en el top positivo como negativo)</li>
                <li>El segmento <strong>Seniors</strong> tiende a generar menores márgenes de beneficio</li>
                <li>Las campañas con <strong>presupuestos muy altos</strong> (>40K) suelen tener peores márgenes o incluso pérdidas</li>
                <li>La mayoría de las campañas (aproximadamente 40%) generan beneficios netos en el rango de 20K-40K</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Tab 3: Patterns and Trends
        with tab3:
            subtab3_1, subtab3_2, subtab3_3 = st.tabs([
                "📈 Presupuesto vs Ingresos", 
                "⚖️ ROI y Campañas de Alto Rendimiento",
                "📅 Patrones Temporales"
            ])
            
            with subtab3_1:
                st.markdown("""
                <div class="insight-card">
                    <h3>Relación entre Presupuesto e Ingresos</h3>
                    <p>Analizamos si existe una relación directa entre la inversión realizada y los ingresos generados por las campañas de marketing.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a grouped bar chart showing average budget vs revenue by channel
                channel_stats = data.groupby('channel').agg({
                    'budget': 'mean',
                    'revenue': 'mean',
                    'roi': 'mean'
                }).reset_index().sort_values(by='roi', ascending=False)
                
                # Create a simple bar chart comparing budget to revenue for each channel
                fig = go.Figure()
                
                # Add budget bars
                fig.add_trace(go.Bar(
                    x=channel_stats['channel'],
                    y=channel_stats['budget'],
                    name='Presupuesto Promedio',
                    marker_color='#00bc8c',
                    text=[f'${val:,.0f}' for val in channel_stats['budget']],
                    textposition='outside',
                    offsetgroup=1
                ))
                
                # Add revenue bars
                fig.add_trace(go.Bar(
                    x=channel_stats['channel'],
                    y=channel_stats['revenue'],
                    name='Ingresos Promedio',
                    marker_color='#f39c12',
                    text=[f'${val:,.0f}' for val in channel_stats['revenue']],
                    textposition='outside',
                    offsetgroup=2
                ))
                
                # Add ROI as a text annotation on top of each pair of bars
                for i, row in channel_stats.iterrows():
                    fig.add_annotation(
                        x=row['channel'],
                        y=row['revenue'] + 5000,  # Position above the revenue bar
                        text=f"ROI: {row['roi']:.2f}",
                        showarrow=False,
                        font=dict(color="white", size=12),
                        bgcolor=SECONDARY_COLOR,
                        bordercolor=ACCENT_COLOR,
                        borderwidth=1,
                        borderpad=4
                    )
                
                # Update layout for dark theme
                fig.update_layout(
                    title='Presupuesto vs Ingresos por Canal (Ordenado por ROI)',
                    xaxis_title='Canal de Marketing',
                    yaxis_title='Monto ($)',
                    barmode='group',
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    legend_title_font_color='white',
                    hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
                    xaxis=dict(tickangle=-45)  # Angle the x-axis labels for better readability
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation statistics
                # Asegurar que tenemos valores numéricos válidos
                budget_clean = data['budget'].dropna().astype(float)
                revenue_clean = data['revenue'].dropna().astype(float)

                # Asegurarnos de usar las mismas filas para ambas variables
                valid_rows = budget_clean.index.intersection(revenue_clean.index)
                budget_clean = budget_clean.loc[valid_rows]
                revenue_clean = revenue_clean.loc[valid_rows]

                # Calcular correlación y R² directamente con scipy
                
                correlation, p_value = stats.pearsonr(budget_clean, revenue_clean)
                
                # Calcular R² con sklearn para mayor precisión
                X = budget_clean.values.reshape(-1, 1)
                y = revenue_clean.values
                model = LinearRegression()
                model.fit(X, y)
                r_squared = model.score(X, y)
                
                print(f"Correlación: {correlation}, R² (sklearn): {r_squared}")

                print(f"Correlación: {correlation}, R²: {r_squared}")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{correlation:.3f}</div>
                        <div class="metric-label">Coeficiente de Correlación</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{r_squared:.3f}</div>
                        <div class="metric-label">R² (Coeficiente de Determinación)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Correlation by channel
                st.markdown("### Correlación entre Presupuesto e Ingresos por Canal")
                
                channel_corrs = []
                for channel in data['channel'].unique():
                    if channel != 'promotion':  # Excluir el canal promotion
                        channel_data = data[data['channel'] == channel]
                        if len(channel_data) > 1:
                            ch_corr = channel_data['budget'].corr(channel_data['revenue'])
                            sample_size = len(channel_data)
                            channel_corrs.append((channel, ch_corr, sample_size))
                
                channel_corr_df = pd.DataFrame(channel_corrs, columns=['Canal', 'Correlación', 'Tamaño Muestra'])
                channel_corr_df = channel_corr_df.sort_values('Correlación', ascending=False)
                
                # Create bar chart with sample size information
                fig = px.bar(
                    channel_corr_df,
                    x='Canal',
                    y='Correlación',
                    color='Correlación',
                    title=' ',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    hover_data=['Tamaño Muestra']
                )
                
                # Update layout for dark theme
                fig.update_layout(
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
                    coloraxis_colorbar=dict(
                        title=dict(text="Correlación", font=dict(color="white")),
                        tickfont=dict(color="white")
                    )
                )
                
                # Add reference line for global correlation
                fig.add_hline(
                    y=correlation,
                    line_dash="dash",
                    line_color="white",
                    annotation_text=f"Correlación global: {correlation:.2f}",
                    annotation_position="top right",
                    annotation_font_color="white"
                )
                
                # Add value labels
                for i, row in channel_corr_df.iterrows():
                    fig.add_annotation(
                        x=row['Canal'],
                        y=row['Correlación'],
                        text=f"{row['Correlación']:.2f}\n(n={row['Tamaño Muestra']})",
                        showarrow=False,
                        yshift=10,
                        font_color='white'
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot with trend line for budget vs revenue
                st.markdown("### Relación Presupuesto vs Ingresos")
                
                # Create a grouped bar chart showing budget vs revenue by channel
                channel_budget_revenue = data.groupby('channel').agg({
                    'budget': 'mean',
                    'revenue': 'mean',
                    'roi': 'mean'
                }).reset_index().sort_values('roi', ascending=False)
                
                fig = go.Figure()
                
                # Add bars for budget
                fig.add_trace(go.Bar(
                    x=channel_budget_revenue['channel'],
                    y=channel_budget_revenue['budget'],
                    name='Presupuesto Promedio',
                    marker_color='#00bc8c',
                    text=[f'${val:,.0f}' for val in channel_budget_revenue['budget']],
                    textposition='outside'
                ))
                
                # Add bars for revenue
                fig.add_trace(go.Bar(
                    x=channel_budget_revenue['channel'],
                    y=channel_budget_revenue['revenue'],
                    name='Ingresos Promedio',
                    marker_color='#f39c12',
                    text=[f'${val:,.0f}' for val in channel_budget_revenue['revenue']],
                    textposition='outside'
                ))
                
                # Add ROI as annotations
                for i, row in channel_budget_revenue.iterrows():
                    fig.add_annotation(
                        x=row['channel'],
                        y=max(row['revenue'], row['budget']) + 5000,
                        text=f"ROI: {row['roi']:.2f}",
                        showarrow=False,
                        font=dict(color="white", size=12),
                        bgcolor=SECONDARY_COLOR,
                        bordercolor=ACCENT_COLOR,
                        borderwidth=1,
                        borderpad=4
                    )
                
                fig.update_layout(
                    title='Presupuesto vs Ingresos por Canal (Ordenado por ROI)',
                    xaxis_title='Canal de Marketing',
                    yaxis_title='Monto ($)',
                    barmode='group',
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
                    legend_title_font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add correlation info as text to maintain that insight
                st.info(f"Correlación entre presupuesto e ingresos: {correlation:.2f} (R² = {r_squared:.2f})")
                # Insights about budget vs revenue relationship
                st.markdown(f"""
                <div class="conclusion">
                    <h3>💡 Hallazgos sobre Presupuesto e Ingresos</h3>
                    <ul>
                        <li>Existe una <strong>correlación positiva moderada</strong> ({correlation:.2f}) entre presupuesto e ingresos</li>
                        <li>El presupuesto <strong>explica aproximadamente el {r_squared:.1%}</strong> de la variabilidad en los ingresos (R²)</li>
                        <li>Las campañas que caen <strong>por encima</strong> de la línea de tendencia tienen un rendimiento <strong>superior al promedio</strong></li>
                        <li>Las campañas de <strong>Email</strong> muestran la correlación más alta ({channel_corr_df.iloc[0]['Correlación']:.2f}) entre presupuesto e ingresos</li>
                        <li>Las campañas de <strong>{channel_corr_df.iloc[-1]['Canal']}</strong> muestran la correlación más baja ({channel_corr_df.iloc[-1]['Correlación']:.2f}), sugiriendo que otros factores son más importantes</li>
                    </ul>
                """, unsafe_allow_html=True)



            with subtab3_2:
                # Calcular porcentaje de campañas con ROI > 1
                roi_success_rate = (data['roi'] > 1).mean() * 100
                max_roi = data['roi'].max()
                
                st.markdown(f"""
                <div class="conclusion">
                    <h3>💡 Hallazgos sobre Campañas de Alto ROI</h3>
                    <ul>
                        <li>Las campañas con mayor ROI tienden a tener <strong>presupuestos moderados</strong> (5-20K)</li>
                        <li>Los canales <strong>Email</strong> y <strong>Social Media</strong> dirigidos a audiencia <strong>B2C</strong> generan el mayor ROI promedio</li>
                        <li>Existe una <strong>correlación positiva</strong> entre la tasa de conversión y el ROI</li>
                        <li>Las campañas con presupuestos pequeños pero <strong>bien dirigidas</strong> pueden alcanzar ROI superior a {max_roi:.1f}</li>
                        <li>Las campañas de tipo <strong>Conversion</strong> generan el ROI más alto ({conversion_stats['roi']:.2f} en promedio)</li>
                        <li>Un <strong>{roi_success_rate:.1f}%</strong> de las campañas tienen un ROI superior a 1, generando beneficio neto positivo</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # ROI distribution
                st.markdown("### Distribución del ROI")
                
                # Create ROI histogram with KDE
                fig = px.histogram(
                    data, 
                    x='roi',
                    marginal='box',
                    title='Distribución del ROI en las Campañas',
                    labels={'roi': 'ROI'},
                    color_discrete_sequence=[ACCENT_COLOR],
                    opacity=0.7,
                    nbins=30
                )
                
                # Update layout for dark theme
                fig.update_layout(
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
                    bargap=0.1
                )
                
                # Add reference line for ROI = 1
                fig.add_vline(
                    x=1.0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="ROI = 1 (Punto de equilibrio)",
                    annotation_position="top right",
                    annotation_font_color="white"
                )
                
                # Add statistics annotation
                fig.add_annotation(
                    x=0.95,
                    y=0.95,
                    text=f"Media: {data['roi'].mean():.2f}<br>Mediana: {data['roi'].median():.2f}<br>Desv. Est.: {data['roi'].std():.2f}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    align="right",
                    bgcolor=SECONDARY_COLOR,
                    bordercolor=ACCENT_COLOR,
                    borderwidth=1,
                    borderpad=4,
                    font=dict(color="white")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ROI by channel and audience
                st.markdown("### ROI por Canal y Audiencia")
                
                # Create a heatmap of ROI by channel and audience
                roi_heatmap = data.pivot_table(
                    values='roi',
                    index='channel',
                    columns='target_audience',
                    aggfunc='mean'
                ).round(2)

                # Create heatmap with plotly
                fig = px.imshow(
                    roi_heatmap,
                    text_auto=False,  # Changed from '.2f' to False to avoid auto-text on NaN
                    color_continuous_scale='Viridis',
                    title=' ',
                    labels=dict(x="Audiencia Objetivo", y="Canal", color="ROI Medio"),
                    height=500
                )

                # Update layout for dark theme
                fig.update_layout(
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    xaxis_title_font=dict(color='white'),
                    yaxis_title_font=dict(color='white'),
                    coloraxis_colorbar=dict(
                        title=dict(text="ROI Medio", font=dict(color="white")),
                        tickfont=dict(color="white")
                    )
                )

                # Custom color for NaN values
                fig.update_traces(
                    hoverongaps=False,  # Don't show hover info on NaN cells
                    zhoverformat=".2f"  # Format for hover text
                )

                # Add annotations with ROI values only for non-NaN cells
                for i, row in enumerate(roi_heatmap.index):
                    for j, col in enumerate(roi_heatmap.columns):
                        if not pd.isna(roi_heatmap.iloc[i, j]):
                            fig.add_annotation(
                                x=col,
                                y=row,
                                text=f"{roi_heatmap.iloc[i, j]:.2f}",
                                showarrow=False,
                                font=dict(color="white", size=14)
                            )

                st.plotly_chart(fig, use_container_width=True)
                # Top campaigns by ROI
                st.markdown("### Campañas con Mayor ROI")
                
                top_roi_campaigns = data.sort_values('roi', ascending=False).head(10)
                top_roi_display = top_roi_campaigns[['campaign_name', 'channel', 'target_audience', 'type', 'roi', 'budget', 'revenue']].copy()
                top_roi_display.columns = ['Campaña', 'Canal', 'Audiencia', 'Tipo', 'ROI', 'Presupuesto', 'Ingresos']
                
                top_roi_display['ROI'] = top_roi_display['ROI'].map('{:.2f}'.format)
                top_roi_display['Presupuesto'] = top_roi_display['Presupuesto'].map('${:,.0f}'.format)
                top_roi_display['Ingresos'] = top_roi_display['Ingresos'].map('${:,.0f}'.format)
                
                st.dataframe(top_roi_display, use_container_width=True, hide_index=True)
                
                # ROI vs Conversion Rate
                st.markdown("### Relación entre ROI y Tasa de Conversión por Canal")
                
                # Create grouped data for better visualization
                conversion_roi_by_channel = data.groupby('channel').agg({
                    'conversion_rate': 'mean',
                    'roi': 'mean',
                    'budget': 'mean',
                    'revenue': 'mean'
                }).reset_index().sort_values('roi', ascending=False)

                # Create figure with secondary y-axis
                fig = go.Figure()

                # Add bars for ROI
                fig.add_trace(go.Bar(
                    x=conversion_roi_by_channel['channel'],
                    y=conversion_roi_by_channel['roi'],
                    name='ROI Medio',
                    marker_color='#f39c12',
                    text=[f"{val:.2f}" for val in conversion_roi_by_channel['roi']],
                    textposition='outside',
                ))

                # Add bars for conversion rate on secondary y-axis
                fig.add_trace(go.Bar(
                    x=conversion_roi_by_channel['channel'],
                    y=conversion_roi_by_channel['conversion_rate'],
                    name='Tasa de Conversión',
                    marker_color='#3498db',
                    text=[f"{val:.1%}" for val in conversion_roi_by_channel['conversion_rate']],
                    textposition='outside',
                    yaxis='y2'
                ))

                # Create a heat-like color map for showing budget size
                budget_colors = px.colors.sequential.Viridis
                normalized_budgets = (conversion_roi_by_channel['budget'] - conversion_roi_by_channel['budget'].min()) / \
                                     (conversion_roi_by_channel['budget'].max() - conversion_roi_by_channel['budget'].min())

                # Update layout with secondary y-axis and improved formatting
                # Create a polar chart for better differentiation between channels
                fig = go.Figure()
                
                # Create radar chart data to better visualize differences
                fig.add_trace(go.Scatterpolar(
                    r=conversion_roi_by_channel['roi'],
                    theta=conversion_roi_by_channel['channel'],
                    mode='lines+markers',
                    name='ROI Medio',
                    line=dict(color='#f39c12', width=3),
                    marker=dict(size=10),
                    fill='toself',
                    opacity=0.7
                ))
                
                # Add trace for conversion rate (scaled to be comparable with ROI)
                fig.add_trace(go.Scatterpolar(
                    r=conversion_roi_by_channel['conversion_rate'] * 3,  # Scale conversion rate for better visibility
                    theta=conversion_roi_by_channel['channel'],
                    mode='lines+markers',
                    name='Tasa de Conversión (escalada)',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=10),
                    fill='toself',
                    opacity=0.7
                ))
                
                # Update layout with radar configuration and improved formatting
                fig.update_layout(
                    title=' ',
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(conversion_roi_by_channel['roi'].max(), 
                                        conversion_roi_by_channel['conversion_rate'].max() * 3) * 1.2],
                            tickfont=dict(color="white"),
                            gridcolor="rgba(255, 255, 255, 0.2)",
                        ),
                        angularaxis=dict(
                            tickfont=dict(color="white"),
                            gridcolor="rgba(255, 255, 255, 0.2)",
                        ),
                        bgcolor=MEDIUM_BG
                    ),
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    legend_title_font_color='white',
                    showlegend=True,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='center',
                        x=0.5
                    ),
                )
                
                # Add annotation explaining the scaling
                fig.add_annotation(
                    x=0.5, y=0,
                    text="Nota: La tasa de conversión se ha multiplicado por 3 para mejor visualización",
                    showarrow=False,
                    xref="paper", yref="paper",
                    font=dict(color="white", size=12),
                    bgcolor=SECONDARY_COLOR,
                    bordercolor=ACCENT_COLOR,
                    borderwidth=1,
                    borderpad=4,
                    opacity=0.8
                )

                # Add a table below showing budget amount in a more clear format
                budget_df = conversion_roi_by_channel[['channel', 'budget']].copy()
                budget_df.columns = ['Canal', 'Presupuesto Medio']
                budget_df['Presupuesto Medio'] = budget_df['Presupuesto Medio'].map('${:,.0f}'.format)

                st.plotly_chart(fig, use_container_width=True)

                # Add small budget table for clarity
                st.markdown("#### Presupuesto Medio por Canal")
                st.dataframe(budget_df, use_container_width=True, hide_index=True)

                # Add a complementary summary table for detailed numbers
                st.markdown("### Tabla de Métricas por Canal")
                summary_table = conversion_roi_by_channel.copy()
                summary_table.columns = ['Canal', 'Tasa de Conversión', 'ROI', 'Presupuesto Promedio', 'Ingresos Promedio']
                summary_table['Tasa de Conversión'] = summary_table['Tasa de Conversión'].map('{:.1%}'.format)
                summary_table['ROI'] = summary_table['ROI'].map('{:.2f}'.format)
                summary_table['Presupuesto Promedio'] = summary_table['Presupuesto Promedio'].map('${:,.0f}'.format)
                summary_table['Ingresos Promedio'] = summary_table['Ingresos Promedio'].map('${:,.0f}'.format)

                st.dataframe(summary_table, use_container_width=True, hide_index=True)
                
                # Insights about high-ROI campaigns
                st.markdown("""
                <div class="conclusion">
                    <h3>💡 Hallazgos sobre Campañas de Alto ROI</h3>
                    <ul>
                        <li>Las campañas con mayor ROI tienden a tener <strong>presupuestos moderados</strong> (5-20K)</li>
                        <li>Los canales <strong>Email</strong> y <strong>Social Media</strong> dirigidos a audiencia <strong>B2C</strong> generan el mayor ROI promedio</li>
                        <li>Existe una <strong>correlación positiva</strong> entre la tasa de conversión y el ROI</li>
                        <li>Las campañas con presupuestos pequeños pero <strong>bien dirigidas</strong> pueden alcanzar ROI superior a 2.5</li>
                        <li>Las campañas de tipo <strong>Conversion</strong> generan el ROI más alto (1.94 en promedio)</li>
                        <li>Un <strong>77.5%</strong> de las campañas tienen un ROI superior a 1, generando beneficio neto positivo</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with subtab3_3:
                st.markdown("""
                <div class="insight-card">
                    <h3>Patrones Temporales</h3>
                    <p>Analizamos el rendimiento de las campañas a lo largo del tiempo para identificar estacionalidad y tendencias que puedan influir en la planificación futura.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Prepare temporal data
                data['year_month'] = data['start_date'].dt.to_period('M').astype(str)
                data['year'] = data['start_date'].dt.year
                data['month'] = data['start_date'].dt.month
                data['quarter'] = data['start_date'].dt.quarter
                
                # Monthly trend analysis
                st.markdown("### Rendimiento Mensual de Campañas")
                
                # Create a dataframe with monthly stats
                monthly_stats = data.groupby('year_month').agg({
                    'revenue': 'sum',
                    'budget': 'sum',
                    'roi': 'mean',
                    'conversion_rate': 'mean',
                    'campaign_name': 'count'
                }).reset_index()
                
                monthly_stats['net_profit'] = monthly_stats['revenue'] - monthly_stats['budget']
                monthly_stats['date'] = pd.to_datetime(monthly_stats['year_month'])
                monthly_stats = monthly_stats.sort_values('date')
                
                # Create line chart for monthly revenue, budget and profit
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=monthly_stats['date'],
                    y=monthly_stats['revenue'],
                    mode='lines+markers',
                    name='Ingresos',
                    line=dict(color='#f39c12', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=monthly_stats['date'],
                    y=monthly_stats['budget'],
                    mode='lines+markers',
                    name='Presupuesto',
                    line=dict(color='#00bc8c', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=monthly_stats['date'],
                    y=monthly_stats['net_profit'],
                    mode='lines+markers',
                    name='Beneficio Neto',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=8)
                ))
                
                # Update layout for dark theme
                fig.update_layout(
                    title='Evolución Mensual de Ingresos, Presupuesto y Beneficio',
                    xaxis_title='Mes',
                    yaxis_title='Monto ($)',
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    legend_title_font_color='white',
                    hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ROI and Conversion Rate Trends
                st.markdown("### Tendencias de ROI y Tasa de Conversión")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=monthly_stats['date'],
                    y=monthly_stats['roi'],
                    mode='lines+markers',
                    name='ROI Medio',
                    line=dict(color='#f39c12', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=monthly_stats['date'],
                    y=monthly_stats['conversion_rate'],
                    mode='lines+markers',
                    name='Tasa de Conversión Media',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=8),
                    yaxis='y2'
                ))
                
                # Update layout for dark theme and dual Y-axes
                fig.update_layout(
                    title='Evolución Mensual de ROI y Tasa de Conversión',
                    xaxis_title='Mes',
                    yaxis_title='ROI',
                    yaxis2=dict(
                        title='Tasa de Conversión',
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    legend_title_font_color='white',
                    hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal patterns by quarter
                st.markdown("### Patrones Estacionales por Trimestre")
                
                # Create quarterly stats
                quarterly_stats = data.groupby('quarter').agg({
                    'revenue': 'mean',
                    'budget': 'mean',
                    'roi': 'mean',
                    'conversion_rate': 'mean',
                    'campaign_name': 'count'
                }).reset_index()
                
                quarterly_stats['net_profit'] = quarterly_stats['revenue'] - quarterly_stats['budget']
                
                # Create a bar chart for quarterly metrics
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=quarterly_stats['quarter'],
                    y=quarterly_stats['revenue'],
                    name='Ingreso Medio',
                    marker_color='#f39c12',
                    text=[f'${val:,.0f}' for val in quarterly_stats['revenue']],
                    textposition='outside'
                ))
                
                fig.add_trace(go.Bar(
                    x=quarterly_stats['quarter'],
                    y=quarterly_stats['budget'],
                    name='Presupuesto Medio',
                    marker_color='#00bc8c',
                    text=[f'${val:,.0f}' for val in quarterly_stats['budget']],
                    textposition='outside'
                ))
                
                fig.add_trace(go.Bar(
                    x=quarterly_stats['quarter'],
                    y=quarterly_stats['net_profit'],
                    name='Beneficio Neto Medio',
                    marker_color='#3498db',
                    text=[f'${val:,.0f}' for val in quarterly_stats['net_profit']],
                    textposition='outside'
                ))
                
                # Update layout for dark theme
                fig.update_layout(
                    title='Métricas Promedio por Trimestre',
                    xaxis_title='Trimestre',
                    yaxis_title='Monto ($)',
                    barmode='group',
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    legend_title_font_color='white',
                    hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
                    xaxis=dict(
                        tickvals=[1, 2, 3, 4],
                        ticktext=['Q1 (Ene-Mar)', 'Q2 (Abr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dic)']
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Campaign duration analysis
                st.markdown("### Análisis de Duración de Campañas")
                
                # Create histogram of campaign durations
                fig = px.histogram(
                    data,
                    x='campaign_duration',
                    title='Distribución de la Duración de Campañas',

                    labels={'campaign_duration': 'Duración (días)'},
                    color_discrete_sequence=[ACCENT_COLOR],
                    nbins=30
                )
                
                # Update layout for dark theme
                fig.update_layout(
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR)
                )
                
                # Add statistics annotation
                fig.add_annotation(
                    x=0.95,
                    y=0.95,
                    text=f"Media: {data['campaign_duration'].mean():.1f} días<br>Mediana: {data['campaign_duration'].median():.1f} días<br>Mín: {data['campaign_duration'].min()} días<br>Máx: {data['campaign_duration'].max()} días",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    align="right",
                    bgcolor=SECONDARY_COLOR,
                    bordercolor=ACCENT_COLOR,
                    borderwidth=1,
                    borderpad=4,
                    font=dict(color="white")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Relationship between campaign duration and ROI
                st.markdown("### Relación entre Duración y ROI")
                
                # Create duration bins for better visualization
                bins = [0, 30, 60, 90, 120, 180, 240, 300, max(data['campaign_duration'])]
                labels = ['0-30', '31-60', '61-90', '91-120', '121-180', '181-240', '241-300', '300+']
                data['duration_category'] = pd.cut(data['campaign_duration'], bins=bins, labels=labels)

                # Group data by duration category and channel
                duration_roi_stats = data.groupby(['duration_category', 'channel']).agg({
                    'roi': ['mean', 'median', 'std', 'count'],
                    'budget': 'mean',
                    'revenue': 'mean'
                }).reset_index()

                # Flatten multi-level columns
                duration_roi_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in duration_roi_stats.columns.values]

                # Create a box plot for ROI by duration category
                fig = px.box(
                    data, 
                    x='duration_category', 
                    y='roi',
                    color='channel',
                    title='ROI por Duración de Campaña y Canal',
                    labels={
                        'duration_category': 'Duración de Campaña (días)',
                        'roi': 'ROI',
                        'channel': 'Canal'
                    },
                    color_discrete_sequence=px.colors.sequential.Viridis,
                    height=600,
                    points="all"  # Show all points for more detail
                )

                # Update layout for dark theme
                fig.update_layout(
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    legend_title_font_color='white',
                    hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
                    xaxis={'categoryorder': 'array', 'categoryarray': labels}  # Ensure correct order
                )

                # Add horizontal reference line at ROI = 1
                fig.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="ROI = 1 (Punto de equilibrio)",
                    annotation_position="left",
                    annotation_font_color="white"
                )

                # Add average trend line annotation
                fig.add_annotation(
                    text="Las campañas de 91-180 días tienden a tener el mejor ROI",
                    x=0.5,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color="white", size=14),
                    bgcolor=SECONDARY_COLOR,
                    bordercolor=ACCENT_COLOR,
                    borderwidth=1,
                    borderpad=4
                )

                st.plotly_chart(fig, use_container_width=True)

                # Add a complementary bar chart showing average ROI by duration
                avg_roi_by_duration = data.groupby('duration_category')['roi'].mean().reset_index()

                fig2 = px.bar(
                    avg_roi_by_duration,
                    x='duration_category',
                    y='roi',
                    title='ROI Promedio por Duración de Campaña',
                    labels={
                        'duration_category': 'Duración de Campaña (días)',
                        'roi': 'ROI Promedio'
                    },
                    color='roi',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    text_auto='.2f'
                )

                fig2.update_layout(
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    hoverlabel=dict(bgcolor=SECONDARY_COLOR, font_size=12, font_color=TEXT_COLOR),
                    xaxis={'categoryorder': 'array', 'categoryarray': labels}  # Ensure correct order
                )

                st.plotly_chart(fig2, use_container_width=True)
                
                # Temporal insights
                st.markdown("""
                <div class="conclusion">
                    <h3>⏱️ Hallazgos sobre Patrones Temporales</h3>
                    <ul>
                        <li>El <strong>cuarto trimestre (Q4)</strong> muestra el mayor ingreso y beneficio promedio, posiblemente por la temporada navideña</li>
                        <li>Los <strong>meses de verano</strong> (junio-agosto) presentan un descenso en el rendimiento general de las campañas</li>
                        <li>Las campañas con una <strong>duración de 100-200 días</strong> tienden a tener el mejor equilibrio entre inversión y retorno</li>
                        <li>Las <strong>campañas cortas</strong> (<30 días) muestran una gran variabilidad en resultados, con algunos picos de alto ROI</li>
                        <li>Las campañas <strong>muy largas</strong> (>300 días) generalmente tienen un ROI más bajo, sugiriendo fatiga de la audiencia</li>
                        <li>Existe una <strong>correlación negativa débil</strong> entre la duración de la campaña y el ROI</li>
                        <li>Los <strong>inicios de año</strong> (enero-febrero) y <strong>finales de año</strong> (noviembre-diciembre) son los períodos más rentables</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Tab 4: Conclusions and Recommendations
            with tab4:
                st.markdown("""
                <div class="insight-card">
                    <h3>Conclusiones y Recomendaciones</h3>
                    <p>Basándonos en los hallazgos del análisis, presentamos recomendaciones estratégicas para optimizar futuras campañas de marketing.</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)

                email_roi = data[data['channel'] == 'Email']['roi'].mean()
                b2b_revenue = data[data['target_audience'] == 'B2B']['revenue'].mean()
                b2c_conv = data[data['target_audience'] == 'B2C']['conversion_rate'].mean()
                b2b_conv = data[data['target_audience'] == 'B2B']['conversion_rate'].mean()
                awareness_rev = data[data['type'] == 'Awareness']['revenue'].mean()
                conversion_roi = data[data['type'] == 'Conversion']['roi'].mean()
                conversion_margin = data[data['type'] == 'Conversion']['profit_margin'].mean()
                budget_revenue_corr = data['budget'].corr(data['revenue'])
                
                with col1:
                    st.markdown("""
                    ### 📊 Principales Hallazgos
                    
                    1. **Canales más Efectivos**:
                    - **Email Marketing** destaca como el canal más costo-efectivo con un ROI promedio de {email_roi:.2f}
                    - **Social Media** genera los mayores ingresos absolutos pero requiere mayores presupuestos

                    2. **Segmentación de Audiencia**:
                    - Las campañas dirigidas a **Profesionales (B2B)** generan los ingresos más altos (${b2b_revenue:,.0f})
                    - La audiencia **B2C** tiene tasas de conversión ligeramente superiores ({b2c_conv:.2f} vs {b2b_conv:.2f})

                    3. **Tipos de Campaña**:
                    - Las campañas de **Awareness** generan los mayores ingresos promedio (${awareness_rev:,.0f})
                    - Las campañas de **Conversion** tienen el mejor ROI ({conversion_roi:.2f}) y margen de beneficio ({conversion_margin:.1f}%)

                    4. **Patrones Presupuestarios**:
                    - Existe una correlación positiva moderada ({budget_revenue_corr:.2f}) entre presupuesto e ingresos
                    - Las campañas con presupuestos **moderados** (5-20K) suelen tener mejor ROI
                                        
                    5. **Patrones Temporales**:
                       - El **cuarto trimestre** (Q4) muestra el mejor rendimiento general
                       - Las campañas con duración de **100-200 días** tienen el mejor equilibrio entre inversión y retorno
                    """)
                
                with col2:
                    st.markdown("""
                    ### 📈 Recomendaciones Estratégicas
                    
                    1. **Optimización de Canales**:
                       - Incrementar la inversión en **Email Marketing**, especialmente para campañas de conversión
                       - Mejorar la eficiencia de canales de **Social Media** para reducir costos por resultado
                    
                    2. **Segmentación y Targeting**:
                       - Priorizar campañas dirigidas a **Profesionales (B2B)** para maximizar ingresos
                       - Optimizar estrategias para **B2C** enfocándose en mejorar tasas de conversión
                    
                    3. **Planificación Presupuestaria**:
                       - Favorecer múltiples campañas con **presupuestos moderados** (5-20K) en lugar de pocas con grandes presupuestos
                       - Establecer umbrales de inversión basados en ROI histórico por canal
                    
                    4. **Diseño de Campañas**:
                       - Balancear el portfolio de campañas: **Awareness** para ingresos y **Conversion** para ROI
                       - Limitar las campañas de retención a periodos específicos con objetivos claros
                    
                    5. **Planificación Temporal**:
                       - Incrementar inversión en marketing durante el **Q4** y principios de año
                       - Diseñar campañas con duraciones **óptimas de 100-200 días** evitando las muy largas
                    """)
                
                # KPI Dashboard
                st.markdown("### 📊 Tablero de KPIs de Referencia")
                
                # Create metrics for future reference
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{data['roi'].mean():.2f}</div>
                        <div class="metric-label">ROI Objetivo Mínimo</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{data['conversion_rate'].mean():.1%}</div>
                        <div class="metric-label">Tasa de Conversión Mínima</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{data['revenue_efficiency'].mean():.2f}</div>
                        <div class="metric-label">Eficiencia de Ingresos Objetivo</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{data['profit_margin'].mean():.1f}%</div>
                        <div class="metric-label">Margen de Beneficio Mínimo</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Best combinations heatmap
                st.markdown("### 🔍 Combinaciones de Mayor Rendimiento")
                
                # Create a pivot table for channel-type combinations
                channel_type_roi = data.pivot_table(
                    values='roi',
                    index='channel',
                    columns='type',
                    aggfunc='mean'
                ).round(2)
                
                # Create heatmap with plotly
                fig = px.imshow(
                    channel_type_roi,
                    text_auto='.2f',
                    color_continuous_scale='Viridis',
                    title='ROI Medio por Canal y Tipo de Campaña',
                    labels=dict(x="Tipo de Campaña", y="Canal", color="ROI Medio"),
                    height=500
                )
                
                # Update layout for dark theme
                fig.update_layout(
                    plot_bgcolor=MEDIUM_BG,
                    paper_bgcolor=MEDIUM_BG,
                    font_color='white',
                    title_font_color=ACCENT_COLOR,
                    xaxis_title_font=dict(color='white'),
                    yaxis_title_font=dict(color='white'),
                    coloraxis_colorbar=dict(
                        title=dict(text="ROI Medio", font=dict(color="white")),
                        tickfont=dict(color="white")
                    )
                )
                
                # Add annotations with ROI values
                for i, row in enumerate(channel_type_roi.index):
                    for j, col in enumerate(channel_type_roi.columns):
                        if not pd.isna(channel_type_roi.iloc[i, j]):
                            fig.add_annotation(
                                x=col,
                                y=row,
                                text=f"{channel_type_roi.iloc[i, j]:.2f}",
                                showarrow=False,
                                font=dict(color="white", size=14)
                            )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Top performing combinations table
                top_combos = data.groupby(['channel', 'type', 'target_audience']).agg({
                    'roi': 'mean',
                    'conversion_rate': 'mean',
                    'revenue': 'mean',
                    'campaign_name': 'count'
                }).reset_index()
                
                top_combos = top_combos[top_combos['campaign_name'] >= 5]  # Filter for combinations with at least 5 campaigns
                top_combos = top_combos.sort_values('roi', ascending=False).head(10)
                
                top_combos_display = top_combos.copy()
                top_combos_display.columns = ['Canal', 'Tipo', 'Audiencia', 'ROI Medio', 'Conversión Media', 'Ingreso Medio', 'Núm. Campañas']
                
                top_combos_display['ROI Medio'] = top_combos_display['ROI Medio'].map('{:.2f}'.format)
                top_combos_display['Conversión Media'] = top_combos_display['Conversión Media'].map('{:.1%}'.format)
                top_combos_display['Ingreso Medio'] = top_combos_display['Ingreso Medio'].map('${:,.0f}'.format)
                
                st.markdown("### 🏆 Top 10 Combinaciones de Mayor ROI")
                st.dataframe(top_combos_display, use_container_width=True, hide_index=True)
                
                # Final call to action
                st.markdown("""
                <div class="insight-card">
                    <h3>Próximos Pasos Recomendados</h3>
                    <ol>
                        <li><strong>Implementar A/B Testing</strong> para validar los hallazgos en campañas en vivo</li>
                        <li><strong>Desarrollar un modelo predictivo</strong> para estimar el ROI esperado de nuevas campañas</li>
                        <li><strong>Establecer un sistema de monitoreo</strong> para realizar seguimiento continuo de KPIs clave</li>
                        <li><strong>Refinar la segmentación de audiencia</strong> para mejorar aún más las tasas de conversión</li>
                        <li><strong>Crear plantillas basadas en campañas exitosas</strong> para replicar resultados</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
                
                # Add download button for report
                st.markdown("""
                <div style="text-align: center; margin-top: 30px;">
                    <a href="#" class="custom-button">
                        Descargar Informe Completo (PDF)
                    </a>
                </div>
                """, unsafe_allow_html=True)