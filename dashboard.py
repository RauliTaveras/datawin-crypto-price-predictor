###
# Pagina de DataWin con: | Dashboard | Chatbot IA | Prediccion en vivo | Graficos de velas | Metricass del modelo | Tabla de datos 
###

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import requests
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Generator

from config       import PICKLE_PROCESSED, CRYPTOS, INTERVALS, MODEL_RIDGE, FEATURE_COLS_PATH
from predictor    import predict_next_price
from visualizer   import plot_crypto_candlestick
from utils.logger import logger

# Configuacion de la página

st.set_page_config(
    page_title="DataWin | Crypto Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inicializar estado para el chatbot y verificar conexión con Ollama

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ollama_available" not in st.session_state:
    st.session_state.ollama_available = False
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        st.session_state.ollama_available = response.status_code == 200
    except:
        st.session_state.ollama_available = False

# Variables para pasar datos al chatbot
if "current_metrics" not in st.session_state:
    st.session_state.current_metrics = {}

if "show_chat" not in st.session_state:
    st.session_state.show_chat = True 

# Funciones de nuestro asistente o  (CHATBOT)

def chat_with_phi4(message: str, conversation_history: list = None, page_context: str = "") -> Generator:
    """Envía mensaje a phi4-mini con contexto de la página"""
    
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "phi4-mini"
    
    # Construir el contexto con datos de la página
    if conversation_history:
        context = "\n".join([
            f"{'Usuario' if msg['role'] == 'user' else 'Asistente'}: {msg['content']}"
            for msg in conversation_history[-4:]  # Últimos 4 mensajes
        ])
        full_prompt = f"{page_context}\n\n{context}\nUsuario: {message}\nAsistente:"
    else:
        full_prompt = f"{page_context}\n\nUsuario: {message}\nAsistente:"
    
    system_prompt = """Eres un experto en análisis técnico de criptomonedas y machine learning.
Tienes acceso a las métricas y datos del dashboard en tiempo real.
Tu rol es:
1. Interpretar las MÉTRICAS ESPECÍFICAS del usuario (MAE, RMSE, R², RSI, etc.)
2. Analizar el GRÁFICO y la predicción actual
3. Dar insights basados en los DATOS QUE VES en la página
4. Ser conciso y directo
5. Cuando el usuario pregunte por "resultados", refierete a lo que VES en la página

Sé específico y referencia los números que ves. Responde en español."""
    
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "stream": True,
            "system": system_prompt,
            "temperature": 0.3,  # Temperatura baja para respuestas más consistentes, ya que con 0.7 estaba habalando mucho con ideas no relacionadas a los datos
            "top_p": 0.8,
            "top_k": 30,
            "num_predict": 400
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=60)
        
        if response.status_code != 200:
            yield f"Error: {response.status_code}"
            return
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                except json.JSONDecodeError:
                    continue
    
    except requests.exceptions.ConnectionError:
        yield "❌ Ollama no está conectado. Ejecuta: `ollama serve`"
    except requests.exceptions.Timeout:
        yield "⏱️ Ollama tardó demasiado."
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def render_chat_interface(page_context: str = ""):
    """Renderiza la interfaz del chatbot"""
    
    # Header del chat con status de conexión
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.markdown('<div style="width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#0af0ff,#00aaff);display:flex;align-items:center;justify-content:center;font-weight:bold;color:#020b18;font-family:Orbitron,monospace;font-size:14px;">🤖</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="font-family:Orbitron,monospace;font-size:0.9rem;color:#0af0ff;letter-spacing:0.1em;">EXPERTO IA</div>', unsafe_allow_html=True)
    with col3:
        status = "● Online" if st.session_state.ollama_available else "● Offline"
        status_color = "#0af0ff" if st.session_state.ollama_available else "#ef5350"
        st.markdown(f'<div style="font-size:0.75rem;text-align:center;padding:5px;border-radius:5px;color:{status_color};border:1px solid {status_color}33;">{status}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Área de los mensajes
    messages_placeholder = st.container()

    # Input y botón de enviar
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_area(
            "Tu pregunta:",
            key="chat_input",
            height=60,
            placeholder="Pregunta sobre MAE, RMSE, R², gráfico...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤", key="send_btn", use_container_width=True)

    # Procesar los mensaje
    if send_button and user_input.strip():
        if not st.session_state.ollama_available:
            st.error("❌ Ollama no está conectado. Ejecuta: `ollama serve`")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with messages_placeholder.container():
                for msg in st.session_state.chat_history:
                    bg = "linear-gradient(90deg,#062d56,#041e3a)" if msg["role"] == "user" else "linear-gradient(90deg,#041526,#061e38)"
                    border = "#0af0ff" if msg["role"] == "user" else "#00aaff"
                    color = "#0af0ff" if msg["role"] == "user" else "#4fc3f7"
                    prefix = "Tú:" if msg["role"] == "user" else "🤖 IA:"
                    
                    st.markdown(f"""
                    <div style="margin-bottom:10px;padding:10px;border-radius:8px;font-family:Exo 2,sans-serif;font-size:0.85rem;background:{bg};border-left:3px solid {border};color:{color};word-wrap:break-word;">
                        <strong>{prefix}</strong> {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("**🤖 IA:** ")
            response_placeholder = st.empty()
            full_response = ""

            with st.spinner("Pensando..."):
                for chunk in chat_with_phi4(user_input, st.session_state.chat_history, page_context):
                    full_response += chunk
                    response_placeholder.markdown(full_response)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    # Mostrar historial
    with messages_placeholder.container():
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                bg = "linear-gradient(90deg,#062d56,#041e3a)" if msg["role"] == "user" else "linear-gradient(90deg,#041526,#061e38)"
                border = "#0af0ff" if msg["role"] == "user" else "#00aaff"
                color = "#0af0ff" if msg["role"] == "user" else "#4fc3f7"
                prefix = "Tú:" if msg["role"] == "user" else "🤖 IA:"
                
                st.markdown(f"""
                <div style="margin-bottom:10px;padding:10px;border-radius:8px;font-family:Exo 2,sans-serif;font-size:0.85rem;background:{bg};border-left:3px solid {border};color:{color};word-wrap:break-word;">
                    <strong>{prefix}</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;color:#4fc3f7;padding:20px;">👋 Pregunta sobre los datos que ves aquí</div>', unsafe_allow_html=True)

# Estilos en CSS

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');

    .stApp {
        background: linear-gradient(135deg, #020b18 0%, #051428 50%, #020d1f 100%);
        font-family: 'Exo 2', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #030f1e 0%, #041526 100%);
        border-right: 1px solid #0af0ff22;
    }
    [data-testid="stSidebar"] * { color: #a0d8ef !important; font-family: 'Exo 2', sans-serif !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label { color: #0af0ff !important; font-weight: 600; letter-spacing: 0.05em; }

    .datawin-header {
        display: flex; align-items: center; gap: 20px;
        padding: 18px 24px;
        background: linear-gradient(90deg, #020e1e, #041e3a, #020e1e);
        border: 1px solid #0af0ff33; border-radius: 12px; margin-bottom: 24px;
        box-shadow: 0 0 30px #0af0ff18, inset 0 0 40px #0a1a2f;
    }
    .datawin-logo-img { height: 70px; filter: drop-shadow(0 0 12px #0af0ffaa); }
    .datawin-team {
        font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 900;
        background: linear-gradient(90deg, #0af0ff, #00aaff, #0af0ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 0.08em; line-height: 1;
    }
    .datawin-subtitle { font-family: 'Exo 2', sans-serif; font-size: 0.85rem; color: #4fc3f7; letter-spacing: 0.2em; text-transform: uppercase; margin-top: 4px; }
    .datawin-badge { margin-left: auto; font-family: 'Orbitron', monospace; font-size: 0.7rem; color: #0af0ff; border: 1px solid #0af0ff44; padding: 4px 10px; border-radius: 20px; letter-spacing: 0.1em; background: #0af0ff0a; }

    [data-testid="stMetric"] { 
        background: linear-gradient(135deg, #041526, #061e38); 
        border: 1px solid #0af0ff22; 
        border-radius: 10px; 
        padding: 16px 20px; 
        box-shadow: 0 0 15px #0af0ff0a;
    }
    [data-testid="stMetricLabel"] { 
        color: #4fc3f7 !important; 
        font-family: 'Exo 2', sans-serif !important;
        font-size: 0.78rem !important; 
        letter-spacing: 0.1em; 
        text-transform: uppercase; 
    }
    [data-testid="stMetricValue"] { 
        color: #0af0ff !important; 
        font-family: 'Orbitron', monospace !important; 
        font-size: 1.4rem !important; 
        font-weight: 700; 
    }

    h2, h3 { 
        font-family: 'Orbitron', monospace !important; 
        color: #0af0ff !important; 
        letter-spacing: 0.05em; 
        font-size: 1rem !important; 
        border-bottom: 1px solid #0af0ff22; 
        padding-bottom: 6px; 
    }

    .stButton > button { 
        background: linear-gradient(135deg, #041e3a, #062d56); 
        color: #0af0ff !important; 
        border: 1px solid #0af0ff55; 
        border-radius: 8px; 
        font-family: 'Exo 2', sans-serif; 
        font-weight: 600; 
        letter-spacing: 0.05em; 
        transition: all 0.2s ease; 
    }
    .stButton > button:hover { 
        background: linear-gradient(135deg, #062d56, #0a3d6b); 
        border-color: #0af0ff; 
        box-shadow: 0 0 15px #0af0ff44; 
    }

    [data-testid="stExpander"] { 
        background: #041526; 
        border: 1px solid #0af0ff22; 
        border-radius: 10px; 
    }
    [data-testid="stExpander"] summary { 
        color: #4fc3f7 !important; 
        font-family: 'Exo 2', sans-serif; 
        font-weight: 600; 
    }

    hr { border-color: #0af0ff22 !important; }

    .section-title { 
        font-family: 'Orbitron', monospace; 
        font-size: 0.8rem; 
        color: #0af0ff; 
        letter-spacing: 0.2em; 
        text-transform: uppercase; 
        margin: 20px 0 10px 0; 
        border-left: 3px solid #0af0ff; 
        padding-left: 10px; 
    }

    .sidebar-logo { 
        text-align: center; 
        padding: 10px 0 20px 0; 
    }
    .sidebar-logo img { 
        width: 130px; 
        filter: drop-shadow(0 0 10px #0af0ffaa); 
    }
    .sidebar-team-name { 
        font-family: 'Orbitron', monospace; 
        font-size: 1.1rem; 
        font-weight: 900; 
        color: #0af0ff; 
        letter-spacing: 0.1em; 
        margin-top: 8px; 
    }
    .sidebar-members { 
        font-family: 'Exo 2', sans-serif; 
        font-size: 0.75rem; 
        color: #4fc3f7; 
        letter-spacing: 0.05em; 
        margin-top: 4px; 
    }

    [data-testid="stPlotlyContainer"] {
        background: linear-gradient(135deg, #041526, #061e38) !important;
        border: 1px solid #0af0ff22 !important;
        border-radius: 10px !important;
        padding: 16px !important;
        box-shadow: 0 0 15px #0af0ff0a !important;
    }

    /* Chat fijo sin scroll infinito */
    .chat-fixed-container {
        position: fixed;
        right: 16px;
        top: 120px;
        width: 340px;
        height: 680px;
        background: linear-gradient(135deg, #041526, #061e38);
        border: 1px solid #0af0ff22;
        border-radius: 10px;
        padding: 12px;
        display: flex;
        flex-direction: column;
        z-index: 1000;
        overflow: hidden;
    }

    .chat-header-fixed {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 0;
        border-bottom: 1px solid #0af0ff22;
        margin-bottom: 10px;
        flex-shrink: 0;
    }

    .chat-messages-fixed {
        flex: 1;
        overflow-y: auto;
        margin-bottom: 10px;
        padding-right: 5px;
    }

    .chat-input-fixed {
        border-top: 1px solid #0af0ff22;
        padding-top: 10px;
        flex-shrink: 0;
    }

    .chat-messages-fixed::-webkit-scrollbar {
        width: 6px;
    }
    .chat-messages-fixed::-webkit-scrollbar-track {
        background: #020b18;
    }
    .chat-messages-fixed::-webkit-scrollbar-thumb {
        background: #0af0ff33;
        border-radius: 3px;
    }
    .chat-messages-fixed::-webkit-scrollbar-thumb:hover {
        background: #0af0ff55;
    }

    /* Toggle button */
    .chat-toggle-btn {
        position: fixed;
        right: 16px;
        bottom: 30px;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0af0ff, #00aaff);
        border: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        z-index: 999;
        box-shadow: 0 0 20px #0af0ff44;
    }

    .chat-toggle-btn:hover {
        box-shadow: 0 0 30px #0af0ff66;
    }
</style>
""", unsafe_allow_html=True)

# Funciones de carga

@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    if not os.path.exists(PICKLE_PROCESSED):
        st.error("No hay datos procesados. Ejecuta primero: python pipeline_etl.py")
        st.stop()
    return pd.read_pickle(PICKLE_PROCESSED)

def get_latest_price(df: pd.DataFrame, symbol: str) -> float:
    subset = df[df["symbol"] == symbol]
    if subset.empty:
        return 0.0
    return float(subset.sort_values("timestamp")["close"].iloc[-1])

# Sidebar izquierdo para el chatbot configuracion

with st.sidebar:
    logo_path = "assets/datawin_logo.png"
    if os.path.exists(logo_path):
        import base64
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'''
        <div class="sidebar-logo">
            <img src="data:image/png;base64,{logo_b64}"/>
            <div class="sidebar-members">Rauli &amp; Stevens</div>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="sidebar-logo">
            <div class="sidebar-team-name">DataWin</div>
            <div class="sidebar-members">Rauli &amp; Stevens</div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Configuracion</div>', unsafe_allow_html=True)
    symbol     = st.selectbox("Criptomoneda", CRYPTOS, index=0)
    interval   = st.selectbox("Intervalo",    INTERVALS, index=0)
    model_type = st.radio("Modelo de prediccion", ["ridge", "sarimax"], index=0)
    show_rsi   = st.checkbox("Mostrar RSI (14)", value=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Acciones</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    refresh = col1.button("Actualizar")
    retrain = col2.button("Re-entrenar")

    if refresh:
        st.cache_data.clear()
        st.rerun()
    if retrain:
        with st.spinner("Ejecutando pipeline completo..."):
            from pipeline_etl import run_full_pipeline
            result = run_full_pipeline(retrain=True)
        st.success("Re-entrenamiento completado")
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Exo 2',sans-serif;font-size:0.7rem;color:#2a6496;text-align:center;letter-spacing:0.05em;">
        Proyecto Final · DataWin<br>Rauli &amp; Stevens
    </div>""", unsafe_allow_html=True)

# CARGA y proeso de los datos

df = load_data()
subset = df[(df["symbol"] == symbol) & (df["interval"] == interval)].copy()

try:
    prediction = predict_next_price(subset, model_type=model_type)
    pred_str   = f"${prediction:,.2f}"
except Exception as exc:
    prediction = None
    pred_str   = "N/A"
    logger.warning(f"[dashboard] Prediccion fallida: {exc}")

# HEADER principal con el logo, nombre del equipo y badge de "LIVE PREDICTOR"

logo_path = "assets/datawin_logo.png"
if os.path.exists(logo_path):
    import base64
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_tag = f'<img src="data:image/png;base64,{logo_b64}" class="datawin-logo-img"/>'
else:
    logo_tag = '<div style="font-size:2.8rem;filter:drop-shadow(0 0 10px #0af0ff);">📊</div>'

st.markdown(f"""
<div class="datawin-header">
    {logo_tag}
    <div>
        <div class="datawin-team">DataWin</div>
        <div class="datawin-subtitle">Crypto Price Predictor · Rauli &amp; Stevens</div>
    </div>
    <div class="datawin-badge">LIVE PREDICTOR</div>
</div>
""", unsafe_allow_html=True)

# Metricas principales

live_price = get_latest_price(subset, symbol) if not subset.empty else 0.0
delta      = prediction - live_price if prediction else 0.0
delta_pct  = (delta / live_price * 100) if live_price else 0.0

st.markdown('<div class="section-title">Resumen de Mercado</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio Actual", f"${live_price:,.2f}")
col2.metric("Prediccion",    pred_str,
            delta=f"{delta:+,.2f} ({delta_pct:+.2f}%)" if prediction else None)
col3.metric("Intervalo",     interval)
col4.metric("Modelo",        model_type.upper())

# LAYOUT principal de la pagina: Grafico (ANCHO COMPLETO) + CHAT sin el scroll infinito

# Botón para mostrar/ocultar chat
col_main_full, col_toggle = st.columns([20, 1])
with col_toggle:
    if st.button("💬" if st.session_state.show_chat else "🤖", key="toggle_chat"):
        st.session_state.show_chat = not st.session_state.show_chat
        st.rerun()

# Columna principal: Grafico y metricas (ANCHO COMPLETO)
st.markdown('<div class="section-title">Grafico de Velas</div>', unsafe_allow_html=True)
if not subset.empty:
    fig = plot_crypto_candlestick(
        df_plot=subset,
        title=f"{symbol} / {interval} — Prediccion proxima vela: {pred_str}",
        prediction=prediction,
        show_rsi=show_rsi,
    )
    
    hover_colors = [
        "rgba(38, 166, 154, 0.9)" if c >= o else "rgba(239, 83, 80, 0.9)" 
        for o, c in zip(subset['open'], subset['close'])
    ]
    
    if len(fig.data) > 0:
        fig.data[0].hoverlabel.bgcolor = hover_colors
        fig.data[0].hoverlabel.font.color = "white"

    fig.update_layout(
        margin=dict(t=50, b=100, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='#0a1a2f',
        paper_bgcolor='#0a1a2f'
    )

    fig.update_xaxes(title_text="Fecha y Hora", title_standoff=40)
    fig.update_yaxes(title_text="Precio", title_standoff=40)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"Sin datos para {symbol} / {interval}. Ejecuta el pipeline.")

# MÉTRICAS DEL MODELO
st.markdown('<div class="section-title">Metricas del Modelo</div>', unsafe_allow_html=True)
mae_value = None
rmse_value = None
r2_value = None

try:
    model = joblib.load(MODEL_RIDGE)
    with open(FEATURE_COLS_PATH, "rb") as f:
        cols = pickle.load(f)

    X = subset[cols].select_dtypes(include="number").replace(
            [float("inf"), float("-inf")], float("nan")).dropna()
    y = subset.loc[X.index, "target"].dropna()
    X = X.loc[y.index]

    if not X.empty:
        y_pred = model.predict(X)
        mae_value = mean_absolute_error(y, y_pred)
        rmse_value = float(np.sqrt(mean_squared_error(y, y_pred)))
        r2_value = r2_score(y, y_pred)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"${mae_value:,.2f}")
        m2.metric("RMSE", f"${rmse_value:,.2f}")
        m3.metric("R2",   f"{r2_value:.4f}")
    else:
        st.info("Datos insuficientes para calcular metricas.")

except FileNotFoundError:
    st.info("Modelo no encontrado. Ejecuta el pipeline para entrenar.")
except Exception as exc:
    st.warning(f"No se pudieron calcular metricas: {exc}")

# TABLA DE DATOS
with st.expander("Ver datos recientes"):
    cols_show = ["timestamp", "open", "high", "low", "close", "volume", "rsi_14", "sma_7", "sma_25"]
    cols_show = [c for c in cols_show if c in subset.columns]
    st.dataframe(subset[cols_show].tail(50).sort_values("timestamp", ascending=False), use_container_width=True)

# Chat fijo lateral cuando (este activo)

if st.session_state.show_chat:
    # Construir el contexto necesaria de la página para el chatbot
    page_context = f"""CONTEXTO DE LA PÁGINA ACTUAL:

Criptomoneda: {symbol}
Intervalo: {interval}
Modelo: {model_type}

MÉTRICAS DEL MODELO:
- MAE (Error Medio Absoluto): ${mae_value:,.2f} si mae_value else "No calculado"
- RMSE (Raíz del Error Cuadrático Medio): ${rmse_value:,.2f} si rmse_value else "No calculado"
- R² (Coeficiente de Determinación): {r2_value:.4f} si r2_value else "No calculado"

DATOS DE MERCADO:
- Precio Actual: ${live_price:,.2f}
- Predicción Próxima Vela: {pred_str}
- Cambio Predicho: {delta:+,.2f} ({delta_pct:+.2f}%)
- Mostrar RSI: {"Sí" if show_rsi else "No"}

Total de datos disponibles: {len(subset)} registros

El usuario está viendo estos datos en el dashboard. Cuando pregunte por "resultados" o las métricas, refierete a estos números específicos."""
    
    # Y aqui renderizamos el chat fijo
    render_chat_interface(page_context)