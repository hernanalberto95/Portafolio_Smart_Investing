import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.optimize as sco
from datetime import datetime, timedelta
from scipy.stats import norm

# --- CONFIGURACIÓN ESTÉTICA ---
st.set_page_config(page_title="Dashboard Pro - Smart Investing", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #E0E0E0; } 
    h1, h2, h3 { color: #BA68C8; text-shadow: 2px 2px 4px #000000; }
    .stMetric { background-color: #121212; padding: 15px; border-radius: 10px; border: 1px solid #4A148C; }
    .stDataFrame { border: 1px solid #4A148C; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIÓN DE CARGA DE DATOS (BLINDADA) ---
@st.cache_data
def get_data(tickers, benchmark):
    all_symbols = list(set(tickers + [benchmark]))
    data_list = []
    for s in all_symbols:
        try:
            df_raw = yf.download(s, period="5y", progress=False)
            if df_raw.empty: continue
            
            # Limpieza de MultiIndex (causa del KeyError en la nube)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)
            
            col = "Adj Close" if "Adj Close" in df_raw.columns else "Close"
            temp = df_raw[[col]].copy()
            temp.columns = [s]
            data_list.append(temp)
        except: continue
            
    if not data_list: return pd.DataFrame()
    return pd.concat(data_list, axis=1).dropna(how='all')

# --- SIDEBAR ---
st.sidebar.header("Configuración de Cartera")
ticker_input = st.sidebar.text_input("Ingresa Tickers (separados por coma):", "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
benchmark = "^GSPC"
rf_rate = 0.04 # Tasa libre de riesgo (4%)

# --- LÓGICA PRINCIPAL ---
st.title("📈 Dashboard Financiero Pro")
data_full = get_data(tickers, benchmark)
valid_tickers = [t for t in tickers if t in data_full.columns]

if not valid_tickers:
    st.error("No se pudieron cargar los tickers. Verifica la conexión o los símbolos.")
    st.stop()

data = data_full[valid_tickers]
returns = np.log(data / data.shift(1)).dropna()

# --- 1. EVOLUCIÓN DE PRECIOS ---
st.header("1. Evolución y Rendimiento")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Crecimiento Acumulado (Base 100)")
    fig_norm = px.line((data / data.iloc[0]) * 100, template="plotly_dark", color_discrete_sequence=px.colors.sequential.Purples_r)
    st.plotly_chart(fig_norm, use_container_width=True)
with col2:
    sel = st.selectbox("Analizar Activo:", valid_tickers)
    fig_ind = px.area(data[sel], template="plotly_dark", color_discrete_sequence=['#BA68C8'])
    st.plotly_chart(fig_ind, use_container_width=True)

# --- 2. MÉTRICAS Y CORRELACIÓN ---
st.header("2. Métricas de Riesgo-Retorno")
metrics_df = pd.DataFrame({
    'Retorno Anual': returns.mean() * 252,
    'Volatilidad Anual': returns.std() * np.sqrt(252),
    'Sharpe Ratio': (returns.mean() * 252 - rf_rate) / (returns.std() * np.sqrt(252))
}).T
st.dataframe(metrics_df.style.format("{:.2%}"), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Matriz de Correlación")
    fig_corr = px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="Purples", template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)
with c4:
    st.subheader("Matriz de Covarianza (Anualizada)")
    fig_cov = px.imshow(returns.cov() * 252, text_auto=".4f", color_continuous_scale="Purples", template="plotly_dark")
    st.plotly_chart(fig_cov, use_container_width=True)

# --- 3. OPTIMIZACIÓN (MAX SHARPE) ---
st.header("3. Portafolio Óptimo")
def get_p_stats(w):
    p_ret = np.sum(returns.mean() * w) * 252
    p_vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
    return p_ret, p_vol

def min_sharpe(w): return -(get_p_stats(w)[0] - rf_rate) / get_p_stats(w)[1]

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 0.4) for _ in range(len(valid_tickers))) # Máximo 40% por activo
opt = sco.minimize(min_sharpe, len(valid_tickers)*[1./len(valid_tickers)], method='SLSQP', bounds=bnds, constraints=cons)

c_pie1, c_pie2 = st.columns([2, 1])
with c_pie1:
    fig_pie = px.pie(names=valid_tickers, values=opt.x, hole=0.4, color_discrete_sequence=px.colors.sequential.Purples_r)
    st.plotly_chart(fig_pie, use_container_width=True)
with c_pie2:
    p_ret, p_vol = get_p_stats(opt.x)
    st.metric("Retorno Esperado", f"{p_ret:.2%}")
    st.metric("Volatilidad Esperada", f"{p_vol:.2%}")
    st.metric("Sharpe Ratio", f"{(p_ret - rf_rate)/p_vol:.2f}")

# --- 4. BACKTESTING Y ROLLING SHARPE ---
st.header("4. Backtesting vs Benchmark")
port_rets = (returns * opt.x).sum(axis=1)
bench_rets = np.log(data_full[benchmark] / data_full[benchmark].shift(1)).dropna()

df_back = pd.DataFrame({
    'Portafolio': (1 + port_rets).cumprod(),
    'Benchmark (S&P500)': (1 + bench_rets).cumprod()
}).dropna()

c5, c6 = st.columns(2)
with c5:
    st.plotly_chart(px.line(df_back, template="plotly_dark", title="Rendimiento Histórico ($1 inv.)"), use_container_width=True)
with c6:
    # Rolling Sharpe (6 meses = 126 días)
    rolling_sharpe = ((port_rets.rolling(126).mean() * 252) - rf_rate) / (port_rets.rolling(126).std() * np.sqrt(252))
    st.plotly_chart(px.line(rolling_sharpe, template="plotly_dark", title="Rolling Sharpe Ratio (6 meses)", color_discrete_sequence=['#CE93D8']), use_container_width=True)

# --- 5. ANÁLISIS DE RIESGO AVANZADO ---
st.header("5. Análisis de Riesgo (VaR)")
conf_level = st.select_slider("Nivel de Confianza:", options=[90, 95, 99], value=95)
alpha = (100 - conf_level) / 100

# VaR Histórico
var_hist = np.percentile(port_rets, alpha * 100)
# VaR Paramétrico
var_param = norm.ppf(alpha, port_rets.mean(), port_rets.std())

cv1, cv2, cv3 = st.columns(3)
cv1.metric(f"VaR Histórico ({conf_level}%)", f"{abs(var_hist):.2%}")
cv2.metric(f"VaR Paramétrico ({conf_level}%)", f"{abs(var_param):.2%}")
cv3.metric("Max Drawdown", f"{((df_back['Portafolio'].cummax() - df_back['Portafolio']) / df_back['Portafolio'].cummax()).max():.2%}")

# Monte Carlo (Proyección)
st.subheader("Simulación Monte Carlo (252 días)")
mc_sims = 500
sim_rets = np.random.normal(p_ret/252, p_vol/np.sqrt(252), (252, mc_sims))
sim_prices = np.cumprod(1 + sim_rets, axis=0)
fig_mc = px.line(sim_prices[:, :50], template="plotly_dark") # Mostramos 50 rutas
fig_mc.update_layout(showlegend=False)
st.plotly_chart(fig_mc, use_container_width=True)
