import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.optimize as sco
from datetime import datetime, timedelta
from scipy.stats import norm

# 1. CONFIGURACIÓN E INTERFAZ
st.set_page_config(page_title="Smart Investing Pro", layout="wide")
st.markdown("""<style>.stApp { background-color: #050505; color: #E0E0E0; } 
    h1, h2, h3 { color: #BA68C8; } .stMetric { background-color: #121212; padding: 15px; border-radius: 10px; border: 1px solid #4A148C; }</style>""", unsafe_allow_html=True)

@st.cache_data
def get_data(tickers, benchmark):
    all_symbols = list(set(tickers + [benchmark]))
    data_list = []
    
    for s in all_symbols:
        try:
            # Descargamos el ticker individualmente
            df_raw = yf.download(s, period="5y", progress=False)
            if df_raw.empty:
                continue
            
            # ELIMINAR MULTIINDEX: Forzamos a que las columnas sean nombres simples (Open, Close, etc.)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)
            
            # Buscamos 'Adj Close' o 'Close' de forma defensiva
            col_name = "Adj Close" if "Adj Close" in df_raw.columns else "Close"
            
            # Extraemos la columna y la renombramos al ticker para que la tabla sea limpia
            temp = df_raw[[col_name]].copy()
            temp.columns = [s]
            data_list.append(temp)
        except Exception:
            continue
            
    if not data_list:
        return pd.DataFrame()
        
    # Unimos todos en una sola tabla: Filas = Fechas, Columnas = Tickers
    final_df = pd.concat(data_list, axis=1)
    return final_df.dropna(how='all')

st.title("📈 Smart Investing Dashboard")
ticker_input = st.sidebar.text_input("Tickers (separados por coma):", "AAPL, MSFT, GOOGL, AMZN")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
benchmark = "^GSPC"

# 2. PROCESAMIENTO DE DATOS
data_raw = get_data(tickers, benchmark)

# Verificamos qué tickers se descargaron realmente para no pedir columnas que no existen
valid_tickers = [t for t in tickers if t in data_raw.columns]

if data_raw.empty or len(valid_tickers) == 0:
    st.error("No se pudieron obtener datos. Verifica los símbolos o la conexión con Yahoo Finance.")
    st.stop()

# Usamos solo los datos de los tickers que sí bajaron
data = data_raw[valid_tickers]

# 3. VISUALIZACIÓN - EVOLUCIÓN
st.header("1. Evolución de Precios")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Rendimiento Acumulado (Base 100)")
    # Base 100: (Precio Actual / Primer Precio) * 100
    fig_all = px.line((data / data.iloc[0]) * 100, template="plotly_dark", color_discrete_sequence=px.colors.sequential.Purples_r)
    fig_all.update_layout(showlegend=True, legend_title="Activos")
    st.plotly_chart(fig_all, use_container_width=True)
with c2:
    sel = st.selectbox("Analizar Activo Individual:", valid_tickers)
    fig_ind = px.line(data[sel], template="plotly_dark", color_discrete_sequence=['#BA68C8'])
    st.plotly_chart(fig_ind, use_container_width=True)

# 4. MÉTRICAS Y RIESGO
returns = np.log(data / data.shift(1)).dropna()
st.header("2. Análisis de Riesgo y Retorno")
st.dataframe(pd.DataFrame({
    'Retorno Anualizado': returns.mean() * 252,
    'Volatilidad (Std)': returns.std() * np.sqrt(252),
    'Máximo Histórico': data.max(),
    'Mínimo Histórico': data.min()
}).T, use_container_width=True)

# 5. OPTIMIZACIÓN DE PORTAFOLIO
st.header("3. Optimización (Max Sharpe Ratio)")
def get_perf(w):
    p_ret = np.sum(returns.mean() * w) * 252
    p_vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
    return p_ret, p_vol

# Maximizar Sharpe = Minimizar -(Retorno - RiskFree) / Volatilidad
def min_func_sharpe(w):
    p_ret, p_vol = get_perf(w)
    return -(p_ret - 0.04) / p_vol

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 0.5) for _ in range(len(valid_tickers)))
init_guess = len(valid_tickers) * [1. / len(valid_tickers)]

opt_res = sco.minimize(min_func_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
weights = opt_res.x

c_p1, c_p2 = st.columns([2, 1])
with c_p1:
    fig_pie = px.pie(names=valid_tickers, values=weights, hole=0.4, title="Distribución Óptima")
    st.plotly_chart(fig_pie, use_container_width=True)
with c_p2:
    st.write("### Pesos Asignados")
    st.table(pd.DataFrame({'Activo': valid_tickers, 'Peso': weights}).style.format({'Peso': '{:.2%}'}))

# 6. BACKTESTING VS BENCHMARK
st.header("4. Comparativa vs Benchmark (S&P 500)")
if benchmark in data_raw.columns:
    port_returns = (returns * weights).sum(axis=1)
    bench_returns = np.log(data_raw[benchmark] / data_raw[benchmark].shift(1)).dropna()
    
    df_compare = pd.DataFrame({
        'Mi Portafolio': (1 + port_returns).cumprod(),
        'Benchmark (^GSPC)': (1 + bench_returns).cumprod()
    }).dropna()
    
    fig_bench = px.line(df_compare, template="plotly_dark", title="Crecimiento de $1 invertido")
    st.plotly_chart(fig_bench, use_container_width=True)

# 7. MONTE CARLO (ESTRELLA FINAL)
st.header("5. Simulación Monte Carlo (Proyección 1 Año)")
mu, sigma = get_perf(weights)
sim_returns = np.random.normal(mu/252, sigma/np.sqrt(252), (252, 1000))
price_sim = np.cumprod(1 + sim_returns, axis=0)

fig_mc = px.line(price_sim[:, :50], template="plotly_dark", title="50 Trayectorias Posibles")
fig_mc.update_layout(showlegend=False)
st.plotly_chart(fig_mc, use_container_width=True)

# Valor en Riesgo (VaR)
var_95 = np.percentile(price_sim[-1], 5)
st.metric("Valor en Riesgo (VaR 95%)", f"{1 - var_95:.2%}", delta="Pérdida máxima esperada", delta_color="inverse")
